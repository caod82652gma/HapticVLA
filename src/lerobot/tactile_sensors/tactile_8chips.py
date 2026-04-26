import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import serial

from .tactile_sensor import TactileConfig

logger = logging.getLogger(__name__)


def _xor_checksum(data: bytes) -> int:
    checksum = 0
    for byte in data:
        checksum ^= byte
    return checksum


def _read_exact(ser: serial.Serial, size: int) -> bytes:
    data = ser.read(size)
    if len(data) != size:
        raise TimeoutError(f"Expected {size} bytes, got {len(data)}")
    return data


@TactileConfig.register_subclass("8chips")
@dataclass
class Tactile8ChipConfig(TactileConfig):
    """Single serial port with one 8-chip board covering both gripper sides."""

    port: str = "/dev/tactile_8chips"
    baudrate: int = 2000000
    timeout: float = 0.1
    header1: int = 0xAA
    header2: int = 0x55
    row_count: int = 6
    chips_total: int = 8
    chips_per_side: int = 4
    channels_per_chip: int = 8
    max_force_sum: float = 150000.0
    flip_left_rows: bool = False
    flip_left_cols: bool = False
    flip_right_rows: bool = False
    flip_right_cols: bool = False


class Tactile8ChipSensor:
    """Passive serial reader for one 8-chip tactile board."""

    def __init__(self, config: Tactile8ChipConfig):
        self.config = config
        self._serial: Optional[serial.Serial] = None
        self._is_connected = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.row_width_total = config.chips_total * config.channels_per_chip
        self.side_width = config.chips_per_side * config.channels_per_chip
        self.frame_size = 2 + 1 + self.row_width_total * 2 + 1
        self._combined_matrix = np.zeros((config.row_count, self.row_width_total), dtype=np.uint16)
        self._rows_seen: set[int] = set()
        self._left_matrix = np.zeros((config.row_count, self.side_width), dtype=np.uint16)
        self._right_matrix = np.zeros((config.row_count, self.side_width), dtype=np.uint16)
        self._last_read_time = 0.0

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        if self._is_connected:
            return
        self._serial = serial.Serial(
            port=self.config.port,
            baudrate=self.config.baudrate,
            timeout=self.config.timeout,
        )
        self._serial.reset_input_buffer()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self._is_connected = True
        logger.info("8-chip tactile sensor connected on %s", self.config.port)

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        self._is_connected = False

    def _read_frame(self) -> tuple[int, np.ndarray]:
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("8-chip tactile serial is not open")

        while True:
            first = _read_exact(self._serial, 1)[0]
            if first != self.config.header1:
                continue
            second = _read_exact(self._serial, 1)[0]
            if second != self.config.header2:
                continue
            break

        row_id = _read_exact(self._serial, 1)[0]
        if row_id >= self.config.row_count:
            raise ValueError(f"Invalid row id {row_id}")

        payload = _read_exact(self._serial, self.row_width_total * 2)
        checksum = _read_exact(self._serial, 1)[0]
        if _xor_checksum(payload) != checksum:
            raise ValueError(f"Checksum mismatch on row {row_id}")

        row = np.frombuffer(payload, dtype="<u2").astype(np.uint16).copy()
        return row_id, row

    def _apply_orientation(self, matrix: np.ndarray, *, flip_rows: bool, flip_cols: bool) -> np.ndarray:
        if flip_rows:
            matrix = matrix[::-1, :]
        if flip_cols:
            matrix = matrix[:, ::-1]
        return matrix

    def _split_sides(self, combined: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left = combined[:, : self.side_width]
        right = combined[:, self.side_width : self.side_width * 2]
        left = self._apply_orientation(
            left,
            flip_rows=self.config.flip_left_rows,
            flip_cols=self.config.flip_left_cols,
        )
        right = self._apply_orientation(
            right,
            flip_rows=self.config.flip_right_rows,
            flip_cols=self.config.flip_right_cols,
        )
        return left, right

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                row_id, row = self._read_frame()
                self._combined_matrix[row_id] = row
                self._rows_seen.add(row_id)
                if len(self._rows_seen) < self.config.row_count:
                    continue

                self._rows_seen.clear()
                left, right = self._split_sides(self._combined_matrix.copy())
                with self._lock:
                    self._left_matrix = left
                    self._right_matrix = right
                    self._last_read_time = time.time()
            except Exception as e:
                logger.debug("8-chip tactile read error: %s", e)
                time.sleep(0.01)

    def get_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._left_matrix.copy(), self._right_matrix.copy()

    def get_observation(self) -> dict:
        left, right = self.get_matrices()
        left_sum = min(float(np.sum(left)) / self.config.max_force_sum, 1.0)
        right_sum = min(float(np.sum(right)) / self.config.max_force_sum, 1.0)
        return {
            "tactile_left": left,
            "tactile_right": right,
            "tactile_left.sum": left_sum,
            "tactile_right.sum": right_sum,
        }

    @staticmethod
    def get_feature_types() -> dict:
        return {
            "tactile_left": (6, 32),
            "tactile_right": (6, 32),
            "tactile_left.sum": float,
            "tactile_right.sum": float,
        }
