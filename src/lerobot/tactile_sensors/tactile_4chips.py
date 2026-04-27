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


@TactileConfig.register_subclass("4chips")
@dataclass
class Tactile4ChipConfig(TactileConfig):
    """Two serial ports, one 4-chip tactile board per gripper side."""

    left_port: str = "/dev/tactile_4chips_left"
    right_port: str = "/dev/tactile_4chips_right"
    baudrate: int = 115200
    timeout: float = 0.1
    header1: int = 0xAA
    header2: int = 0x55
    row_count: int = 6
    chips_per_side: int = 4
    channels_per_chip: int = 6
    max_force_sum: float = 150000.0
    flip_left_rows: bool = False
    flip_left_cols: bool = False
    flip_right_rows: bool = False
    flip_right_cols: bool = False


class _FourChipPortReader:
    def __init__(self, port: str, config: Tactile4ChipConfig, side_name: str):
        self.port = port
        self.config = config
        self.side_name = side_name
        self.serial: Optional[serial.Serial] = None
        self.row_width = config.chips_per_side * config.channels_per_chip
        self.frame_size = 2 + 1 + self.row_width * 2 + 1
        self.matrix = np.zeros((config.row_count, self.row_width), dtype=np.uint16)
        self._rows_seen: set[int] = set()

    def connect(self) -> None:
        self.serial = serial.Serial(
            port=self.port,
            baudrate=self.config.baudrate,
            timeout=self.config.timeout,
        )
        self.serial.reset_input_buffer()
        logger.info("%s tactile port connected on %s", self.side_name, self.port)

    def disconnect(self) -> None:
        if self.serial is not None:
            self.serial.close()
            self.serial = None

    def _read_frame(self) -> tuple[int, np.ndarray]:
        if self.serial is None or not self.serial.is_open:
            raise RuntimeError(f"{self.side_name} tactile serial is not open")

        while True:
            first = _read_exact(self.serial, 1)[0]
            if first != self.config.header1:
                continue
            second = _read_exact(self.serial, 1)[0]
            if second != self.config.header2:
                continue
            break

        row_id = _read_exact(self.serial, 1)[0]
        if row_id >= self.config.row_count:
            raise ValueError(f"Invalid row id {row_id} on {self.side_name}")

        payload = _read_exact(self.serial, self.row_width * 2)
        checksum = _read_exact(self.serial, 1)[0]
        if _xor_checksum(payload) != checksum:
            raise ValueError(f"Checksum mismatch on {self.side_name} row {row_id}")

        row = np.frombuffer(payload, dtype="<u2").astype(np.uint16).copy()
        return row_id, row

    def read_matrix(self) -> Optional[np.ndarray]:
        row_id, row = self._read_frame()
        self.matrix[row_id] = row
        self._rows_seen.add(row_id)
        if len(self._rows_seen) < self.config.row_count:
            return None
        self._rows_seen.clear()
        return self.matrix.copy()


class Tactile4ChipSensor:
    """Passive serial reader for two 4-chip tactile boards."""

    def __init__(self, config: Tactile4ChipConfig):
        self.config = config
        self.left_reader = _FourChipPortReader(config.left_port, config, "left")
        self.right_reader = _FourChipPortReader(config.right_port, config, "right")
        self._is_connected = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        width = config.chips_per_side * config.channels_per_chip
        self._left_matrix = np.zeros((config.row_count, width), dtype=np.uint16)
        self._right_matrix = np.zeros((config.row_count, width), dtype=np.uint16)
        self._last_read_time = 0.0

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        if self._is_connected:
            return
        self.left_reader.connect()
        self.right_reader.connect()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self._is_connected = True

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.left_reader.disconnect()
        self.right_reader.disconnect()
        self._is_connected = False

    def _apply_orientation(self, matrix: np.ndarray, *, flip_rows: bool, flip_cols: bool) -> np.ndarray:
        if flip_rows:
            matrix = matrix[::-1, :]
        if flip_cols:
            matrix = matrix[:, ::-1]
        return matrix

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                left = self.left_reader.read_matrix()
                right = self.right_reader.read_matrix()
                if left is None or right is None:
                    continue

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

                with self._lock:
                    self._left_matrix = left
                    self._right_matrix = right
                    self._last_read_time = time.time()
            except Exception as e:
                logger.debug("4-chip tactile read error: %s", e)
                time.sleep(0.01)

    def get_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._left_matrix.copy(), self._right_matrix.copy()

    def get_observation(self) -> dict:
        left, right = self.get_matrices()
        # (6, 24) -> (4, 6, 6): 每芯片 6 列，4 芯片，6 行
        left = left.reshape(6, 4, 6).transpose(1, 0, 2).astype(np.float32)
        right = right.reshape(6, 4, 6).transpose(1, 0, 2).astype(np.float32)
        return {
            "tactile_left": left,
            "tactile_right": right,
        }

    @staticmethod
    def get_feature_types() -> dict:
        return {
            "tactile_left": (4, 6, 6),
            "tactile_right": (4, 6, 6),
        }
