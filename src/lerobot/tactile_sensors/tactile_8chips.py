# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
自研触觉传感器 — 8-chip 版（单串口，左右夹爪共用一路）

硬件规格:
  - 8 片 AD7606（8 通道 ADC），单串口
  - 波特率: 2000000
  - 左侧: AD1,AD2,AD5,AD6 → 数据流 index 0,1,4,5
  - 右侧: AD3,AD4,AD7,AD8 → 数据流 index 2,3,6,7
  - 每侧输出: (4, 4, 6) float32，即 4 芯片 × 4 行 × 6 列

帧格式 (每行一帧):
  AA 55 | row_id(0-5) | 8×8×2 bytes payload (little-endian uint16) | XOR checksum
  payload 大小 = 8 chips × 8 channels × 2 bytes = 128 bytes

通道说明:
  - ch0-5: 触觉列（固件已做翻转，flip_cols=True 可还原物理顺序）
  - ch6-7: 未使用
  - row 0-3: 物理传感器有效行；row 4-5: 无传感器（舍弃）
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import serial

from .tactile_sensor import TactileConfig

logger = logging.getLogger(__name__)

# AD7606 sends 8 ADC channels per chip per scan line; ch0-5 are tactile, ch6-7 unused.
_ACTIVE_COLS = 6
# Firmware scans row_count=6 lines per frame; physical sensors occupy only the first 4.
_ACTIVE_ROWS = 4

# Chip indices in the serial data stream (AD1→idx0 … AD8→idx7):
#   Left side:  AD1,AD2,AD5,AD6 → indices 0,1,4,5
#   Right side: AD3,AD4,AD7,AD8 → indices 2,3,6,7
_LEFT_CHIP_IDX = [0, 1, 4, 5]
_RIGHT_CHIP_IDX = [2, 3, 6, 7]


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
    channels_per_chip: int = 8  # AD7606: 8 channels per chip per row; ch0-5 are tactile
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

        # payload = chips_total * channels_per_chip * 2 = 8*8*2 = 128 bytes per row
        self.row_width_total = config.chips_total * config.channels_per_chip
        self.frame_size = 2 + 1 + self.row_width_total * 2 + 1
        self._combined_matrix = np.zeros((config.row_count, self.row_width_total), dtype=np.uint16)
        self._rows_seen: set[int] = set()
        # Processed results stored as (4, 4, 6) float32: (chip, row, col)
        self._left_matrix = np.zeros((4, _ACTIVE_ROWS, _ACTIVE_COLS), dtype=np.float32)
        self._right_matrix = np.zeros((4, _ACTIVE_ROWS, _ACTIVE_COLS), dtype=np.float32)
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

    def _split_sides(self, combined: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert raw (6, 64) uint16 → two oriented (4, 4, 6) float32 arrays.

        Steps:
          reshape(6, 8, 8)                 → (scan_row, chip, channel)
          [:_ACTIVE_ROWS, :, :_ACTIVE_COLS] → (4, 8, 6) as (row, chip, col)
          fancy-index chips per side       → (4, 4, 6) as (row, chip, col)
          transpose(1, 0, 2)               → (chip, row, col) = (4, 4, 6)
          flip_rows / flip_cols            → spatial orientation correction
        """
        m = combined.reshape(6, 8, 8)[:_ACTIVE_ROWS, :, :_ACTIVE_COLS]  # (4, 8, 6): row,chip,col

        left  = m[:, _LEFT_CHIP_IDX,  :].transpose(1, 0, 2).astype(np.float32)  # (4,4,6): chip,row,col
        right = m[:, _RIGHT_CHIP_IDX, :].transpose(1, 0, 2).astype(np.float32)

        if self.config.flip_left_rows:
            left = left[:, ::-1, :]
        if self.config.flip_left_cols:
            left = left[:, :, ::-1]
        if self.config.flip_right_rows:
            right = right[:, ::-1, :]
        if self.config.flip_right_cols:
            right = right[:, :, ::-1]

        return np.ascontiguousarray(left), np.ascontiguousarray(right)

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
        return {
            "tactile_left": left,
            "tactile_right": right,
        }

    @staticmethod
    def get_feature_types() -> dict:
        return {
            "tactile_left": (4, _ACTIVE_ROWS, _ACTIVE_COLS),   # (4, 4, 6): chip, row, col
            "tactile_right": (4, _ACTIVE_ROWS, _ACTIVE_COLS),
        }
