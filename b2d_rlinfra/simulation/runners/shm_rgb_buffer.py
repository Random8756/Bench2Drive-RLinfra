"""Per-worker RGB shared-memory buffer backed by /dev/shm."""

import mmap
import os
import threading
import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np

__layer__ = (3, "Simulation")


class ShmRgbBuffer:
    """Fixed-slot RGB shared-memory buffer for one CARLA worker."""

    _SEQ_DTYPE = np.dtype(np.int64)

    def __init__(
        self,
        worker_id: int,
        rgb_shape: Tuple[int, ...],
        rgb_dtype: Any = np.uint8,
        prefix: Optional[str] = None,
        num_slots: int = 2,
        create: bool = True,
    ):
        self.worker_id = int(worker_id)
        self.prefix = prefix or "carla_rgb_{}".format(uuid.uuid4().hex[:12])
        self.num_slots = int(num_slots)
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")

        self.rgb_shape = tuple(int(dim) for dim in rgb_shape)
        self.rgb_dtype = np.dtype(rgb_dtype)
        self.shape = (self.num_slots,) + self.rgb_shape
        self.name = "{}_w{}_rgb".format(self.prefix, self.worker_id)
        self.path = "/dev/shm/{}".format(self.name)

        self._lock = threading.RLock()
        self._closed = False
        self._seq_nbytes = int(self.num_slots * self._SEQ_DTYPE.itemsize)
        self._rgb_nbytes = int(np.prod(self.shape) * self.rgb_dtype.itemsize)
        self.nbytes = self._seq_nbytes + self._rgb_nbytes

        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        self.fd = os.open(self.path, flags, 0o600)
        try:
            if create:
                os.ftruncate(self.fd, self.nbytes)
            self.mm = mmap.mmap(self.fd, self.nbytes)
            self._seq_array = np.ndarray(
                (self.num_slots,),
                dtype=self._SEQ_DTYPE,
                buffer=self.mm,
                offset=0,
            )
            self.array = np.ndarray(
                self.shape,
                dtype=self.rgb_dtype,
                buffer=self.mm,
                offset=self._seq_nbytes,
            )
            if create:
                self._seq_array[:] = -1
                self.array[:] = 0
        except Exception:
            try:
                os.close(self.fd)
            except OSError:
                pass
            raise

    def _ensure_open(self) -> None:
        if self._closed or self.array is None or self._seq_array is None:
            raise RuntimeError("RGB shm buffer is closed: {}".format(self.path))

    def _validate_slot(self, slot: int) -> int:
        slot = int(slot)
        if slot < 0 or slot >= self.num_slots:
            raise IndexError(
                "RGB shm slot {} out of range [0, {}) for {}".format(
                    slot,
                    self.num_slots,
                    self.path,
                )
            )
        return slot

    def write_rgb(self, slot: int, rgb: Any, seq: int) -> None:
        slot = self._validate_slot(slot)
        seq = int(seq)
        with self._lock:
            self._ensure_open()
            self._seq_array[slot] = -seq - 2
            np.copyto(self.array[slot], rgb, casting="safe")
            self._seq_array[slot] = seq

    def read_rgb(self, slot: int, expected_seq: Optional[int] = None) -> np.ndarray:
        slot = self._validate_slot(slot)
        with self._lock:
            self._ensure_open()
            if expected_seq is not None:
                seq_before = int(self._seq_array[slot])
                if seq_before != int(expected_seq):
                    raise RuntimeError(
                        "RGB shm seq mismatch before read for {} slot {}: "
                        "expected {}, found {}".format(
                            self.path,
                            slot,
                            int(expected_seq),
                            seq_before,
                        )
                    )
            rgb = np.array(self.array[slot], copy=True)
            if expected_seq is not None:
                seq_after = int(self._seq_array[slot])
                if seq_after != int(expected_seq):
                    raise RuntimeError(
                        "RGB shm seq mismatch after read for {} slot {}: "
                        "expected {}, found {}".format(
                            self.path,
                            slot,
                            int(expected_seq),
                            seq_after,
                        )
                    )
            return rgb

    def read_seq(self, slot: int) -> int:
        slot = self._validate_slot(slot)
        with self._lock:
            self._ensure_open()
            return int(self._seq_array[slot])

    def metadata(self, rgb_obs_key: str) -> Dict[str, Any]:
        return {
            "enabled": True,
            "worker_id": self.worker_id,
            "prefix": self.prefix,
            "name": self.name,
            "path": self.path,
            "rgb_shape": self.rgb_shape,
            "rgb_dtype": self.rgb_dtype.str,
            "num_slots": self.num_slots,
            "rgb_obs_key": rgb_obs_key,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.array = None
            self._seq_array = None
            try:
                self.mm.close()
            finally:
                try:
                    os.close(self.fd)
                finally:
                    self._closed = True

    def unlink(self) -> None:
        with self._lock:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
