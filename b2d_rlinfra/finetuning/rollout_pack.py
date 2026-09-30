"""Versioned, mmap-friendly single-episode rollout container."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import threading
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from b2d_rlinfra.finetuning.coordination import fsync_parent_best_effort

FORMAT_NAME = "b2d-rollout-pack"
FORMAT_VERSION = 1
FIELD_ALIGNMENT = 4096
PREAMBLE_MAGIC = b"B2DRLPK1"
FOOTER_MAGIC = b"B2DRLFT1"
PREAMBLE_STRUCT = struct.Struct("<8sII")
FOOTER_STRUCT = struct.Struct("<8sQQIQ")
_ALLOWED_DTYPE_KINDS = frozenset("biuf")


class RolloutPackFormatError(ValueError):
    """Raised when a RolloutPack is malformed or unsupported."""


def _align(value: int, alignment: int = FIELD_ALIGNMENT) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"RolloutPack metadata is not JSON-serializable: {type(value).__name__}")


def _flatten_tree(value: Any, path: Tuple[str, ...]) -> List[Tuple[Tuple[str, ...], np.ndarray]]:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"RolloutPack sample tree at {'.'.join(path)!r} is empty")
        flattened: List[Tuple[Tuple[str, ...], np.ndarray]] = []
        for key in sorted(value, key=lambda item: str(item)):
            flattened.extend(_flatten_tree(value[key], path + (str(key),)))
        return flattened
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"RolloutPack sample field {'.'.join(path)!r} has no sample dimension")
    if array.dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise TypeError(
            f"RolloutPack sample field {'.'.join(path)!r} has unsupported dtype {array.dtype}; "
            "only bool/int/uint/float arrays are supported"
        )
    return [(path, np.ascontiguousarray(array))]


def flatten_sample_fields(
    fields: Mapping[str, Any],
    *,
    num_steps: int,
) -> List[Tuple[Tuple[str, ...], np.ndarray]]:
    num_steps = int(num_steps)
    if num_steps <= 0:
        raise ValueError("RolloutPack num_steps must be positive")
    flattened: List[Tuple[Tuple[str, ...], np.ndarray]] = []
    for key in sorted(fields):
        flattened.extend(_flatten_tree(fields[key], (str(key),)))
    if not flattened:
        raise ValueError("RolloutPack must contain at least one sample field")
    seen = set()
    for path, array in flattened:
        if path in seen:
            raise ValueError(f"duplicate RolloutPack field path: {path}")
        seen.add(path)
        if int(array.shape[0]) != int(num_steps):
            raise ValueError(
                f"RolloutPack field {'.'.join(path)!r} length {array.shape[0]} != num_steps {num_steps}"
            )
    return flattened


def unflatten_sample_fields(flattened: Mapping[Tuple[str, ...], Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for path, value in flattened.items():
        if not path:
            raise ValueError("RolloutPack field path cannot be empty")
        cursor: Dict[str, Any] = result
        for component in path[:-1]:
            existing = cursor.get(component)
            if existing is None:
                nested: Dict[str, Any] = {}
                cursor[component] = nested
                cursor = nested
            elif isinstance(existing, dict):
                cursor = existing
            else:
                raise ValueError(f"RolloutPack field tree collision at {path}")
        leaf = path[-1]
        if leaf in cursor:
            raise ValueError(f"duplicate RolloutPack field path: {path}")
        cursor[leaf] = value
    return result


def schema_fingerprint(fields: Sequence[Mapping[str, Any]]) -> str:
    schema = [
        {
            "path": list(field["path"]),
            "dtype": str(field["dtype"]),
            "trailing_shape": list(field["shape"])[1:],
        }
        for field in fields
    ]
    encoded = json.dumps(schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RolloutPackWriter:
    """Write one complete episode as an atomic RolloutPack v1 file."""

    @classmethod
    def write(
        cls,
        path: str | Path,
        *,
        metadata: Mapping[str, Any],
        sample_fields: Mapping[str, Any],
        num_steps: int,
    ) -> Path:
        flattened = flatten_sample_fields(sample_fields, num_steps=int(num_steps))
        final_path = Path(path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_suffix(
            final_path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}"
        )
        field_entries: List[Dict[str, Any]] = []
        try:
            with open(tmp_path, "wb") as handle:
                handle.write(PREAMBLE_STRUCT.pack(PREAMBLE_MAGIC, FORMAT_VERSION, 0))
                for field_path, array in flattened:
                    offset = _align(handle.tell())
                    if offset > handle.tell():
                        handle.write(b"\0" * (offset - handle.tell()))
                    raw = memoryview(array).cast("B")
                    handle.write(raw)
                    field_entries.append(
                        {
                            "path": list(field_path),
                            "dtype": array.dtype.str,
                            "shape": [int(value) for value in array.shape],
                            "offset": int(offset),
                            "nbytes": int(array.nbytes),
                        }
                    )

                header_offset = int(handle.tell())
                header = {
                    "format": FORMAT_NAME,
                    "version": FORMAT_VERSION,
                    "num_steps": int(num_steps),
                    "metadata": _json_safe(dict(metadata)),
                    "fields": field_entries,
                    "schema_fingerprint": schema_fingerprint(field_entries),
                }
                header_bytes = json.dumps(
                    header,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                handle.write(header_bytes)
                expected_size = handle.tell() + FOOTER_STRUCT.size
                handle.write(
                    FOOTER_STRUCT.pack(
                        FOOTER_MAGIC,
                        header_offset,
                        len(header_bytes),
                        zlib.crc32(header_bytes) & 0xFFFFFFFF,
                        expected_size,
                    )
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, final_path)
            fsync_parent_best_effort(final_path.parent)
        except Exception:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise
        return final_path


class RolloutPackReader:
    """Validate and expose read-only ndarray views over a RolloutPack."""

    def __init__(self, path: str | Path, *, mmap_payload: bool = True):
        self.path = Path(path)
        self._file = None
        self._mmap = None
        self._arrays: Dict[Tuple[str, ...], np.ndarray] = {}
        self.header, self.header_crc32 = self.read_header(self.path)
        self.metadata = dict(self.header.get("metadata") or {})
        self.num_steps = int(self.header["num_steps"])
        self.fields = list(self.header["fields"])
        self.schema_fingerprint = str(self.header["schema_fingerprint"])
        if mmap_payload:
            self._file = open(self.path, "rb")
            self._mmap = mmap.mmap(self._file.fileno(), length=0, access=mmap.ACCESS_READ)
            for field in self.fields:
                field_path = tuple(str(item) for item in field["path"])
                self._arrays[field_path] = np.ndarray(
                    shape=tuple(int(item) for item in field["shape"]),
                    dtype=np.dtype(field["dtype"]),
                    buffer=self._mmap,
                    offset=int(field["offset"]),
                    order="C",
                )

    @staticmethod
    def read_header(path: str | Path) -> Tuple[Dict[str, Any], int]:
        source = Path(path)
        file_size = source.stat().st_size
        minimum_size = PREAMBLE_STRUCT.size + FOOTER_STRUCT.size
        if file_size < minimum_size:
            raise RolloutPackFormatError(f"RolloutPack is truncated: {source}")
        with open(source, "rb") as handle:
            preamble = handle.read(PREAMBLE_STRUCT.size)
            magic, version, flags = PREAMBLE_STRUCT.unpack(preamble)
            if magic != PREAMBLE_MAGIC:
                raise RolloutPackFormatError(f"invalid RolloutPack magic: {source}")
            if int(version) != FORMAT_VERSION:
                raise RolloutPackFormatError(
                    f"unsupported RolloutPack version={version}; expected {FORMAT_VERSION}"
                )
            if int(flags) != 0:
                raise RolloutPackFormatError(f"unsupported RolloutPack flags={flags}: {source}")
            handle.seek(file_size - FOOTER_STRUCT.size)
            footer = handle.read(FOOTER_STRUCT.size)
            footer_magic, header_offset, header_length, header_crc, expected_size = FOOTER_STRUCT.unpack(footer)
            if footer_magic != FOOTER_MAGIC:
                raise RolloutPackFormatError(f"invalid RolloutPack footer magic: {source}")
            if int(expected_size) != int(file_size):
                raise RolloutPackFormatError(
                    f"RolloutPack size mismatch: footer={expected_size} actual={file_size}"
                )
            header_end = int(header_offset) + int(header_length)
            if int(header_offset) < PREAMBLE_STRUCT.size or header_end != file_size - FOOTER_STRUCT.size:
                raise RolloutPackFormatError(f"invalid RolloutPack header bounds: {source}")
            handle.seek(int(header_offset))
            header_bytes = handle.read(int(header_length))
        actual_crc = zlib.crc32(header_bytes) & 0xFFFFFFFF
        if actual_crc != int(header_crc):
            raise RolloutPackFormatError(
                f"RolloutPack header CRC mismatch: expected={header_crc} actual={actual_crc}"
            )
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RolloutPackFormatError(f"invalid RolloutPack JSON header: {source}") from exc
        RolloutPackReader._validate_header(header, header_offset=int(header_offset), source=source)
        return dict(header), int(header_crc)

    @staticmethod
    def _validate_header(header: Mapping[str, Any], *, header_offset: int, source: Path) -> None:
        if header.get("format") != FORMAT_NAME or int(header.get("version", -1)) != FORMAT_VERSION:
            raise RolloutPackFormatError(f"invalid RolloutPack format header: {source}")
        num_steps = int(header.get("num_steps", 0))
        if num_steps <= 0:
            raise RolloutPackFormatError(f"RolloutPack num_steps must be positive: {source}")
        fields = header.get("fields")
        if not isinstance(fields, list) or not fields:
            raise RolloutPackFormatError(f"RolloutPack has no fields: {source}")
        paths = set()
        ranges = []
        for field in fields:
            if not isinstance(field, Mapping):
                raise RolloutPackFormatError(f"invalid RolloutPack field entry: {source}")
            path = tuple(str(item) for item in field.get("path", []))
            if not path or path in paths:
                raise RolloutPackFormatError(f"invalid or duplicate RolloutPack field path {path}: {source}")
            paths.add(path)
            try:
                dtype = np.dtype(field["dtype"])
                shape = tuple(int(item) for item in field["shape"])
                offset = int(field["offset"])
                nbytes = int(field["nbytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RolloutPackFormatError(f"invalid RolloutPack field metadata for {path}: {source}") from exc
            if (
                dtype.kind not in _ALLOWED_DTYPE_KINDS
                or not shape
                or shape[0] != num_steps
                or any(dimension < 0 for dimension in shape)
            ):
                raise RolloutPackFormatError(f"invalid RolloutPack field schema for {path}: {source}")
            expected_nbytes = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
            if nbytes != expected_nbytes or offset % FIELD_ALIGNMENT != 0:
                raise RolloutPackFormatError(f"invalid RolloutPack field size/alignment for {path}: {source}")
            if offset < PREAMBLE_STRUCT.size or offset + nbytes > int(header_offset):
                raise RolloutPackFormatError(f"RolloutPack field {path} is outside payload bounds: {source}")
            ranges.append((offset, offset + nbytes, path))
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if previous[1] > current[0]:
                raise RolloutPackFormatError(
                    f"RolloutPack fields overlap: {previous[2]} and {current[2]} in {source}"
                )
        expected_fingerprint = schema_fingerprint(fields)
        if str(header.get("schema_fingerprint", "")) != expected_fingerprint:
            raise RolloutPackFormatError(f"RolloutPack schema fingerprint mismatch: {source}")

    def array(self, path: Iterable[str]) -> np.ndarray:
        key = tuple(str(item) for item in path)
        try:
            return self._arrays[key]
        except KeyError as exc:
            raise KeyError(f"RolloutPack has no field path {key}: {self.path}") from exc

    def materialize(self) -> Dict[str, Any]:
        copied = {path: np.array(array, copy=True) for path, array in self._arrays.items()}
        return unflatten_sample_fields(copied)

    def close(self) -> None:
        self._arrays.clear()
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "RolloutPackReader":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
