from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from b2d_rlinfra.finetuning.rollout_file_store import (
    RolloutFileStore,
    write_update_index,
)
from b2d_rlinfra.finetuning.rollout_pack import (
    FIELD_ALIGNMENT,
    FOOTER_STRUCT,
    RolloutPackFormatError,
    RolloutPackReader,
    RolloutPackWriter,
)


def _episode(store: RolloutFileStore, episode_id: int, count: int, *, offset: int = 0):
    return store.write_episode(
        collector_id=0,
        episode_id=episode_id,
        policy_version=3,
        actions={
            "discrete": np.arange(offset, offset + count, dtype=np.int64),
            "continuous": np.arange(count * 2, dtype=np.float32).reshape(count, 2),
        },
        rewards=np.linspace(0.0, 1.0, count, dtype=np.float32),
        episode_starts=np.arange(count) == 0,
        values=np.zeros(count, dtype=np.float32),
        old_action_log_probs=np.zeros(count, dtype=np.float32),
        advantages=np.arange(offset, offset + count, dtype=np.float32),
        returns=np.ones(count, dtype=np.float32),
        policy_input_state={
            "image": np.zeros((count, 2, 3), dtype=np.uint8),
            "mask": np.ones((count, 2), dtype=np.bool_),
        },
        terminated=True,
        truncated=False,
        info={"route_completed_ratio": 1.0, "terminal": "ok"},
        optional_fields={"sample_weight": np.ones(count, dtype=np.float32), "tag": "test"},
    )


class RolloutPackTest(unittest.TestCase):
    def test_writer_rejects_empty_fields_and_non_positive_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "num_steps must be positive"):
                RolloutPackWriter.write(
                    root / "zero.rollout",
                    metadata={},
                    sample_fields={"value": np.empty((0,), dtype=np.float32)},
                    num_steps=0,
                )
            with self.assertRaisesRegex(ValueError, "at least one sample field"):
                RolloutPackWriter.write(
                    root / "empty.rollout",
                    metadata={},
                    sample_fields={},
                    num_steps=1,
                )
            self.assertEqual(list(root.glob("*.rollout")), [])

    def test_round_trips_all_supported_numeric_dtype_families(self) -> None:
        dtypes = (
            np.bool_,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
        )
        fields = {
            f"field_{index}": np.asarray([0, 1, 1], dtype=dtype)
            for index, dtype in enumerate(dtypes)
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = RolloutPackWriter.write(
                Path(tmp) / "dtypes.rollout",
                metadata={},
                sample_fields=fields,
                num_steps=3,
            )
            with RolloutPackReader(path) as reader:
                for index, dtype in enumerate(dtypes):
                    value = reader.array((f"field_{index}",))
                    self.assertEqual(value.dtype, np.dtype(dtype))
                    np.testing.assert_array_equal(value, fields[f"field_{index}"])

    def test_round_trip_nested_numeric_fields_and_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RolloutFileStore(tmp)
            meta = _episode(store, 2, 5)
            self.assertTrue(meta.file_path.endswith(".rollout"))
            with RolloutPackReader(meta.file_path) as reader:
                self.assertEqual(reader.num_steps, 5)
                self.assertEqual(reader.metadata["policy_version"], 3)
                self.assertEqual(reader.metadata["optional"]["tag"], "test")
                self.assertTrue(all(int(field["offset"]) % FIELD_ALIGNMENT == 0 for field in reader.fields))
                np.testing.assert_array_equal(
                    reader.array(("actions", "discrete")), np.arange(5, dtype=np.int64)
                )
                self.assertFalse(reader.array(("policy_input_state", "image")).flags.writeable)
                restored = reader.materialize()
            np.testing.assert_array_equal(restored["actions"]["discrete"], np.arange(5))

    def test_rejects_object_and_ragged_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.rollout"
            with self.assertRaises(TypeError):
                RolloutPackWriter.write(
                    path,
                    metadata={},
                    sample_fields={"bad": np.asarray([{"x": 1}], dtype=object)},
                    num_steps=1,
                )
            with self.assertRaises((TypeError, ValueError)):
                RolloutPackWriter.write(
                    path,
                    metadata={},
                    sample_fields={"bad": [[1], [2, 3]]},
                    num_steps=2,
                )
            with self.assertRaises(TypeError):
                RolloutPackWriter.write(
                    path,
                    metadata={},
                    sample_fields={"bad": np.ones(2, dtype=np.complex64)},
                    num_steps=2,
                )

            store = RolloutFileStore(str(Path(tmp) / "store"))
            with self.assertRaises((TypeError, ValueError)):
                store.write_episode(
                    collector_id=0,
                    episode_id=0,
                    policy_version=0,
                    actions=np.arange(2, dtype=np.int64),
                    rewards=np.zeros(2, dtype=np.float32),
                    episode_starts=np.asarray([True, False]),
                    values=np.zeros(2, dtype=np.float32),
                    old_action_log_probs=np.zeros(2, dtype=np.float32),
                    advantages=np.ones(2, dtype=np.float32),
                    returns=np.ones(2, dtype=np.float32),
                    policy_input_state=np.zeros((2, 1), dtype=np.float32),
                    terminated=True,
                    truncated=False,
                    optional_fields={"ragged": [[1], [2, 3]]},
                )
            self.assertEqual(list((Path(tmp) / "store").rglob("*.rollout")), [])

    def test_detects_header_crc_and_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RolloutFileStore(tmp)
            meta = _episode(store, 0, 3)
            header, _ = RolloutPackReader.read_header(meta.file_path)
            with open(meta.file_path, "r+b") as handle:
                handle.seek(-FOOTER_STRUCT.size - 1, os.SEEK_END)
                byte = handle.read(1)
                handle.seek(-1, os.SEEK_CUR)
                handle.write(bytes([byte[0] ^ 1]))
            with self.assertRaises(RolloutPackFormatError):
                RolloutPackReader.read_header(meta.file_path)

            truncated = Path(tmp) / "truncated.rollout"
            truncated.write_bytes(b"short")
            with self.assertRaises(RolloutPackFormatError):
                RolloutPackReader.read_header(truncated)
            self.assertEqual(header["num_steps"], 3)

    def test_atomic_publish_does_not_leave_final_file_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.rollout"
            with mock.patch(
                "b2d_rlinfra.finetuning.rollout_pack.os.replace",
                side_effect=OSError("injected replace failure"),
            ):
                with self.assertRaises(OSError):
                    RolloutPackWriter.write(
                        path,
                        metadata={},
                        sample_fields={"value": np.arange(3, dtype=np.float32)},
                        num_steps=3,
                    )
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).glob("*.tmp.*")), [])

    def test_update_plan_filter_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RolloutFileStore(str(root / "shared"))
            first = _episode(store, 0, 5, offset=0)
            second = _episode(store, 1, 7, offset=5)
            plan_path = write_update_index(
                str(root / "plan.json"),
                4,
                [first.file_path, second.file_path],
                sample_filter={"success_terminal_window_steps": 3},
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["version"], 2)
            self.assertEqual(plan["samples_before_filter"], 12)
            self.assertEqual(plan["samples_after_filter"], 6)
            self.assertEqual([entry["retained_start"] for entry in plan["entries"]], [2, 4])
            self.assertEqual([entry["global_stop"] for entry in plan["entries"]], [3, 6])
            self.assertEqual(
                [entry["source_path"] for entry in plan["entries"]],
                [first.file_path, second.file_path],
            )
            self.assertTrue(all("read_path" not in entry for entry in plan["entries"]))

            incompatible = root / "incompatible.rollout"
            RolloutPackWriter.write(
                incompatible,
                metadata={"outcome": "failure"},
                sample_fields={"different": np.zeros((2, 3), dtype=np.float32)},
                num_steps=2,
            )
            with self.assertRaisesRegex(ValueError, "schemas differ"):
                write_update_index(
                    str(root / "invalid_plan.json"),
                    5,
                    [first.file_path, str(incompatible)],
                )

if __name__ == "__main__":
    unittest.main()
