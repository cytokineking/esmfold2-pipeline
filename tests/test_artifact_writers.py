from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from esmfold2_pipeline.artifacts import (
    ArtifactExistsError,
    FastaRecord,
    write_bytes_atomic,
    write_fasta,
    write_json_atomic,
    write_text_atomic,
)


class ArtifactWriterTest(unittest.TestCase):
    def test_write_text_atomic_creates_parents_and_writes_exact_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shards" / "shard_000000" / "worker.log"

            result = write_text_atomic(path, "started\ncompleted\n")

            self.assertEqual(result, path)
            self.assertEqual(path.read_text(), "started\ncompleted\n")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_overwrite_false_preserves_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidate.fasta"
            write_text_atomic(path, ">old\nAAAA\n")

            with self.assertRaises(ArtifactExistsError):
                write_text_atomic(path, ">new\nCCCC\n", overwrite=False)

            self.assertEqual(path.read_text(), ">old\nAAAA\n")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_failed_publish_removes_temp_file_and_leaves_no_final_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"

            with patch(
                "esmfold2_pipeline.artifacts.writers.os.replace",
                side_effect=RuntimeError("simulated publish failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "publish failure"):
                    write_json_atomic(path, {"iptm": 0.7})

            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            self.assertEqual(list(path.parent.glob(".metrics.json.*.tmp")), [])

    def test_write_json_atomic_uses_stable_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"

            write_json_atomic(path, {"z": 1, "a": {"b": 2}})

            self.assertEqual(json.loads(path.read_text()), {"a": {"b": 2}, "z": 1})
            self.assertTrue(path.read_text().endswith("\n"))
            self.assertLess(path.read_text().find('"a"'), path.read_text().find('"z"'))

    def test_write_bytes_atomic_handles_structure_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "complex.pdb"
            payload = b"ATOM      1  N   ALA A   1      0.000   0.000   0.000\n"

            write_bytes_atomic(path, payload)

            self.assertEqual(path.read_bytes(), payload)

    def test_write_fasta_formats_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sequence.fasta"

            write_fasta(
                path,
                [
                    FastaRecord(
                        identifier="cand_00000000",
                        sequence="acdefghiklmnp",
                        description="seed=0 critic=fast",
                    ),
                    ("cand_00000001", "QRSTVWY", "seed=1"),
                ],
                line_width=5,
            )

            self.assertEqual(
                path.read_text(),
                (
                    ">cand_00000000 seed=0 critic=fast\n"
                    "ACDEF\n"
                    "GHIKL\n"
                    "MNP\n"
                    ">cand_00000001 seed=1\n"
                    "QRSTV\n"
                    "WY\n"
                ),
            )

    def test_write_fasta_rejects_unsafe_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sequence.fasta"

            with self.assertRaisesRegex(ValueError, "identifier"):
                write_fasta(path, [("bad id", "ACD")])

            with self.assertRaisesRegex(ValueError, "sequence"):
                write_fasta(path, [("ok_id", "")])

            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()

