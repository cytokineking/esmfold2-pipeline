from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from esmfold2_pipeline.validation.target_sequences import (
    EffectiveTargetSequenceError,
    effective_target_sequences,
)


class EffectiveTargetSequencesTest(unittest.TestCase):
    def test_prepared_crop_is_authoritative_over_full_config_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_dir = root / "target"
            target_dir.mkdir()
            target_dir.joinpath("chain_summary.json").write_text(
                json.dumps(
                    {
                        "chains": [
                            {
                                "canonical_chain_id": "A",
                                "sequence": "A" * 150,
                                "length": 150,
                            }
                        ]
                    }
                )
            )

            sequences, labels = effective_target_sequences(
                root,
                {
                    "target": {
                        "chains": ["A"],
                        "crop": {"A": ["1-150"]},
                        "sequences": {"A": "A" * 449},
                    }
                },
            )

            self.assertEqual(sequences, ("A" * 150,))
            self.assertEqual(labels, ("A",))

    def test_prepared_multichain_crop_preserves_summary_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_dir = root / "target"
            target_dir.mkdir()
            target_dir.joinpath("chain_summary.json").write_text(
                json.dumps(
                    {
                        "chains": [
                            {"canonical_chain_id": "B", "sequence": "GG"},
                            {"canonical_chain_id": "A", "sequence": "ACD"},
                        ]
                    }
                )
            )

            sequences, labels = effective_target_sequences(
                root,
                {
                    "target": {
                        "chains": ["A", "B"],
                        "sequences": {"A": "ACDE", "B": "GGGG"},
                    }
                },
            )

            self.assertEqual(sequences, ("GG", "ACD"))
            self.assertEqual(labels, ("B", "A"))

    def test_unprepared_sequence_only_target_uses_config_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sequences, labels = effective_target_sequences(
                tmpdir,
                {"target": {"sequence": "ac d\n"}},
            )

            self.assertEqual(sequences, ("ACD",))
            self.assertEqual(labels, ("B",))

    def test_unprepared_multichain_fallback_keeps_sequence_label_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sequences, labels = effective_target_sequences(
                tmpdir,
                {
                    "target": {
                        "chains": ["missing", "B", "A"],
                        "sequences": {"A": "AA", "B": "GG"},
                    }
                },
            )

            self.assertEqual(sequences, ("GG", "AA"))
            self.assertEqual(labels, ("B", "A"))

    def test_malformed_prepared_summary_does_not_fall_back_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_dir = root / "target"
            target_dir.mkdir()
            target_dir.joinpath("chain_summary.json").write_text("{not-json")

            with self.assertRaisesRegex(
                EffectiveTargetSequenceError,
                "chain summary is unreadable",
            ):
                effective_target_sequences(
                    root,
                    {"target": {"sequence": "ACDE"}},
                )

    def test_inconsistent_prepared_summary_length_does_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_dir = root / "target"
            target_dir.mkdir()
            target_dir.joinpath("chain_summary.json").write_text(
                json.dumps(
                    {
                        "chains": [
                            {
                                "canonical_chain_id": "A",
                                "sequence": "ACD",
                                "length": 4,
                            }
                        ]
                    }
                )
            )

            with self.assertRaisesRegex(
                EffectiveTargetSequenceError,
                "length does not match",
            ):
                effective_target_sequences(
                    root,
                    {"target": {"sequence": "ACDE"}},
                )


if __name__ == "__main__":
    unittest.main()
