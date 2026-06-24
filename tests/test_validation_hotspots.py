from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from esmfold2_pipeline.artifacts import write_text_atomic
from esmfold2_pipeline.validation import (
    score_validation_hotspots,
    validation_hotspot_context,
)


class ValidationHotspotTest(unittest.TestCase):
    def test_score_validation_hotspots_uses_binder_to_target_heavy_atom_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_chain_summary(root, hotspot_indices=[1])
            cif_path = root / "prediction.cif"
            _write_hotspot_cif(cif_path, hotspot_distance=3.2)

            context = validation_hotspot_context(
                root,
                chain_role_map={"binder": ["A"], "target": ["B"]},
                contact_cutoff_angstrom=5.0,
            )
            assert context is not None
            metrics = score_validation_hotspots(cif_path, context=context)

            self.assertTrue(metrics["validation_hotspot_pass"])
            self.assertAlmostEqual(
                metrics["validation_hotspot_distance_angstrom"],
                3.2,
            )
            self.assertEqual(metrics["validation_hotspot_count"], 1)
            self.assertEqual(
                metrics["validation_hotspot_by_chain"]["B"]["hotspot_residue_numbers"],
                [2],
            )

    def test_score_validation_hotspots_rejects_missed_hotspot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_chain_summary(root, hotspot_indices=[1])
            cif_path = root / "prediction.cif"
            _write_hotspot_cif(cif_path, hotspot_distance=12.0)

            context = validation_hotspot_context(
                root,
                chain_role_map={"binder": ["A"], "target": ["B"]},
                contact_cutoff_angstrom=5.0,
            )
            assert context is not None
            metrics = score_validation_hotspots(cif_path, context=context)

            self.assertFalse(metrics["validation_hotspot_pass"])
            self.assertIn("exceeds cutoff", metrics["validation_hotspot_fail_reason"])


def _write_chain_summary(root: Path, *, hotspot_indices: list[int]) -> None:
    write_text_atomic(
        root / "target" / "chain_summary.json",
        json.dumps(
            {
                "chains": [
                    {
                        "canonical_chain_id": "T",
                        "sequence": "GGG",
                        "hotspot_indices": hotspot_indices,
                    }
                ]
            }
        ),
    )


def _write_hotspot_cif(path: Path, *, hotspot_distance: float) -> None:
    write_text_atomic(
        path,
        f"""
data_fake
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 C CA GLY A 1 0.000 0.000 0.000
ATOM 2 H H  GLY B 2 0.100 0.000 0.000
ATOM 3 C CA GLY B 1 40.000 0.000 0.000
ATOM 4 C CA GLY B 2 {hotspot_distance:.3f} 0.000 0.000
#
""".lstrip(),
    )


if __name__ == "__main__":
    unittest.main()
