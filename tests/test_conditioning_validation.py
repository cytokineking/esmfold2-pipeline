from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from esmfold2_pipeline.validation import (
    compute_fold_hotspot_contact_metrics,
    compute_fold_target_geometry_diagnostics,
    compute_fold_target_geometry_region_metrics,
    default_validation_binder_sequence,
    inspect_distogram_tensors,
    validate_conditioning_config,
)
from esmfold2_pipeline.structure import StructureTargetConfig, parse_structure_target


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConditioningValidationTest(unittest.TestCase):
    def test_default_validation_binder_sequence(self) -> None:
        sequence = default_validation_binder_sequence(10)
        self.assertEqual(len(sequence), 10)
        self.assertTrue(set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"))

    def test_inspect_distogram_tensors_accepts_expected_conditioned_mask(self) -> None:
        features = {
            "disto_cond": np.zeros((7, 7), dtype=np.int64),
            "disto_cond_mask": np.zeros((7, 7), dtype=bool),
        }
        features["disto_cond_mask"][:3, :3] = True

        check = inspect_distogram_tensors(
            features,
            target_length=3,
            expect_conditioned=True,
        )

        self.assertEqual(check.total_length, 7)
        self.assertEqual(check.target_block_true, 9)
        self.assertEqual(check.outside_target_block_true, 0)

    def test_inspect_distogram_tensors_rejects_missing_conditioning(self) -> None:
        features = {
            "disto_cond": np.zeros((7, 7), dtype=np.int64),
            "disto_cond_mask": np.zeros((7, 7), dtype=bool),
        }

        with self.assertRaisesRegex(ValueError, "expected 9 true"):
            inspect_distogram_tensors(
                features,
                target_length=3,
                expect_conditioned=True,
            )

    def test_inspect_distogram_tensors_rejects_outside_target_mask(self) -> None:
        features = {
            "disto_cond": np.zeros((7, 7), dtype=np.int64),
            "disto_cond_mask": np.zeros((7, 7), dtype=bool),
        }
        features["disto_cond_mask"][:3, :3] = True
        features["disto_cond_mask"][4, 4] = True

        with self.assertRaisesRegex(ValueError, "outside the target-chain block"):
            inspect_distogram_tensors(
                features,
                target_length=3,
                expect_conditioned=True,
            )

    def test_compute_fold_hotspot_contact_metrics_from_predicted_coords(self) -> None:
        inputs = {
            "atom_to_token": np.array([[0, 1, 2, 3]], dtype=np.int64),
            "ref_atom_name_chars": np.array(
                [
                    [
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("CA"),
                    ]
                ],
                dtype=np.int64,
            ),
            "atom_attention_mask": np.array([[True, True, True, True]]),
        }
        output = {
            "sample_atom_coords": np.array(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [10.0, 0.0, 0.0],
                        [14.0, 0.0, 0.0],
                        [45.0, 0.0, 0.0],
                    ]
                ],
                dtype=np.float64,
            )
        }

        metrics = compute_fold_hotspot_contact_metrics(
            inputs,
            output,
            target_sequence="GS",
            binder_sequence="AA",
            hotspot_indices=(0, 1),
            contact_cutoff_angstrom=12.0,
        )

        self.assertEqual(metrics.hotspot_count, 2)
        self.assertEqual(metrics.hotspot_contact_fraction, 0.5)
        self.assertEqual(metrics.hotspot_min_binder_distance_mean, 9.0)
        self.assertEqual(metrics.hotspot_min_binder_distance_min, 4.0)
        self.assertEqual(metrics.hotspot_heavy_atom_contact_fraction, 0.5)
        self.assertEqual(metrics.hotspot_representative_contact_fraction, 0.5)
        self.assertEqual(metrics.hotspot_contact_cutoff_angstrom, 12.0)

    def test_compute_fold_hotspot_contact_metrics_can_limit_binder_indices(self) -> None:
        inputs = {
            "atom_to_token": np.array([[0, 1, 2, 3]], dtype=np.int64),
            "ref_atom_name_chars": np.array(
                [
                    [
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("CA"),
                    ]
                ],
                dtype=np.int64,
            ),
            "atom_attention_mask": np.array([[True, True, True, True]]),
        }
        output = {
            "sample_atom_coords": np.array(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [10.0, 0.0, 0.0],
                        [14.0, 0.0, 0.0],
                        [45.0, 0.0, 0.0],
                    ]
                ],
                dtype=np.float64,
            )
        }

        metrics = compute_fold_hotspot_contact_metrics(
            inputs,
            output,
            target_sequence="GS",
            binder_sequence="AA",
            hotspot_indices=(0, 1),
            contact_cutoff_angstrom=12.0,
            binder_indices=(1,),
        )

        self.assertEqual(metrics.hotspot_contact_fraction, 0.0)
        self.assertEqual(metrics.hotspot_min_binder_distance_mean, 40.0)
        self.assertEqual(metrics.hotspot_min_binder_distance_min, 35.0)

    def test_hotspot_metrics_use_heavy_atoms_not_only_representatives(self) -> None:
        inputs = {
            "atom_to_token": np.array([[0, 0, 1, 1]], dtype=np.int64),
            "ref_atom_name_chars": np.array(
                [
                    [
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("ND2"),
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("NH1"),
                    ]
                ],
                dtype=np.int64,
            ),
            "atom_attention_mask": np.array([[True, True, True, True]]),
        }
        output = {
            "sample_atom_coords": np.array(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [10.0, 0.0, 0.0],
                        [20.0, 0.0, 0.0],
                        [13.0, 0.0, 0.0],
                    ]
                ],
                dtype=np.float64,
            )
        }

        metrics = compute_fold_hotspot_contact_metrics(
            inputs,
            output,
            target_sequence="N",
            binder_sequence="R",
            hotspot_indices=(0,),
            contact_cutoff_angstrom=5.0,
        )

        self.assertEqual(metrics.hotspot_heavy_atom_contact_fraction, 1.0)
        self.assertEqual(metrics.hotspot_min_heavy_atom_distance_min, 3.0)
        self.assertEqual(metrics.hotspot_representative_contact_fraction, 0.0)
        self.assertEqual(metrics.hotspot_min_representative_distance_min, 20.0)

    def test_target_geometry_diagnostics_split_chains_and_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_two_chain_gly_pdb(target_path)
            prepared_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "B"),
                    conditioning_mode="none",
                )
            )
        inputs = {
            "atom_to_token": np.array([[0, 1, 2, 3]], dtype=np.int64),
            "ref_atom_name_chars": np.array(
                [
                    [
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("CA"),
                    ]
                ],
                dtype=np.int64,
            ),
            "atom_attention_mask": np.array([[True, True, True, True]]),
        }
        output = {
            "sample_atom_coords": np.array(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 8.0, 0.0],
                        [1.0, 8.0, 0.0],
                    ]
                ],
                dtype=np.float64,
            )
        }

        diagnostics = compute_fold_target_geometry_diagnostics(
            inputs,
            output,
            prepared_target=prepared_target,
        )

        self.assertEqual(
            set(diagnostics["target_chain_geometry"]),
            {"A", "B"},
        )
        self.assertEqual(
            diagnostics["target_chain_geometry"]["A"]["distance_rmse"],
            0.0,
        )
        self.assertEqual(
            diagnostics["target_chain_geometry"]["B"]["aligned_rmsd"],
            0.0,
        )
        pair = diagnostics["target_assembly_geometry"]["A__B"]
        self.assertAlmostEqual(pair["pair_distance_rmse"], 1.9938185484595643)
        self.assertIsNone(pair["contact_recovery_8A"])
        self.assertEqual(pair["contact_recovery_12A"], 1.0)
        self.assertEqual(pair["residue_pair_count"], 4)

    def test_target_geometry_region_metrics_use_selected_residues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_two_chain_gly_pdb(target_path)
            prepared_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "B"),
                    conditioning_mode="none",
                )
            )
        inputs = {
            "atom_to_token": np.array([[0, 1, 2, 3]], dtype=np.int64),
            "ref_atom_name_chars": np.array(
                [
                    [
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("CA"),
                        _encoded_atom_name("CA"),
                    ]
                ],
                dtype=np.int64,
            ),
            "atom_attention_mask": np.array([[True, True, True, True]]),
        }
        output = {
            "sample_atom_coords": np.array(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 8.0, 0.0],
                        [1.0, 8.0, 0.0],
                    ]
                ],
                dtype=np.float64,
            )
        }

        metrics = compute_fold_target_geometry_region_metrics(
            inputs,
            output,
            prepared_target=prepared_target,
            target_indices=(0, 2),
        )

        self.assertEqual(metrics.target_residue_count, 2)
        self.assertEqual(metrics.target_distance_rmse, 2.0)
        self.assertEqual(metrics.target_aligned_rmsd, 1.0)

    def test_validate_conditioning_requires_structure_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "config.yaml"
            config.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {Path(tmpdir) / "campaign"}
""".lstrip()
            )

            with self.assertRaisesRegex(ValueError, "requires target.structure"):
                validate_conditioning_config(config)

    def test_validate_conditioning_help_is_available(self) -> None:
        result = _run_cli("validate-conditioning", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fold-only GPU validation", result.stdout)
        self.assertIn("--binder-sequence", result.stdout)


def _run_cli(*args: object) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "esmfold2_pipeline", *[str(arg) for arg in args]],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _encoded_atom_name(name: str) -> list[int]:
    return [ord(char) - 32 for char in name] + [0] * (4 - len(name))


def _write_two_chain_gly_pdb(path: Path) -> None:
    lines = [
        _pdb_atom_line(1, "N", "GLY", "A", 1, 0.0, 0.0, 0.0),
        _pdb_atom_line(2, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0),
        _pdb_atom_line(3, "C", "GLY", "A", 1, 0.0, 0.0, 0.0),
        _pdb_atom_line(4, "N", "GLY", "A", 2, 1.0, 0.0, 0.0),
        _pdb_atom_line(5, "CA", "GLY", "A", 2, 1.0, 0.0, 0.0),
        _pdb_atom_line(6, "C", "GLY", "A", 2, 1.0, 0.0, 0.0),
        _pdb_atom_line(7, "N", "GLY", "B", 1, 0.0, 10.0, 0.0),
        _pdb_atom_line(8, "CA", "GLY", "B", 1, 0.0, 10.0, 0.0),
        _pdb_atom_line(9, "C", "GLY", "B", 1, 0.0, 10.0, 0.0),
        _pdb_atom_line(10, "N", "GLY", "B", 2, 1.0, 10.0, 0.0),
        _pdb_atom_line(11, "CA", "GLY", "B", 2, 1.0, 10.0, 0.0),
        _pdb_atom_line(12, "C", "GLY", "B", 2, 1.0, 10.0, 0.0),
    ]
    path.write_text("".join(lines))


def _pdb_atom_line(
    serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_id: int,
    x: float,
    y: float,
    z: float,
) -> str:
    element = atom_name.strip()[0]
    return (
        f"ATOM  {serial:5d} {atom_name:^4} {res_name:>3} {chain_id}"
        f"{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00 20.00          {element:>2s}\n"
    )


if __name__ == "__main__":
    unittest.main()
