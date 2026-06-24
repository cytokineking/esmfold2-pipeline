from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import biotite.structure.io.pdbx as pdbx

from esmfold2_pipeline.config import check_campaign_config
from esmfold2_pipeline.planning import plan_campaign
from esmfold2_pipeline.structure import (
    StructureTargetConfig,
    StructureTargetError,
    parse_structure_target,
    resolve_target_geometry_drift_indices,
    write_target_artifacts,
)


class StructureTargetTest(unittest.TestCase):
    def test_parse_pdb_chain_crop_and_hotspot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "target.pdb"
            _write_test_pdb(pdb_path, [("GLY", 1), ("SER", 2), ("HIS", 3), ("SER", 4), ("MET", 5)])
            prepared = parse_structure_target(
                StructureTargetConfig(
                    path=pdb_path,
                    chains=("A",),
                    structure_indexing="auth_seq_id",
                    crop={"*": ("1-5",)},
                    hotspots={"A": ("2",)},
                    conditioning_mode="distogram",
                )
            )

        self.assertEqual(prepared.input_format, "pdb")
        self.assertEqual(len(prepared.chains), 1)
        chain = prepared.chains[0]
        self.assertEqual(chain.canonical_chain_id, "A")
        self.assertEqual(chain.sequence, "GSHSM")
        self.assertEqual(chain.hotspot_indices, (1,))
        self.assertEqual(len(chain.distogram), 5)
        self.assertEqual(len(chain.distogram[0]), 5)
        self.assertEqual(chain.residues[0].representative_atom, "CA")
        self.assertEqual(chain.residues[1].representative_atom, "CB")

    def test_parse_mmcif_preserves_auth_and_label_residue_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = Path(tmpdir) / "target.cif"
            _write_test_cif(
                cif_path,
                [("SER", 6), ("THR", 7), ("LYS", 8), ("LYS", 9), ("THR", 10)],
            )
            prepared = parse_structure_target(
                StructureTargetConfig(
                    path=cif_path,
                    chains=("A",),
                    structure_indexing="auth_seq_id",
                    crop={"A": ("6-10",)},
                    hotspots={"A": ("7",)},
                    conditioning_mode="distogram",
                )
            )

        chain = prepared.chains[0]
        first = chain.residues[0]
        self.assertEqual(prepared.input_format, "mmcif")
        self.assertEqual(chain.sequence, "STKKT")
        self.assertEqual(first.auth_asym_id, "A")
        self.assertEqual(first.label_asym_id, "A")
        self.assertEqual(first.auth_seq_id, "6")
        self.assertEqual(first.label_seq_id, "6")
        self.assertEqual(first.model_residue_index_0, 0)
        self.assertEqual(chain.hotspot_indices, (1,))

    def test_bindcraft_style_hotspot_string_uses_auth_numbering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            cif_path = root / "target.cif"
            _write_test_cif_with_auth_and_label(
                cif_path,
                [
                    ("SER", 1, 6),
                    ("THR", 2, 7),
                    ("LYS", 3, 8),
                    ("GLY", 4, 9),
                    ("MET", 5, 10),
                ],
            )
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  structure: {cif_path}
  chains: [A]
  structure_indexing: auth_seq_id
  hotspots: "A7,9"
  conditioning:
    mode: distogram
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Cutoff2025
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            check = check_campaign_config(config)

        self.assertTrue(check.ok, check.errors)
        assert check.config is not None
        assert check.config.target_structure is not None
        assert check.prepared_target is not None
        self.assertEqual(check.config.target_structure.hotspots, {"A": ("7", "9")})
        chain = check.prepared_target.chains[0]
        self.assertEqual(chain.hotspot_indices, (1, 3))
        self.assertEqual(chain.residues[1].auth_seq_id, "7")
        self.assertEqual(chain.residues[1].label_seq_id, "2")
        self.assertEqual(chain.residues[3].auth_seq_id, "9")
        self.assertEqual(chain.residues[3].label_seq_id, "4")

    def test_bindcraft_style_hotspot_string_requires_initial_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            cif_path = root / "target.cif"
            _write_test_cif(cif_path, [("SER", 6), ("THR", 7), ("LYS", 8)])
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  structure: {cif_path}
  chains: [A]
  structure_indexing: auth_seq_id
  hotspots: "7,8"
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Cutoff2025
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            check = check_campaign_config(config)

        self.assertFalse(check.ok)
        self.assertIn("chain-qualified", check.errors[0])

    def test_write_target_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cif_path = root / "target.cif"
            _write_test_cif(cif_path, [("SER", 6), ("THR", 7), ("LYS", 8), ("LYS", 9)])
            prepared = parse_structure_target(
                StructureTargetConfig(
                    path=cif_path,
                    chains=("A",),
                    structure_indexing="auth_seq_id",
                    crop={"A": ("6-9",)},
                    hotspots={"A": ("7",)},
                    conditioning_mode="distogram",
                )
            )
            result = write_target_artifacts(
                prepared,
                root / "target",
                conditioning_mode="distogram",
            )

            self.assertTrue(result.normalized_target.exists())
            self.assertTrue(result.residue_map_csv.exists())
            self.assertTrue(result.chain_summary_json.exists())
            self.assertEqual(len(result.conditioning_files), 4)
            self.assertIn("auth_seq_id", result.residue_map_csv.read_text())

            summary = json.loads(result.chain_summary_json.read_text())
            self.assertEqual(summary["chains"][0]["length"], 4)
            self.assertEqual(summary["chains"][0]["hotspot_indices"], [1])

            shapes = sorted(_read_npy_shape(path) for path in result.conditioning_files)
            self.assertEqual(shapes, [(4,), (4, 3), (4, 4), (4, 4)])

    def test_parse_multichain_hotspot_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            cif_path = root / "target.cif"
            _write_test_cif_multichain(
                cif_path,
                [
                    ("A", [("SER", 6), ("THR", 7), ("LYS", 8)]),
                    ("B", [("GLY", 101), ("TYR", 102)]),
                ],
            )
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  structure: {cif_path}
  chains: [A, B]
  structure_indexing: auth_seq_id
  hotspots: "A:7;B:102"
  conditioning:
    mode: distogram
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            check = check_campaign_config(config)

        self.assertTrue(check.ok, check.errors)
        assert check.config is not None
        assert check.config.target_structure is not None
        assert check.prepared_target is not None
        self.assertEqual(
            check.config.target_structure.hotspots,
            {"A": ("7",), "B": ("102",)},
        )
        self.assertEqual(len(check.prepared_target.chains), 2)
        self.assertEqual(check.prepared_target.chains[0].canonical_chain_id, "A")
        self.assertEqual(check.prepared_target.chains[0].sequence, "STK")
        self.assertEqual(check.prepared_target.chains[0].hotspot_indices, (1,))
        self.assertEqual(check.prepared_target.chains[1].canonical_chain_id, "B")
        self.assertEqual(check.prepared_target.chains[1].sequence, "GY")
        self.assertEqual(check.prepared_target.chains[1].hotspot_indices, (1,))

    def test_parse_multichain_duplicate_sequences_preserves_chains(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = Path(tmpdir) / "homodimer.cif"
            _write_test_cif_multichain(
                cif_path,
                [
                    ("A", [("SER", 1), ("THR", 2), ("LYS", 3)]),
                    ("B", [("SER", 1), ("THR", 2), ("LYS", 3)]),
                ],
            )

            prepared = parse_structure_target(
                StructureTargetConfig(
                    path=cif_path,
                    chains=("A", "B"),
                    structure_indexing="auth_seq_id",
                    hotspots={"A": ("2",), "B": ("3",)},
                    conditioning_mode="distogram",
                    conditioning_assembly=True,
                )
            )

        self.assertEqual(len(prepared.chains), 2)
        self.assertEqual(prepared.chains[0].canonical_chain_id, "A")
        self.assertEqual(prepared.chains[1].canonical_chain_id, "B")
        self.assertEqual(prepared.chains[0].sequence, "STK")
        self.assertEqual(prepared.chains[1].sequence, "STK")
        self.assertEqual(prepared.chains[0].hotspot_indices, (1,))
        self.assertEqual(prepared.chains[1].hotspot_indices, (2,))

    def test_multichain_unqualified_crop_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            cif_path = root / "target.cif"
            _write_test_cif_multichain(
                cif_path,
                [
                    ("A", [("SER", 1), ("THR", 2)]),
                    ("B", [("GLY", 1), ("TYR", 2)]),
                ],
            )
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  structure: {cif_path}
  chains: [A, B]
  structure_indexing: auth_seq_id
  crop: ["1-2"]
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            check = check_campaign_config(config)

        self.assertFalse(check.ok)
        self.assertIn("target.crop uses an unqualified selector", check.errors[0])

    def test_multichain_unqualified_hotspots_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            cif_path = root / "target.cif"
            _write_test_cif_multichain(
                cif_path,
                [
                    ("A", [("SER", 1), ("THR", 2)]),
                    ("B", [("GLY", 1), ("TYR", 2)]),
                ],
            )
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  structure: {cif_path}
  chains: [A, B]
  structure_indexing: auth_seq_id
  hotspots: [1]
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            check = check_campaign_config(config)

        self.assertFalse(check.ok)
        self.assertIn("target.hotspots", check.errors[0])

    def test_write_multichain_assembly_conditioning_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cif_path = root / "target.cif"
            _write_test_cif_multichain(
                cif_path,
                [
                    ("A", [("SER", 6), ("THR", 7), ("LYS", 8)]),
                    ("B", [("GLY", 101), ("TYR", 102)]),
                ],
            )
            prepared = parse_structure_target(
                StructureTargetConfig(
                    path=cif_path,
                    chains=("A", "B"),
                    structure_indexing="auth_seq_id",
                    hotspots={"A": ("7",), "B": ("102",)},
                    conditioning_mode="distogram",
                    conditioning_assembly=True,
                )
            )

            result = write_target_artifacts(
                prepared,
                root / "target",
                conditioning_mode="distogram",
                conditioning_assembly=True,
                representative_atom="esmfold2_default",
                require_resolved=True,
            )

            self.assertIsNotNone(result.assembly_conditioning_json)
            assert result.assembly_conditioning_json is not None
            self.assertTrue(result.assembly_conditioning_json.exists())
            self.assertEqual(len(result.conditioning_files), 10)
            pair_path = root / "target" / "conditioning" / "pair_A_B_distogram.npy"
            pair_mask_path = root / "target" / "conditioning" / "pair_A_B_distogram_mask.npy"
            self.assertTrue(pair_path.exists())
            self.assertTrue(pair_mask_path.exists())
            self.assertEqual(_read_npy_shape(pair_path), (3, 2))
            self.assertEqual(_read_npy_shape(pair_mask_path), (3, 2))

            metadata = json.loads(result.assembly_conditioning_json.read_text())
            self.assertEqual(metadata["chain_order"], ["A", "B"])
            self.assertEqual(
                metadata["target_chain_spans"],
                [
                    {
                        "auth_asym_id": "A",
                        "chain_id": "A",
                        "end": 3,
                        "label_asym_id": "A",
                        "length": 3,
                        "start": 0,
                    },
                    {
                        "auth_asym_id": "B",
                        "chain_id": "B",
                        "end": 5,
                        "label_asym_id": "B",
                        "length": 2,
                        "start": 3,
                    },
                ],
            )
            self.assertEqual(metadata["chain_pairs"][0]["chain_id_1"], "A")
            self.assertEqual(metadata["chain_pairs"][0]["chain_id_2"], "B")
            self.assertEqual(metadata["chain_pairs"][0]["shape"], [3, 2])

    def test_normalized_cif_for_pdb_multichain_writes_polymer_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdb_path = root / "target.pdb"
            _write_test_pdb_multichain(
                pdb_path,
                [
                    ("A", [("SER", 6), ("THR", 7), ("LYS", 8)]),
                    ("C", [("GLY", 101), ("TYR", 102)]),
                ],
            )
            prepared = parse_structure_target(
                StructureTargetConfig(
                    path=pdb_path,
                    chains=("A", "C"),
                    structure_indexing="auth_seq_id",
                    conditioning_mode="distogram",
                    conditioning_assembly=True,
                )
            )

            result = write_target_artifacts(
                prepared,
                root / "target",
                conditioning_mode="distogram",
                conditioning_assembly=True,
                representative_atom="esmfold2_default",
                require_resolved=True,
            )

            cif_file = pdbx.CIFFile.read(result.normalized_target)
            struct_asym = cif_file.block["struct_asym"]
            self.assertEqual(set(struct_asym["id"].as_array()), {"A", "C"})
            self.assertEqual(
                set(struct_asym["entity_id"].as_array()),
                {"1", "2"},
            )

            entity_poly_seq = cif_file.block["entity_poly_seq"]
            rows_by_entity: dict[str, list[str]] = {"1": [], "2": []}
            for entity_id, mon_id in zip(
                entity_poly_seq["entity_id"].as_array(),
                entity_poly_seq["mon_id"].as_array(),
            ):
                rows_by_entity[str(entity_id)].append(str(mon_id))
            self.assertEqual(rows_by_entity["1"], ["SER", "THR", "LYS"])
            self.assertEqual(rows_by_entity["2"], ["GLY", "TYR"])

    def test_plan_multichain_assembly_conditioning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            cif_path = root / "target.cif"
            _write_test_cif_multichain(
                cif_path,
                [
                    ("A", [("SER", 6), ("THR", 7), ("LYS", 8)]),
                    ("B", [("GLY", 101), ("TYR", 102)]),
                ],
            )
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  structure: {cif_path}
  chains: [A, B]
  structure_indexing: auth_seq_id
  hotspots: "A:7;B:102"
  conditioning:
    mode: distogram
    assembly: true
    chain_pairs:
      - [A, B]
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            check = check_campaign_config(config)
            self.assertTrue(check.ok, check.errors)
            assert check.config is not None
            assert check.config.target_structure is not None
            self.assertTrue(check.config.target_structure.conditioning_assembly)
            self.assertEqual(
                check.config.target_structure.conditioning_chain_pairs,
                (("A", "B"),),
            )

            plan = plan_campaign(config)

            self.assertEqual(plan.shard_count, 1)
            self.assertTrue(
                (
                    campaign_dir
                    / "target"
                    / "conditioning"
                    / "pair_A_B_distogram.npy"
                ).exists()
            )
            assembly_json = (
                campaign_dir
                / "target"
                / "conditioning"
                / "assembly_conditioning.json"
            )
            self.assertTrue(assembly_json.exists())
            resolved = (campaign_dir / "resolved_config.yaml").read_text()
            self.assertIn("assembly: true", resolved)
            self.assertIn("chain_pairs", resolved)

    def test_pdb_insertion_code_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "insertion.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        _pdb_atom_line(1, "N", "SER", "A", 10, "A", 0.0, 0.0, 0.0),
                        _pdb_atom_line(2, "CA", "SER", "A", 10, "A", 1.0, 0.0, 0.0),
                        _pdb_atom_line(3, "CB", "SER", "A", 10, "A", 1.0, 1.0, 0.0),
                        _pdb_atom_line(4, "C", "SER", "A", 10, "A", 2.0, 0.0, 0.0),
                        _pdb_atom_line(5, "N", "GLY", "A", 11, "", 3.0, 0.0, 0.0),
                        _pdb_atom_line(6, "CA", "GLY", "A", 11, "", 4.0, 0.0, 0.0),
                    ]
                )
            )

            prepared = parse_structure_target(
                StructureTargetConfig(
                    path=pdb_path,
                    chains=("A",),
                    structure_indexing="auth_seq_id",
                    crop={"A": ("10A",)},
                    hotspots={"A": ("10A",)},
                    conditioning_mode="distogram",
                )
            )

            chain = prepared.chains[0]
            self.assertEqual(chain.sequence, "S")
            self.assertEqual(chain.hotspot_indices, (0,))
            self.assertEqual(chain.residues[0].auth_seq_id, "10")
            self.assertEqual(chain.residues[0].pdbx_pdb_ins_code, "A")

    def test_missing_representative_atom_uses_partial_mask_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "missing_rep.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        _pdb_atom_line(1, "N", "ALA", "A", 1, "", 0.0, 0.0, 0.0),
                        _pdb_atom_line(2, "C", "ALA", "A", 1, "", 1.0, 0.0, 0.0),
                        _pdb_atom_line(3, "O", "ALA", "A", 1, "", 2.0, 0.0, 0.0),
                    ]
                )
            )

            prepared = parse_structure_target(
                StructureTargetConfig(
                    path=pdb_path,
                    chains=("A",),
                    conditioning_mode="distogram",
                )
            )

        chain = prepared.chains[0]
        self.assertEqual(chain.sequence, "A")
        self.assertEqual(chain.representative_coord_mask, (False,))
        self.assertEqual(chain.distogram_mask, ((False,),))

    def test_missing_representative_atom_fails_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "missing_rep.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        _pdb_atom_line(1, "N", "ALA", "A", 1, "", 0.0, 0.0, 0.0),
                        _pdb_atom_line(2, "C", "ALA", "A", 1, "", 1.0, 0.0, 0.0),
                        _pdb_atom_line(3, "O", "ALA", "A", 1, "", 2.0, 0.0, 0.0),
                    ]
                )
            )

            with self.assertRaisesRegex(StructureTargetError, "unresolved representative"):
                parse_structure_target(
                    StructureTargetConfig(
                        path=pdb_path,
                        chains=("A",),
                        conditioning_mode="distogram",
                        require_resolved=True,
                    )
                )

    def test_discontinuous_pdb_without_full_sequence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "gapped.pdb"
            _write_test_pdb(pdb_path, [("GLY", 1), ("MET", 4)])

            with self.assertRaisesRegex(StructureTargetError, "discontinuous observed"):
                parse_structure_target(
                    StructureTargetConfig(
                        path=pdb_path,
                        chains=("A",),
                        conditioning_mode="distogram",
                    )
                )

    def test_pdb_seqres_preserves_missing_internal_residues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "seqres_missing_loop.pdb"
            _write_seqres_pdb(
                pdb_path,
                chain_id="A",
                seqres=["GLY", "SER", "THR", "MET"],
                observed=[("GLY", 1), ("MET", 4)],
            )

            prepared = parse_structure_target(
                StructureTargetConfig(
                    path=pdb_path,
                    chains=("A",),
                    conditioning_mode="distogram",
                )
            )

        chain = prepared.chains[0]
        self.assertEqual(chain.sequence, "GSTM")
        self.assertEqual(chain.representative_coord_mask, (True, False, False, True))
        self.assertEqual(chain.distogram_mask[0], (True, False, False, True))
        self.assertEqual(chain.distogram_mask[1], (False, False, False, False))
        self.assertEqual(chain.residues[3].auth_seq_id, "4")
        self.assertEqual(chain.residues[3].label_seq_id, "4")

    def test_user_sequence_preserves_discontinuous_pdb_register(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "gapped_with_sequence.pdb"
            _write_test_pdb(pdb_path, [("GLY", 1), ("MET", 4)])

            prepared = parse_structure_target(
                StructureTargetConfig(
                    path=pdb_path,
                    chains=("A",),
                    sequences={"A": "GSTM"},
                    conditioning_mode="distogram",
                )
            )

        chain = prepared.chains[0]
        self.assertEqual(chain.sequence, "GSTM")
        self.assertEqual(chain.sequence_source, "user_sequence")
        self.assertEqual(chain.representative_coord_mask, (True, False, False, True))

    def test_hotspot_on_unresolved_template_residue_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "seqres_missing_loop.pdb"
            _write_seqres_pdb(
                pdb_path,
                chain_id="A",
                seqres=["GLY", "SER", "THR", "MET"],
                observed=[("GLY", 1), ("MET", 4)],
            )

            with self.assertRaisesRegex(StructureTargetError, "target.hotspots"):
                parse_structure_target(
                    StructureTargetConfig(
                        path=pdb_path,
                        chains=("A",),
                        structure_indexing="auth_seq_id",
                        hotspots={"A": ("2",)},
                        conditioning_mode="distogram",
                    )
                )

    def test_drift_regions_default_to_whole_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cif_path = root / "target.cif"
            _write_test_cif_multichain(
                cif_path,
                [
                    ("A", [("SER", 6), ("THR", 7), ("LYS", 8)]),
                    ("B", [("GLY", 101), ("TYR", 102)]),
                ],
            )
            prepared = parse_structure_target(
                StructureTargetConfig(
                    path=cif_path,
                    chains=("A", "B"),
                    structure_indexing="auth_seq_id",
                )
            )

            indices = resolve_target_geometry_drift_indices(
                prepared,
                None,
                structure_indexing="auth_seq_id",
            )

        self.assertEqual(indices, (0, 1, 2, 3, 4))

    def test_drift_regions_select_ranges_across_chains(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cif_path = root / "target.cif"
            _write_test_cif_multichain(
                cif_path,
                [
                    ("A", [("SER", 6), ("THR", 7), ("LYS", 8)]),
                    ("B", [("GLY", 101), ("TYR", 102)]),
                ],
            )
            prepared = parse_structure_target(
                StructureTargetConfig(
                    path=cif_path,
                    chains=("A", "B"),
                    structure_indexing="auth_seq_id",
                )
            )

            indices = resolve_target_geometry_drift_indices(
                prepared,
                {"A": ("7-8",), "B": ("101",)},
                structure_indexing="auth_seq_id",
            )

        self.assertEqual(indices, (1, 2, 3))

    def test_drift_regions_support_whole_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cif_path = root / "target.cif"
            _write_test_cif_multichain(
                cif_path,
                [
                    ("A", [("SER", 6), ("THR", 7)]),
                    ("B", [("GLY", 101), ("TYR", 102)]),
                ],
            )
            prepared = parse_structure_target(
                StructureTargetConfig(path=cif_path, chains=("A", "B"))
            )

            indices = resolve_target_geometry_drift_indices(
                prepared,
                {"B": ("all",)},
            )

        self.assertEqual(indices, (2, 3))

    def test_drift_regions_reject_unknown_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cif_path = root / "target.cif"
            _write_test_cif(cif_path, [("SER", 6), ("THR", 7)])
            prepared = parse_structure_target(
                StructureTargetConfig(path=cif_path, chains=("A",))
            )

            with self.assertRaisesRegex(StructureTargetError, "unknown chain Z"):
                resolve_target_geometry_drift_indices(prepared, {"Z": ("all",)})

    def test_check_rejects_invalid_drift_region_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cif_path = root / "target.cif"
            _write_test_cif(cif_path, [("SER", 6), ("THR", 7)])
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  structure: {cif_path}
  chains: [A]
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
loss:
  target_geometry_drift:
    enabled: true
    regions:
      Z: all
output: {root / "campaign"}
""".lstrip()
            )

            result = check_campaign_config(config)

        self.assertFalse(result.ok)
        self.assertIn("loss.target_geometry_drift.regions", result.errors[0])

    def test_missing_hotspot_fails_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = Path(tmpdir) / "target.cif"
            _write_test_cif(cif_path, [("SER", 6), ("THR", 7), ("LYS", 8)])
            with self.assertRaisesRegex(StructureTargetError, "matched no residues"):
                parse_structure_target(
                    StructureTargetConfig(
                        path=cif_path,
                        chains=("A",),
                        structure_indexing="auth_seq_id",
                        crop={"A": ("6-8",)},
                        hotspots={"A": ("9999",)},
                    )
                )

    def test_negative_hotspot_contact_weight_fails_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
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
loss:
  hotspot_contact_weight: -0.1
output: {root / "campaign"}
""".lstrip()
            )

            check = check_campaign_config(config)
            self.assertFalse(check.ok)
            self.assertIn("loss.hotspot_contact_weight", check.errors[0])

    def test_nonpositive_hotspot_contact_cutoff_fails_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
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
loss:
  hotspot_contact_cutoff_angstrom: 0
output: {root / "campaign"}
""".lstrip()
            )

            check = check_campaign_config(config)
            self.assertFalse(check.ok)
            self.assertIn("loss.hotspot_critic_contact_cutoff_angstrom", check.errors[0])

    def test_nonpositive_hotspot_distogram_contact_cutoff_fails_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
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
loss:
  hotspot_distogram_contact_cutoff_angstrom: 0
output: {root / "campaign"}
""".lstrip()
            )

            check = check_campaign_config(config)
            self.assertFalse(check.ok)
            self.assertIn(
                "loss.hotspot_distogram_contact_cutoff_angstrom",
                check.errors[0],
            )

    def test_check_and_plan_structure_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            cif_path = root / "target.cif"
            _write_test_cif(cif_path, [("SER", 6), ("THR", 7), ("LYS", 8)])
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  structure: {cif_path}
  chains: [A]
  structure_indexing: auth_seq_id
  crop: ["6-8"]
  hotspots:
    A: [7]
  conditioning:
    mode: distogram
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
loss:
  hotspot_contact_weight: 0.5
  hotspot_contact_cutoff_angstrom: 4.5
output: {campaign_dir}
""".lstrip()
            )

            check = check_campaign_config(config)
            self.assertTrue(check.ok, check.errors)
            self.assertIsNotNone(check.prepared_target)
            self.assertFalse((campaign_dir / "campaign.sqlite").exists())

            plan = plan_campaign(config)
            self.assertEqual(plan.shard_count, 1)
            self.assertEqual(plan.config.hotspot_contact_weight, 0.5)
            self.assertEqual(plan.config.hotspot_distogram_contact_cutoff_angstrom, 20.0)
            self.assertEqual(plan.config.hotspot_critic_contact_cutoff_angstrom, 4.5)
            self.assertEqual(plan.config.hotspot_contact_cutoff_angstrom, 4.5)
            self.assertTrue((campaign_dir / "campaign.sqlite").exists())
            self.assertTrue((campaign_dir / "target" / "normalized_target.cif").exists())
            self.assertTrue((campaign_dir / "target" / "residue_map.csv").exists())
            self.assertTrue(
                (campaign_dir / "target" / "conditioning" / "chain_A_distogram.npy").exists()
            )


def _read_npy_shape(path: Path) -> tuple[int, ...]:
    data = path.read_bytes()
    if not data.startswith(b"\x93NUMPY\x01\x00"):
        raise AssertionError(f"not a v1 npy file: {path}")
    header_len = struct.unpack("<H", data[8:10])[0]
    header = data[10 : 10 + header_len].decode("latin1")
    marker = "'shape': "
    start = header.index(marker) + len(marker)
    end = header.index(")", start) + 1
    shape_text = header[start + 1 : end - 1]
    shape = tuple(
        int(part.strip())
        for part in shape_text.split(",")
        if part.strip()
    )
    return shape


def _write_test_pdb(path: Path, residues: list[tuple[str, int]]) -> None:
    lines = []
    serial = 1
    for index, (res_name, res_id) in enumerate(residues):
        x = float(index * 3)
        for atom_name, dx, dy, dz in [
            ("N", 0.0, 0.0, 0.0),
            ("CA", 1.0, 0.0, 0.0),
            ("C", 2.0, 0.0, 0.0),
            ("O", 2.5, 0.5, 0.0),
        ]:
            lines.append(_pdb_atom_line(serial, atom_name, res_name, "A", res_id, "", x + dx, dy, dz))
            serial += 1
        if res_name != "GLY":
            lines.append(_pdb_atom_line(serial, "CB", res_name, "A", res_id, "", x + 1.0, 1.0, 0.0))
            serial += 1
    path.write_text("".join(lines))


def _write_test_pdb_multichain(
    path: Path,
    chains: list[tuple[str, list[tuple[str, int]]]],
) -> None:
    lines = []
    serial = 1
    for chain_index, (chain_id, residues) in enumerate(chains):
        y = float(chain_index * 10)
        for index, (res_name, res_id) in enumerate(residues):
            x = float(index * 3)
            for atom_name, dx, dy, dz in [
                ("N", 0.0, 0.0, 0.0),
                ("CA", 1.0, 0.0, 0.0),
                ("C", 2.0, 0.0, 0.0),
                ("O", 2.5, 0.5, 0.0),
            ]:
                lines.append(
                    _pdb_atom_line(
                        serial,
                        atom_name,
                        res_name,
                        chain_id,
                        res_id,
                        "",
                        x + dx,
                        y + dy,
                        dz,
                    )
                )
                serial += 1
            if res_name != "GLY":
                lines.append(
                    _pdb_atom_line(
                        serial,
                        "CB",
                        res_name,
                        chain_id,
                        res_id,
                        "",
                        x + 1.0,
                        y + 1.0,
                        0.0,
                    )
                )
                serial += 1
    path.write_text("".join(lines))


def _write_seqres_pdb(
    path: Path,
    *,
    chain_id: str,
    seqres: list[str],
    observed: list[tuple[str, int]],
) -> None:
    seqres_line = (
        f"SEQRES{1:4d} {chain_id:1s}{len(seqres):5d}  "
        + " ".join(seqres)
        + "\n"
    )
    body = []
    serial = 1
    for index, (res_name, res_id) in enumerate(observed):
        x = float(index * 9)
        for atom_name, dx, dy, dz in [
            ("N", 0.0, 0.0, 0.0),
            ("CA", 1.0, 0.0, 0.0),
            ("C", 2.0, 0.0, 0.0),
            ("O", 2.5, 0.5, 0.0),
        ]:
            body.append(
                _pdb_atom_line(
                    serial,
                    atom_name,
                    res_name,
                    chain_id,
                    res_id,
                    "",
                    x + dx,
                    dy,
                    dz,
                )
            )
            serial += 1
        if res_name != "GLY":
            body.append(
                _pdb_atom_line(
                    serial,
                    "CB",
                    res_name,
                    chain_id,
                    res_id,
                    "",
                    x + 1.0,
                    1.0,
                    0.0,
                )
            )
            serial += 1
    path.write_text(seqres_line + "".join(body))


def _write_test_cif(path: Path, residues: list[tuple[str, int]]) -> None:
    _write_test_cif_with_auth_and_label(
        path,
        [(res_name, res_id, res_id) for res_name, res_id in residues],
    )


def _write_test_cif_with_auth_and_label(
    path: Path,
    residues: list[tuple[str, int, int]],
) -> None:
    lines = [
        "data_test",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    atom_id = 1
    for index, (res_name, label_res_id, auth_res_id) in enumerate(residues):
        x = float(index * 3)
        atoms = [
            ("N", "N", x + 0.0, 0.0, 0.0),
            ("C", "CA", x + 1.0, 0.0, 0.0),
            ("C", "C", x + 2.0, 0.0, 0.0),
            ("O", "O", x + 2.5, 0.5, 0.0),
        ]
        if res_name != "GLY":
            atoms.append(("C", "CB", x + 1.0, 1.0, 0.0))
        for element, atom_name, ax, ay, az in atoms:
            lines.append(
                " ".join(
                    [
                        "ATOM",
                        str(atom_id),
                        element,
                        atom_name,
                        ".",
                        res_name,
                        "A",
                        "1",
                        str(label_res_id),
                        "?",
                        f"{ax:.3f}",
                        f"{ay:.3f}",
                        f"{az:.3f}",
                        "1.00",
                        "20.00",
                        "?",
                        str(auth_res_id),
                        res_name,
                        "A",
                        atom_name,
                        "1",
                    ]
                )
            )
            atom_id += 1
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")


def _write_test_cif_multichain(
    path: Path,
    chains: list[tuple[str, list[tuple[str, int]]]],
) -> None:
    lines = [
        "data_test",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    atom_id = 1
    for entity_id, (chain_id, residues) in enumerate(chains, start=1):
        y = float((entity_id - 1) * 10)
        for label_index, (res_name, auth_res_id) in enumerate(residues, start=1):
            x = float((label_index - 1) * 3)
            atoms = [
                ("N", "N", x + 0.0, y, 0.0),
                ("C", "CA", x + 1.0, y, 0.0),
                ("C", "C", x + 2.0, y, 0.0),
                ("O", "O", x + 2.5, y + 0.5, 0.0),
            ]
            if res_name != "GLY":
                atoms.append(("C", "CB", x + 1.0, y + 1.0, 0.0))
            for element, atom_name, ax, ay, az in atoms:
                lines.append(
                    " ".join(
                        [
                            "ATOM",
                            str(atom_id),
                            element,
                            atom_name,
                            ".",
                            res_name,
                            chain_id,
                            str(entity_id),
                            str(label_index),
                            "?",
                            f"{ax:.3f}",
                            f"{ay:.3f}",
                            f"{az:.3f}",
                            "1.00",
                            "20.00",
                            "?",
                            str(auth_res_id),
                            res_name,
                            chain_id,
                            atom_name,
                            "1",
                        ]
                    )
                )
                atom_id += 1
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")


def _pdb_atom_line(
    serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_id: int,
    ins_code: str,
    x: float,
    y: float,
    z: float,
) -> str:
    element = atom_name.strip()[0]
    return (
        f"ATOM  {serial:5d} {atom_name:^4} {res_name:>3} {chain_id}"
        f"{res_id:4d}{ins_code:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00 20.00          {element:>2s}\n"
    )


if __name__ == "__main__":
    unittest.main()
