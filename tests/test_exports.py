from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from esmfold2_pipeline.artifact_layout import structure_relpath
from esmfold2_pipeline.artifacts import write_text_atomic
from esmfold2_pipeline.esm_adapter import DesignCandidateArtifact
from esmfold2_pipeline.execution import run_campaign
from esmfold2_pipeline.planning import plan_campaign
from esmfold2_pipeline.reports import aggregate_campaign, export_campaign, select_campaign


class ExportReportsTest(unittest.TestCase):
    def test_aggregate_select_and_export_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: miniprotein
campaign:
  num_designs: 3
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            plan_campaign(config_path)
            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_design_artifact,
            ):
                run_campaign(campaign_dir, worker_id="export-test", gpu_id="0")

            aggregate = aggregate_campaign(campaign_dir)
            self.assertEqual(aggregate.row_count, 3)
            self.assertEqual(
                aggregate.metrics_csv,
                campaign_dir / "esmfold2" / "metrics_all.csv",
            )
            self.assertEqual(
                aggregate.summary_json,
                campaign_dir / "esmfold2" / "campaign_summary.json",
            )

            self.assertIn(
                "# campaign,target_name=ctla4,binder_scaffold=miniprotein",
                aggregate.metrics_csv.read_text().splitlines()[0],
            )
            metric_rows = _read_csv(aggregate.metrics_csv)
            self.assertEqual(len(metric_rows), 3)
            self.assertEqual(
                list(metric_rows[0].keys()),
                [
                    "candidate_id",
                    "seed",
                    "designed_sequence",
                    "binder_length",
                    "binder_chain_id",
                    "iptm",
                    "distogram_iptm_proxy",
                    "ptm",
                    "plddt_complex",
                    "plddt_binder",
                    "plddt_target",
                    "final_loss",
                    "iptm_scope",
                    "complex_iptm",
                    "critic_name",
                    "structure_path",
                ],
            )
            self.assertEqual(metric_rows[0]["candidate_id"], "ctla4_mp_seed2")
            self.assertEqual(metric_rows[0]["seed"], "2")
            self.assertEqual(metric_rows[0]["binder_length"], "19")
            self.assertEqual(metric_rows[0]["binder_chain_id"], "B")
            self.assertEqual(metric_rows[0]["iptm_scope"], "binder_target")
            self.assertEqual(metric_rows[0]["complex_iptm"], "0.97")
            self.assertEqual(metric_rows[0]["ptm"], "0.6")
            self.assertEqual(metric_rows[0]["plddt_complex"], "82")
            self.assertEqual(metric_rows[0]["plddt_binder"], "72")
            self.assertEqual(metric_rows[0]["plddt_target"], "92")
            self.assertEqual(metric_rows[0]["final_loss"], "0.25")
            self.assertNotIn("hotspot_pass", metric_rows[0])
            self.assertNotIn("hotspot_distance_angstrom", metric_rows[0])
            summary = json.loads(aggregate.summary_json.read_text())
            self.assertEqual(summary["metrics"]["completed_metric_rows"], 3)
            self.assertEqual(
                summary["metrics"]["iptm_scope_counts"],
                {"binder_target": 3},
            )
            self.assertEqual(summary["metrics"]["hotspot"]["pass_count"], 0)
            selection = select_campaign(campaign_dir, max_designs=10)
            self.assertEqual(selection.candidate_count, 2)
            self.assertEqual(selection.selected_count, 2)
            self.assertIn(
                "# campaign,target_name=ctla4,binder_scaffold=miniprotein",
                selection.ranked_csv.read_text().splitlines()[0],
            )
            ranked_rows = _read_csv(selection.ranked_csv)
            self.assertEqual(
                [(row["rank"], row["candidate_id"]) for row in ranked_rows],
                [
                    ("1", "ctla4_mp_seed2"),
                    ("2", "ctla4_mp_seed0"),
                ],
            )

            export = export_campaign(campaign_dir, max_designs=1)
            self.assertEqual(export.selected_count, 1)
            self.assertEqual(export.copied_files, 1)
            self.assertTrue((export.selected_dir / "ctla4_mp_seed2.pdb").exists())
            self.assertFalse((export.selected_dir / "ctla4_mp_seed1.pdb").exists())

            manifest_rows = _read_csv(export.manifest_csv)
            self.assertEqual(len(manifest_rows), 1)
            self.assertEqual(manifest_rows[0]["candidate_id"], "ctla4_mp_seed2")
            self.assertEqual(manifest_rows[0]["seed"], "2")
            self.assertEqual(manifest_rows[0]["binder_length"], "19")
            self.assertEqual(manifest_rows[0]["binder_chain_id"], "B")
            self.assertEqual(manifest_rows[0]["iptm_scope"], "binder_target")
            self.assertEqual(manifest_rows[0]["complex_iptm"], "0.97")
            self.assertEqual(manifest_rows[0]["plddt_binder"], "72")
            self.assertNotIn("hotspot_pass", manifest_rows[0])
            self.assertNotIn("hotspot_distance_angstrom", manifest_rows[0])
            export_summary = json.loads(export.summary_json.read_text())
            self.assertEqual(export_summary["export"]["selected_count"], 1)
            self.assertEqual(export_summary["export"]["copied_files"], 1)
            self.assertIn(
                "FAKE CAMPAIGN COMPLEX 2",
                (export.selected_dir / "ctla4_mp_seed2.pdb").read_text(),
            )

            second_export = export_campaign(campaign_dir, max_designs=1)
            self.assertEqual(second_export.selected_count, 1)
            self.assertEqual(second_export.copied_files, 1)

    def test_target_geometry_drift_metadata_and_region_metrics_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_test_pdb(target_path)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  structure: {target_path}
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
    weight: 0.5
    tolerance_angstrom: 2.0
    stiffness_angstrom: 0.2
    regions:
      A:
        - 1-2
output: {campaign_dir}
""".lstrip()
            )

            plan_campaign(config_path)
            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_target_geometry_drift_design_artifact,
            ):
                run_campaign(campaign_dir, worker_id="drift-export-test")

            aggregate = aggregate_campaign(campaign_dir)
            metadata_line = aggregate.metrics_csv.read_text().splitlines()[0]
            self.assertIn("target_geometry_drift_enabled=true", metadata_line)
            self.assertIn("target_geometry_drift_weight=0.5", metadata_line)
            self.assertIn(
                "target_geometry_drift_tolerance_angstrom=2",
                metadata_line,
            )
            self.assertIn(
                "target_geometry_drift_stiffness_angstrom=0.2",
                metadata_line,
            )
            self.assertIn("target_geometry_drift_regions=A:1-2", metadata_line)
            metric_rows = _read_csv(aggregate.metrics_csv)
            self.assertEqual(
                metric_rows[0]["target_geometry_drift_distance_rmse"],
                "1.25",
            )
            self.assertEqual(
                metric_rows[0]["target_geometry_drift_aligned_rmsd"],
                "0.4",
            )

            selection = select_campaign(campaign_dir)
            ranked_rows = _read_csv(selection.ranked_csv)
            self.assertEqual(
                ranked_rows[0]["target_geometry_drift_distance_rmse"],
                "1.25",
            )
            export = export_campaign(campaign_dir)
            manifest_rows = _read_csv(export.manifest_csv)
            self.assertEqual(
                manifest_rows[0]["target_geometry_drift_aligned_rmsd"],
                "0.4",
            )

    def test_scfv_exports_include_framework_and_cdr_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: cd45
binder:
  scaffold: scfv
  frameworks:
    - trastuzumab_framework_vhvl
    - atezolizumab_framework_vhvl
campaign:
  num_designs: 2
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            plan_campaign(config_path)
            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_scfv_design_artifact,
            ):
                run_campaign(campaign_dir, worker_id="scfv-export-test", gpu_id="0")

            aggregate = aggregate_campaign(campaign_dir)
            first_line = aggregate.metrics_csv.read_text().splitlines()[0]
            self.assertIn("binder_scaffold=scfv", first_line)
            self.assertNotIn("binder_type=scfv", first_line)
            self.assertIn(
                "frameworks=trastuzumab_framework_vhvl;atezolizumab_framework_vhvl",
                first_line,
            )

            metric_rows = _read_csv(aggregate.metrics_csv)
            self.assertEqual(
                list(metric_rows[0].keys()),
                [
                    "candidate_id",
                    "framework",
                    "seed",
                    "designed_sequence",
                    "cdrh1",
                    "cdrh2",
                    "cdrh3",
                    "cdrl1",
                    "cdrl2",
                    "cdrl3",
                    "binder_length",
                    "binder_chain_id",
                    "iptm",
                    "cdr_distogram_iptm_proxy",
                    "distogram_iptm_proxy",
                    "ptm",
                    "plddt_complex",
                    "plddt_binder",
                    "plddt_target",
                    "final_loss",
                    "iptm_scope",
                    "complex_iptm",
                    "critic_name",
                    "structure_path",
                ],
            )
            by_id = {row["candidate_id"]: row for row in metric_rows}
            self.assertEqual(
                by_id["cd45_scfv_seed0"]["framework"],
                "trastuzumab_framework_vhvl",
            )
            self.assertEqual(by_id["cd45_scfv_seed0"]["cdrh1"], "H1_0")
            self.assertEqual(by_id["cd45_scfv_seed0"]["cdrh2"], "H2_0")
            self.assertEqual(by_id["cd45_scfv_seed0"]["cdrh3"], "H3_0")
            self.assertEqual(by_id["cd45_scfv_seed0"]["cdrl1"], "L1_0")
            self.assertEqual(by_id["cd45_scfv_seed0"]["cdrl2"], "L2_0")
            self.assertEqual(by_id["cd45_scfv_seed0"]["cdrl3"], "L3_0")
            self.assertEqual(
                by_id["cd45_scfv_seed0"]["cdr_distogram_iptm_proxy"],
                "0.82",
            )

            selection = select_campaign(campaign_dir, max_designs=10)
            ranked_rows = _read_csv(selection.ranked_csv)
            self.assertEqual(
                [row["candidate_id"] for row in ranked_rows],
                [
                    "cd45_scfv_seed1",
                    "cd45_scfv_seed0",
                ],
            )

            export = export_campaign(campaign_dir, max_designs=1)
            manifest_rows = _read_csv(export.manifest_csv)
            self.assertEqual(
                manifest_rows[0]["candidate_id"],
                "cd45_scfv_seed1",
            )
            self.assertEqual(manifest_rows[0]["framework"], "atezolizumab_framework_vhvl")
            self.assertEqual(manifest_rows[0]["cdrh3"], "H3_1")
            self.assertEqual(manifest_rows[0]["cdr_distogram_iptm_proxy"], "0.9")

    def test_vhh_exports_include_only_heavy_cdr_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: cd45
binder:
  scaffold: vhh
  frameworks:
    - caplacizumab
    - vobarilizumab_il6r
campaign:
  num_designs: 2
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            plan_campaign(config_path)
            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_vhh_design_artifact,
            ):
                run_campaign(campaign_dir, worker_id="vhh-export-test", gpu_id="0")

            aggregate = aggregate_campaign(campaign_dir)
            first_line = aggregate.metrics_csv.read_text().splitlines()[0]
            self.assertIn("binder_scaffold=vhh", first_line)
            self.assertIn(
                "frameworks=caplacizumab_framework_vhh;vobarilizumab_il6r_framework_vhh",
                first_line,
            )

            metric_rows = _read_csv(aggregate.metrics_csv)
            self.assertEqual(
                list(metric_rows[0].keys()),
                [
                    "candidate_id",
                    "framework",
                    "seed",
                    "designed_sequence",
                    "hcdr1",
                    "hcdr2",
                    "hcdr3",
                    "binder_length",
                    "binder_chain_id",
                    "iptm",
                    "cdr_distogram_iptm_proxy",
                    "distogram_iptm_proxy",
                    "ptm",
                    "plddt_complex",
                    "plddt_binder",
                    "plddt_target",
                    "final_loss",
                    "iptm_scope",
                    "complex_iptm",
                    "critic_name",
                    "structure_path",
                ],
            )
            self.assertNotIn("lcdr1", metric_rows[0])
            self.assertNotIn("cdrl1", metric_rows[0])
            by_id = {row["candidate_id"]: row for row in metric_rows}
            self.assertEqual(by_id["cd45_vhh_seed0"]["hcdr1"], "H1_0")
            self.assertEqual(by_id["cd45_vhh_seed0"]["hcdr2"], "H2_0")
            self.assertEqual(by_id["cd45_vhh_seed0"]["hcdr3"], "H3_0")

            selection = select_campaign(campaign_dir, max_designs=10)
            ranked_rows = _read_csv(selection.ranked_csv)
            self.assertEqual(
                [row["candidate_id"] for row in ranked_rows],
                [
                    "cd45_vhh_seed1",
                    "cd45_vhh_seed0",
                ],
            )
            self.assertNotIn("lcdr1", ranked_rows[0])
            self.assertNotIn("cdrl1", ranked_rows[0])

            export = export_campaign(campaign_dir, max_designs=1)
            manifest_rows = _read_csv(export.manifest_csv)
            self.assertEqual(manifest_rows[0]["candidate_id"], "cd45_vhh_seed1")
            self.assertEqual(
                manifest_rows[0]["framework"],
                "vobarilizumab_il6r_framework_vhh",
            )
            self.assertEqual(manifest_rows[0]["hcdr3"], "H3_1")
            self.assertNotIn("lcdr1", manifest_rows[0])
            self.assertNotIn("cdrl1", manifest_rows[0])

    def test_scfv_hotspot_exports_prefer_cdr_restricted_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: cd45
binder:
  scaffold: scfv
  framework: trastuzumab_framework_vhvl
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            plan_campaign(config_path)
            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_scfv_cdr_hotspot_design_artifact,
            ):
                run_campaign(campaign_dir, worker_id="scfv-hotspot-export-test")

            aggregate = aggregate_campaign(campaign_dir)
            rows = _read_csv(aggregate.metrics_csv)

            self.assertEqual(rows[0]["hotspot_pass"], "false")
            self.assertEqual(rows[0]["hotspot_distance_angstrom"], "20")

            selection = select_campaign(campaign_dir)
            self.assertEqual(selection.candidate_count, 0)

    def test_min_iptm_filter_can_select_empty_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
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
output: {campaign_dir}
""".lstrip()
            )

            plan_campaign(config_path)
            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_design_artifact,
            ):
                run_campaign(campaign_dir)

            selection = select_campaign(campaign_dir, min_iptm=0.99)
            self.assertEqual(selection.candidate_count, 0)
            self.assertEqual(selection.selected_count, 0)
            self.assertEqual(_read_csv(selection.ranked_csv), [])

    def test_mosaic_cdr_metrics_are_exported_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: cd45
binder:
  scaffold: vhh
  framework: caplacizumab
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
loss:
  binder_target_contact_mode: mosaic_cdr
  mosaic_cdr_contact_weight: 0.8
  mosaic_framework_contact_penalty_weight: 0.2
  mosaic_framework_contact_penalty_scope: target_all
output: {campaign_dir}
""".lstrip()
            )

            def fake_mosaic_artifact(**kwargs) -> DesignCandidateArtifact:
                artifact = _fake_vhh_design_artifact(**kwargs)
                design_metrics = dict(artifact.design_metrics)
                design_metrics.update(
                    {
                        "binder_target_contact_mode": "mosaic_cdr",
                        "mosaic_cdr_contact_loss_enabled": True,
                        "mosaic_cdr_contact_weight": 0.8,
                        "mosaic_cdr_contact_cutoff_angstrom": 22.0,
                        "mosaic_cdr_num_target_contacts": 3,
                        "mosaic_cdr_contact_scope": "target_all",
                        "mosaic_cdr_contact_probability_mean": 0.7,
                        "mosaic_cdr_contact_probability_min": 0.6,
                        "mosaic_cdr_contact_probability_max": 0.8,
                        "mosaic_cdr_contact_loss": 0.04,
                        "mosaic_framework_contact_penalty_enabled": True,
                        "mosaic_framework_contact_penalty_weight": 0.2,
                        "mosaic_framework_contact_penalty_scope": "target_all",
                        "mosaic_framework_contact_penalty_target_scope": (
                            "target_all"
                        ),
                        "mosaic_framework_contact_probability_mean": 0.1,
                        "mosaic_framework_contact_probability_max": 0.18,
                        "mosaic_framework_contact_penalty_loss": 0.0,
                    }
                )
                return DesignCandidateArtifact(
                    candidate_id=artifact.candidate_id,
                    designed_sequence=artifact.designed_sequence,
                    sequence_path=artifact.sequence_path,
                    critic_name=artifact.critic_name,
                    structure_path=artifact.structure_path,
                    design_metrics=design_metrics,
                    critic_metrics=artifact.critic_metrics,
                )

            plan_campaign(config_path)
            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=fake_mosaic_artifact,
            ):
                run_campaign(campaign_dir, worker_id="mosaic-export-test")

            aggregate = aggregate_campaign(campaign_dir)
            metadata_line = aggregate.metrics_csv.read_text().splitlines()[0]
            rows = _read_csv(aggregate.metrics_csv)

            self.assertIn("binder_target_contact_mode=mosaic_cdr", metadata_line)
            self.assertIn(
                "mosaic_framework_contact_penalty_scope=target_all",
                metadata_line,
            )
            self.assertEqual(rows[0]["binder_target_contact_mode"], "mosaic_cdr")
            self.assertEqual(rows[0]["mosaic_cdr_contact_scope"], "target_all")
            self.assertEqual(rows[0]["mosaic_cdr_contact_probability_mean"], "0.7")
            self.assertEqual(
                rows[0]["mosaic_framework_contact_penalty_scope"],
                "target_all",
            )
            self.assertEqual(
                rows[0]["mosaic_framework_contact_penalty_target_scope"],
                "target_all",
            )
            self.assertEqual(rows[0]["mosaic_framework_contact_penalty_loss"], "0")

    def test_hotspot_filter_prefers_lower_iptm_contact_over_high_iptm_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: il2_asn88_il2rb
binder:
  scaffold: miniprotein
campaign:
  num_designs: 2
  critics:
    - ESMFold2-Experimental
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            plan_campaign(config_path)
            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_hotspot_design_artifact,
            ):
                run_campaign(campaign_dir)

            selection = select_campaign(campaign_dir, max_designs=10)
            ranked_rows = _read_csv(selection.ranked_csv)
            self.assertEqual(selection.candidate_count, 1)
            self.assertEqual(selection.selected_count, 1)
            self.assertEqual(ranked_rows[0]["candidate_id"], "il2_asn88_il2rb_mp_seed1")
            self.assertEqual(ranked_rows[0]["hotspot_pass"], "true")
            self.assertEqual(ranked_rows[0]["hotspot_distance_angstrom"], "3.2")

            legacy_selection = select_campaign(
                campaign_dir,
                max_designs=10,
                require_hotspot_contact="never",
            )
            legacy_rows = _read_csv(legacy_selection.ranked_csv)
            self.assertEqual(
                [(row["rank"], row["candidate_id"]) for row in legacy_rows],
                [
                    ("1", "il2_asn88_il2rb_mp_seed0"),
                    ("2", "il2_asn88_il2rb_mp_seed1"),
                ],
            )


def _fake_design_artifact(**kwargs) -> DesignCandidateArtifact:
    root = Path(kwargs["campaign_dir"])
    seed = int(kwargs["seed"])
    cid = kwargs["candidate_id"]
    critic_name = kwargs["critic_name"]
    sequence = "ACDEFGHIKLMNPQRSTVWY" if seed == 0 else "VVVVVVVVVVACDEFGHIK"
    structure_path = structure_relpath(kwargs.get("artifact_stem", cid))
    write_text_atomic(
        root / structure_path,
        f"HEADER    FAKE CAMPAIGN COMPLEX {seed}\nEND\n",
    )
    return DesignCandidateArtifact(
        candidate_id=cid,
        designed_sequence=sequence,
        sequence_path=None,
        critic_name=critic_name,
        structure_path=structure_path.as_posix(),
        design_metrics={
            "target_name": "ctla4",
            "steps": kwargs["steps"],
            "seed": seed,
            "binder_chain_id": "B",
            "binder_scaffold": "miniprotein",
            "binder_type": "mp",
            "final_loss": 0.75 - (0.25 * seed),
        },
        critic_metrics={
            "iptm": 0.5 + (0.1 * seed),
            "iptm_scope": "binder_target",
            "complex_iptm": 0.95 + (0.01 * seed),
            "ptm": 0.4 + (0.1 * seed),
            "plddt": 80.0 + seed,
            "plddt_complex": 80.0 + seed,
            "plddt_binder": 70.0 + seed,
            "plddt_target": 90.0 + seed,
            "distogram_iptm_proxy": 0.45 + (0.1 * seed),
            "steps": kwargs["steps"],
        },
    )


def _fake_scfv_design_artifact(**kwargs) -> DesignCandidateArtifact:
    root = Path(kwargs["campaign_dir"])
    seed = int(kwargs["seed"])
    cid = kwargs["candidate_id"]
    critic_name = kwargs["critic_name"]
    framework = kwargs["binder_framework_name"]
    sequence = f"SCFV{seed}"
    structure_path = structure_relpath(kwargs.get("artifact_stem", cid))
    write_text_atomic(
        root / structure_path,
        f"HEADER    FAKE SCFV COMPLEX {seed}\nEND\n",
    )
    return DesignCandidateArtifact(
        candidate_id=cid,
        designed_sequence=sequence,
        sequence_path=None,
        critic_name=critic_name,
        structure_path=structure_path.as_posix(),
        design_metrics={
            "target_name": "cd45",
            "steps": kwargs["steps"],
            "seed": seed,
            "binder_chain_id": "B",
            "binder_scaffold": "scfv",
            "binder_type": "scfv",
            "framework": framework,
            "framework_name": framework,
            "framework_source": kwargs["binder_framework_source"],
            "cdr_sequences": {
                "hcdr1": f"H1_{seed}",
                "hcdr2": f"H2_{seed}",
                "hcdr3": f"H3_{seed}",
                "lcdr1": f"L1_{seed}",
                "lcdr2": f"L2_{seed}",
                "lcdr3": f"L3_{seed}",
            },
            "final_loss": 0.25 + (0.1 * seed),
        },
        critic_metrics={
            "iptm": 0.75 + (0.05 * seed),
            "iptm_scope": "binder_target",
            "complex_iptm": 0.85 + (0.03 * seed),
            "distogram_iptm_proxy": 0.7 + (0.05 * seed),
            "cdr_distogram_iptm_proxy": 0.82 + (0.08 * seed),
            "ptm": 0.6,
            "plddt_complex": 81.0,
            "plddt_binder": 73.0,
            "plddt_target": 88.0,
            "steps": kwargs["steps"],
        },
    )


def _fake_vhh_design_artifact(**kwargs) -> DesignCandidateArtifact:
    root = Path(kwargs["campaign_dir"])
    seed = int(kwargs["seed"])
    cid = kwargs["candidate_id"]
    critic_name = kwargs["critic_name"]
    framework = kwargs["binder_framework_name"]
    sequence = f"VHH{seed}"
    structure_path = structure_relpath(kwargs.get("artifact_stem", cid))
    write_text_atomic(
        root / structure_path,
        f"HEADER    FAKE VHH COMPLEX {seed}\nEND\n",
    )
    return DesignCandidateArtifact(
        candidate_id=cid,
        designed_sequence=sequence,
        sequence_path=None,
        critic_name=critic_name,
        structure_path=structure_path.as_posix(),
        design_metrics={
            "target_name": "cd45",
            "steps": kwargs["steps"],
            "seed": seed,
            "binder_chain_id": "B",
            "binder_scaffold": "vhh",
            "binder_type": "vhh",
            "framework": framework,
            "framework_name": framework,
            "framework_source": kwargs["binder_framework_source"],
            "cdr_sequences": {
                "cdr1": f"H1_{seed}",
                "cdr2": f"H2_{seed}",
                "cdr3": f"H3_{seed}",
            },
            "final_loss": 0.25 + (0.1 * seed),
        },
        critic_metrics={
            "iptm": 0.8,
            "iptm_scope": "binder_target",
            "complex_iptm": 0.85 + (0.03 * seed),
            "distogram_iptm_proxy": 0.9 - (0.8 * seed),
            "cdr_distogram_iptm_proxy": 0.1 + (0.8 * seed),
            "ptm": 0.6,
            "plddt_complex": 81.0,
            "plddt_binder": 73.0,
            "plddt_target": 88.0,
            "steps": kwargs["steps"],
        },
    )


def _fake_target_geometry_drift_design_artifact(**kwargs) -> DesignCandidateArtifact:
    artifact = _fake_design_artifact(**kwargs)
    critic_metrics = dict(artifact.critic_metrics)
    critic_metrics.update(
        {
            "target_geometry_drift_distance_rmse": 1.25,
            "target_geometry_drift_aligned_rmsd": 0.4,
            "target_geometry_drift_residue_count": 2,
        }
    )
    return DesignCandidateArtifact(
        candidate_id=artifact.candidate_id,
        designed_sequence=artifact.designed_sequence,
        sequence_path=artifact.sequence_path,
        critic_name=artifact.critic_name,
        structure_path=artifact.structure_path,
        design_metrics=artifact.design_metrics,
        critic_metrics=critic_metrics,
    )


def _fake_scfv_cdr_hotspot_design_artifact(**kwargs) -> DesignCandidateArtifact:
    artifact = _fake_scfv_design_artifact(**kwargs)
    critic_metrics = dict(artifact.critic_metrics)
    critic_metrics.update(
        {
            "hotspot_contact_cutoff_angstrom": 5.0,
            "hotspot_satisfaction": 1.0,
            "hotspot_min_heavy_atom_distance_min": 3.0,
            "cdr_hotspot_contact_cutoff_angstrom": 5.0,
            "cdr_hotspot_pass": False,
            "cdr_hotspot_distance_angstrom": 20.0,
            "cdr_hotspot_satisfaction": 0.0,
            "cdr_hotspot_min_heavy_atom_distance_min": 20.0,
        }
    )
    return DesignCandidateArtifact(
        candidate_id=artifact.candidate_id,
        designed_sequence=artifact.designed_sequence,
        sequence_path=artifact.sequence_path,
        critic_name=artifact.critic_name,
        structure_path=artifact.structure_path,
        design_metrics=artifact.design_metrics,
        critic_metrics=critic_metrics,
    )


def _fake_hotspot_design_artifact(**kwargs) -> DesignCandidateArtifact:
    root = Path(kwargs["campaign_dir"])
    seed = int(kwargs["seed"])
    cid = kwargs["candidate_id"]
    critic_name = kwargs["critic_name"]
    sequence = f"BINDER{seed}"
    structure_path = structure_relpath(kwargs.get("artifact_stem", cid))
    write_text_atomic(
        root / structure_path,
        f"HEADER    FAKE HOTSPOT COMPLEX {seed}\nEND\n",
    )
    is_hit = seed == 1
    return DesignCandidateArtifact(
        candidate_id=cid,
        designed_sequence=sequence,
        sequence_path=None,
        critic_name=critic_name,
        structure_path=structure_path.as_posix(),
        design_metrics={"steps": kwargs["steps"], "seed": seed},
        critic_metrics={
            "iptm": 0.9 if seed == 0 else 0.2,
            "hotspot_contact_cutoff_angstrom": 5.0,
            "hotspot_min_heavy_atom_distance_min": 17.5 if not is_hit else 3.2,
            "hotspot_satisfaction": 0.0 if not is_hit else 1.0,
            "steps": kwargs["steps"],
        },
    )


def _write_test_pdb(path: Path) -> None:
    lines = [
        _pdb_atom_line(1, "N", "GLY", "A", 1, 0.0, 0.0, 0.0),
        _pdb_atom_line(2, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0),
        _pdb_atom_line(3, "C", "GLY", "A", 1, 0.0, 0.0, 0.0),
        _pdb_atom_line(4, "N", "SER", "A", 2, 1.0, 0.0, 0.0),
        _pdb_atom_line(5, "CA", "SER", "A", 2, 1.0, 0.0, 0.0),
        _pdb_atom_line(6, "CB", "SER", "A", 2, 1.0, 1.0, 0.0),
        _pdb_atom_line(7, "C", "SER", "A", 2, 2.0, 0.0, 0.0),
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


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    assert path is not None
    text = "".join(
        line
        for line in path.read_text().splitlines(keepends=True)
        if not line.startswith("#")
    )
    return list(csv.DictReader(text.splitlines()))


if __name__ == "__main__":
    unittest.main()
