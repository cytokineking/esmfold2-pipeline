from __future__ import annotations

import hashlib
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from esmfold2_pipeline.artifact_layout import structure_relpath
from esmfold2_pipeline.artifacts import write_text_atomic
from esmfold2_pipeline.config import check_campaign_config, load_campaign_config
from esmfold2_pipeline.db import CampaignStore, connect_database
from esmfold2_pipeline.esm_adapter import DesignCandidateArtifact
from esmfold2_pipeline.execution import run_campaign
from esmfold2_pipeline.planning import plan_campaign
from esmfold2_pipeline.reports import inspect_campaign


class CampaignRunnerTest(unittest.TestCase):
    def test_plan_expands_seeds_and_run_is_idempotent(self) -> None:
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
  length: 70-90
campaign:
  num_designs: 2
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            plan = plan_campaign(config_path)
            self.assertEqual(plan.shard_count, 2)
            self.assertEqual(plan.config.seeds, (0, 1))
            self.assertEqual(plan.config.binder.scaffold, "miniprotein")
            self.assertEqual(plan.config.binder.length_range, (70, 90))
            self.assertTrue((campaign_dir / "config.yaml").exists())
            self.assertTrue((campaign_dir / "resolved_config.yaml").exists())
            self.assertTrue((campaign_dir / "campaign.sqlite").exists())

            conn = connect_database(campaign_dir / "campaign.sqlite")
            shards = conn.execute(
                "SELECT shard_id, seed, batch_index, status FROM shards ORDER BY shard_id"
            ).fetchall()
            conn.close()
            self.assertEqual(
                [(row["shard_id"], row["seed"], row["batch_index"], row["status"]) for row in shards],
                [
                    ("shard_000000", 0, 0, "pending"),
                    ("shard_000001", 1, 1, "pending"),
                ],
            )

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_design_artifact,
            ):
                result = run_campaign(campaign_dir, worker_id="test-runner", gpu_id="0")

            self.assertEqual(result.completed_shards, 2)
            self.assertFalse(result.skipped_no_pending)
            status = inspect_campaign(campaign_dir)
            self.assertEqual(status.shard_status_counts, {"completed": 2})
            self.assertEqual(status.candidate_status_counts, {"completed": 2})
            self.assertEqual(status.critic_status_counts, {"completed": 2})
            self.assertEqual(status.issues, [])

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=AssertionError("adapter should not run when no shards are pending"),
            ):
                second = run_campaign(campaign_dir, worker_id="test-runner-2")
            self.assertEqual(second.completed_shards, 0)
            self.assertTrue(second.skipped_no_pending)

    def test_plan_semantic_hash_allows_format_only_config_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
# comments and key order are not campaign semantics
target:
  name: ctla4
  sequence: GGGG
binder:
  scaffold: miniprotein
campaign:
  steps: 1
  seed_start: 0
  num_designs: 1
  critics:
    - fast
output: {campaign_dir}
""".lstrip()
            )
            plan_campaign(config_path)
            config_path.write_text(
                f"""
campaign:
  num_designs: 1
  critics: [fast]
  steps: 1
binder: {{scaffold: miniprotein}}
target:
  sequence: GGGG
  name: ctla4
output: {campaign_dir}  # same resolved campaign
""".lstrip()
            )

            plan_campaign(config_path)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                shard_count = conn.execute("SELECT COUNT(*) FROM shards").fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(shard_count, 1)

    def test_plan_semantic_hash_rejects_real_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: ctla4
  sequence: GGGG
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  steps: 1
output: {campaign_dir}
""".lstrip()
            )
            plan_campaign(config_path)
            config_path.write_text(
                f"""
target:
  name: ctla4
  sequence: GGGG
binder:
  scaffold: miniprotein
campaign:
  num_designs: 2
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            with self.assertRaisesRegex(ValueError, "different config"):
                plan_campaign(config_path)

    def test_plan_accepts_existing_raw_hash_when_resolved_config_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            first_config = f"""
target:
  name: ctla4
  sequence: GGGG
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - fast
  steps: 1
output: {campaign_dir}
""".lstrip()
            config_path.write_text(first_config)
            plan_campaign(config_path)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                conn.execute(
                    "UPDATE campaign SET config_hash = ? WHERE id = 1",
                    (hashlib.sha256(first_config.encode("utf-8")).hexdigest(),),
                )
                conn.commit()
            finally:
                conn.close()

            config_path.write_text(
                f"""
# old DB stores raw-byte hash, but resolved config is unchanged
campaign:
  critics: [fast]
  steps: 1
  num_designs: 1
binder:
  scaffold: miniprotein
target:
  sequence: GGGG
  name: ctla4
output: {campaign_dir}
""".lstrip()
            )

            plan_campaign(config_path)

    def test_run_campaign_recovers_stale_shards_before_claiming(self) -> None:
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

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            claim = store.claim_next_pending_shard(worker_id="preempted-worker")
            assert claim is not None
            conn.execute(
                """
                UPDATE shards
                SET heartbeat_at = '1970-01-01T00:00:00.000Z'
                WHERE shard_id = ?
                """,
                (claim.shard_id,),
            )
            conn.commit()
            conn.close()

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_design_artifact,
            ):
                result = run_campaign(
                    campaign_dir,
                    worker_id="resume-worker",
                    stale_after_seconds=1.0,
                )

            self.assertEqual(result.recovered_stale_shards, 1)
            self.assertEqual(result.completed_shards, 1)
            status = inspect_campaign(campaign_dir)
            self.assertEqual(status.shard_status_counts, {"completed": 1})
            self.assertEqual(status.attempt_status_counts, {"completed": 1, "stale": 1})

    def test_config_defaults_to_experimental_cutoff2025_models_and_entropy_hotspots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            config = load_campaign_config(config_path)
            self.assertEqual(
                config.inversion_model_name,
                "ESMFold2-Experimental-Cutoff2025",
            )
            self.assertEqual(config.critic_name, "ESMFold2-Experimental-Cutoff2025")
            self.assertEqual(config.hotspot_contact_weight, 2.0)
            self.assertEqual(config.hotspot_distogram_contact_cutoff_angstrom, 20.0)
            self.assertEqual(config.hotspot_critic_contact_cutoff_angstrom, 5.0)
            self.assertEqual(config.hotspot_contact_cutoff_angstrom, 5.0)
            self.assertEqual(config.hotspot_num_contacts, 1)
            self.assertEqual(config.hotspot_contact_probability_target, 0.6)
            self.assertEqual(config.hotspot_loss_mode, "entropy_hotspot")
            self.assertFalse(config.target_geometry_drift.enabled)
            self.assertEqual(config.target_geometry_drift.weight, 2.5)
            self.assertEqual(config.target_geometry_drift.tolerance_angstrom, 0.1)
            self.assertEqual(config.target_geometry_drift.stiffness_angstrom, 0.1)
            self.assertIsNone(config.target_geometry_drift.regions)

    def test_config_accepts_short_model_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  inversion_model: fast
  critics:
    - fast
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            config = load_campaign_config(config_path)

            self.assertEqual(config.inversion_model_name, "ESMFold2-Experimental-Fast")
            self.assertEqual(config.critic_name, "ESMFold2-Experimental-Fast")

    def test_config_rejects_removed_bindcraft_entropy_hotspot_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  steps: 1
loss:
  hotspot_loss_mode: bindcraft_entropy
output: {root / "campaign"}
""".lstrip()
            )

            with self.assertRaisesRegex(ValueError, "entropy_hotspot"):
                load_campaign_config(config_path)

    def test_config_rejects_removed_legacy_binder_name_and_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_binder_config = root / "legacy-binder.yaml"
            legacy_binder_config.write_text(
                f"""
target:
  name: ctla4
binder:
  name: minibinder
campaign:
  num_designs: 1
  steps: 1
output: {root / "legacy-binder"}
""".lstrip()
            )
            with self.assertRaisesRegex(ValueError, "binder.scaffold is required"):
                load_campaign_config(legacy_binder_config)

            legacy_seeds_config = root / "legacy-seeds.yaml"
            legacy_seeds_config.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: miniprotein
campaign:
  seeds: [0]
  steps: 1
output: {root / "legacy-seeds"}
""".lstrip()
            )
            with self.assertRaisesRegex(ValueError, "campaign.num_designs is required"):
                load_campaign_config(legacy_seeds_config)

    def test_config_rejects_structure_only_fields_without_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: ctla4
  hotspots: "A:88"
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            with self.assertRaisesRegex(ValueError, "target.hotspots require target.structure"):
                load_campaign_config(config_path)

    def test_config_rejects_target_geometry_drift_without_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  sequence: ACD
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  steps: 1
loss:
  target_geometry_drift:
    enabled: true
output: {root / "campaign"}
""".lstrip()
            )

            with self.assertRaisesRegex(ValueError, "target_geometry_drift"):
                load_campaign_config(config_path)

    def test_config_accepts_num_designs_seed_start_and_scfv_framework(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: scfv
  framework: trastuzumab
campaign:
  num_designs: 3
  seed_start: 10
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            config = load_campaign_config(config_path)
            self.assertEqual(config.seeds, (10, 11, 12))
            self.assertEqual(config.binder.scaffold, "scfv")
            self.assertEqual(config.binder.framework, "trastuzumab_framework_vhvl")
            self.assertIsNone(config.binder.length_range)
            self.assertEqual(config.binder_name, "trastuzumab_framework_vhvl")
            self.assertEqual(config.binder.frameworks[0].source, "builtin")
            self.assertIsNotNone(config.binder.frameworks[0].template)
            self.assertEqual(config.binder.frameworks[0].cdr_lengths["hcdr1"], (7, 9))
            resolved = config.to_resolved_dict()
            self.assertEqual(resolved["campaign"]["num_designs"], 3)
            self.assertEqual(resolved["campaign"]["seed_start"], 10)
            self.assertNotIn("seeds", resolved["campaign"])

    def test_config_and_plan_round_robin_multiple_scfv_frameworks(self) -> None:
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
    - belimumab_framework_vhvl
campaign:
  num_designs: 5
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            plan = plan_campaign(config_path)

            self.assertEqual(plan.shard_count, 5)
            self.assertIsNone(plan.config.binder.framework)
            self.assertEqual(
                plan.config.binder.framework_names,
                (
                    "trastuzumab_framework_vhvl",
                    "atezolizumab_framework_vhvl",
                    "belimumab_framework_vhvl",
                ),
            )
            self.assertEqual(
                plan.config.to_resolved_dict()["campaign"]["framework_schedule"],
                [
                    "trastuzumab_framework_vhvl",
                    "atezolizumab_framework_vhvl",
                    "belimumab_framework_vhvl",
                    "trastuzumab_framework_vhvl",
                    "atezolizumab_framework_vhvl",
                ],
            )

            conn = connect_database(campaign_dir / "campaign.sqlite")
            rows = conn.execute(
                "SELECT shard_id, seed, batch_index, binder_key "
                "FROM shards ORDER BY shard_id"
            ).fetchall()
            conn.close()
            self.assertEqual(
                [(row["shard_id"], row["seed"], row["batch_index"]) for row in rows],
                [
                    ("shard_000000", 0, 0),
                    ("shard_000001", 1, 1),
                    ("shard_000002", 2, 2),
                    ("shard_000003", 3, 3),
                    ("shard_000004", 4, 4),
                ],
            )
            self.assertIn("trastuzumab_framework_vhvl", rows[0]["binder_key"])
            self.assertIn("atezolizumab_framework_vhvl", rows[1]["binder_key"])
            self.assertIn("belimumab_framework_vhvl", rows[2]["binder_key"])

    def test_config_accepts_builtin_vhh_framework(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: vhh
  framework: caplacizumab
campaign:
  num_designs: 2
  seed_start: 5
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            config = load_campaign_config(config_path)
            framework = config.binder.frameworks[0]

            self.assertEqual(config.seeds, (5, 6))
            self.assertEqual(config.binder.scaffold, "vhh")
            self.assertEqual(config.binder.framework, "caplacizumab_framework_vhh")
            self.assertIsNone(config.binder.length_range)
            self.assertEqual(config.binder_name, "caplacizumab_framework_vhh")
            self.assertEqual(framework.source, "builtin")
            self.assertEqual(framework.cdr_names, ("cdr1", "cdr2", "cdr3"))
            self.assertEqual(framework.cdr_report_names, ("hcdr1", "hcdr2", "hcdr3"))
            self.assertEqual(framework.cdr_lengths["cdr3"], (18, 24))

    def test_config_and_plan_round_robin_multiple_vhh_frameworks(self) -> None:
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
    - ozoralizumab_tnf
    - vobarilizumab_il6r
campaign:
  num_designs: 5
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            plan = plan_campaign(config_path)

            self.assertEqual(plan.shard_count, 5)
            self.assertIsNone(plan.config.binder.framework)
            self.assertEqual(
                plan.config.binder.framework_names,
                (
                    "caplacizumab_framework_vhh",
                    "ozoralizumab_tnf_framework_vhh",
                    "vobarilizumab_il6r_framework_vhh",
                ),
            )
            self.assertEqual(
                plan.config.to_resolved_dict()["campaign"]["framework_schedule"],
                [
                    "caplacizumab_framework_vhh",
                    "ozoralizumab_tnf_framework_vhh",
                    "vobarilizumab_il6r_framework_vhh",
                    "caplacizumab_framework_vhh",
                    "ozoralizumab_tnf_framework_vhh",
                ],
            )

            conn = connect_database(campaign_dir / "campaign.sqlite")
            rows = conn.execute(
                "SELECT shard_id, seed, batch_index, binder_key "
                "FROM shards ORDER BY shard_id"
            ).fetchall()
            conn.close()
            self.assertIn("caplacizumab_framework_vhh", rows[0]["binder_key"])
            self.assertIn("ozoralizumab_tnf_framework_vhh", rows[1]["binder_key"])
            self.assertIn("vobarilizumab_il6r_framework_vhh", rows[2]["binder_key"])

    def test_run_campaign_passes_framework_specific_adapter_kwargs(self) -> None:
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
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            plan_campaign(config_path)
            calls: list[dict] = []

            def fake_scfv_artifact(**kwargs):
                calls.append(kwargs)
                return _fake_scfv_design_artifact(**kwargs)

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=fake_scfv_artifact,
            ):
                run_campaign(campaign_dir, worker_id="scfv-worker", gpu_id="0")

            self.assertEqual(
                [call["candidate_id"] for call in calls],
                [
                    "cd45_scfv_seed0",
                    "cd45_scfv_seed1",
                ],
            )
            self.assertEqual(
                [call["binder_name"] for call in calls],
                [
                    "trastuzumab_framework_vhvl",
                    "atezolizumab_framework_vhvl",
                ],
            )
            self.assertEqual(
                [call["binder_framework_name"] for call in calls],
                [
                    "trastuzumab_framework_vhvl",
                    "atezolizumab_framework_vhvl",
                ],
            )
            self.assertEqual(
                [call["binder_framework_source"] for call in calls],
                ["builtin", "builtin"],
            )
            self.assertTrue(
                all(call["binder_framework_template"] is not None for call in calls)
            )
            self.assertTrue(
                all(call["binder_framework_cdr_lengths"] is not None for call in calls)
            )

    def test_run_campaign_passes_vhh_framework_adapter_kwargs(self) -> None:
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
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            plan_campaign(config_path)
            calls: list[dict] = []

            def fake_vhh_artifact(**kwargs):
                calls.append(kwargs)
                return _fake_vhh_design_artifact(**kwargs)

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=fake_vhh_artifact,
            ):
                run_campaign(campaign_dir, worker_id="vhh-worker", gpu_id="0")

            self.assertEqual(
                [call["candidate_id"] for call in calls],
                [
                    "cd45_vhh_seed0",
                    "cd45_vhh_seed1",
                ],
            )
            self.assertEqual(
                [call["binder_name"] for call in calls],
                [
                    "caplacizumab_framework_vhh",
                    "vobarilizumab_il6r_framework_vhh",
                ],
            )
            self.assertEqual(
                [call["binder_scaffold"] for call in calls],
                ["vhh", "vhh"],
            )
            self.assertTrue(
                all(call["binder_framework_template"] is not None for call in calls)
            )
            self.assertTrue(
                all(call["binder_framework_cdr_lengths"] is not None for call in calls)
            )

    def test_run_campaign_passes_target_drift_to_scfv_adapter_path(self) -> None:
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
  scaffold: scfv
  framework: trastuzumab_framework_vhvl
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
loss:
  target_geometry_drift:
    enabled: true
    weight: 0.25
    tolerance_angstrom: 2.0
    stiffness_angstrom: 0.2
output: {campaign_dir}
""".lstrip()
            )

            plan_campaign(config_path)

            def fake_scfv_artifact(**kwargs):
                self.assertEqual(kwargs["binder_scaffold"], "scfv")
                self.assertIsNotNone(kwargs["structure_target"])
                self.assertTrue(kwargs["target_geometry_drift"].enabled)
                self.assertEqual(kwargs["target_geometry_drift"].weight, 0.25)
                self.assertEqual(
                    kwargs["target_geometry_drift"].tolerance_angstrom,
                    2.0,
                )
                self.assertEqual(
                    kwargs["target_geometry_drift"].stiffness_angstrom,
                    0.2,
                )
                self.assertIsNone(kwargs["target_geometry_drift"].regions)
                return _fake_scfv_design_artifact(**kwargs)

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=fake_scfv_artifact,
            ):
                result = run_campaign(campaign_dir, worker_id="scfv-drift-worker")

        self.assertEqual(result.completed_shards, 1)

    def test_config_accepts_custom_scfv_template_framework(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: cd45
binder:
  scaffold: scfv
  framework:
    name: lab_template
    template: EVQL{{hcdr1}}WVRQ{{hcdr2}}YYCAR{{hcdr3}}GGGGSQSV{{lcdr1}}WYQQ{{lcdr2}}FGG{{lcdr3}}
    cdr_lengths:
      hcdr1: 7-9
      hcdr2: 5
      hcdr3: {{min: 9, max: 12}}
      lcdr1: [11, 13]
      lcdr2: 7
      lcdr3: 9
campaign:
  num_designs: 1
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            config = load_campaign_config(config_path)
            framework = config.binder.frameworks[0]

            self.assertEqual(framework.name, "lab_template")
            self.assertEqual(framework.source, "template")
            self.assertEqual(framework.cdr_lengths["hcdr1"], (7, 9))
            self.assertEqual(framework.cdr_lengths["lcdr2"], (7, 7))
            self.assertEqual(config.binder_name, "lab_template")

    def test_config_rejects_malformed_custom_scfv_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: cd45
binder:
  scaffold: scfv
  framework:
    name: broken_template
    template: EVQL{{hcdr1}}WVRQ{{hcdr2}}YYCAR{{hcdr3}}GGGGSQSV{{lcdr1}}WYQQ{{lcdr2}}FGG{{lcdr3}}{{oops}}
    cdr_lengths:
      hcdr1: 7
      hcdr2: 5
      hcdr3: 9
      lcdr1: 11
      lcdr2: 7
      lcdr3: 9
campaign:
  num_designs: 1
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            with self.assertRaisesRegex(ValueError, "unsupported placeholders"):
                load_campaign_config(config_path)

    def test_config_rejects_custom_scfv_framework_name_colliding_with_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: cd45
binder:
  scaffold: scfv
  framework:
    name: trastuzumab
    template: EVQL{{hcdr1}}WVRQ{{hcdr2}}YYCAR{{hcdr3}}GGGGSQSV{{lcdr1}}WYQQ{{lcdr2}}FGG{{lcdr3}}
    cdr_lengths:
      hcdr1: 7
      hcdr2: 5
      hcdr3: 9
      lcdr1: 11
      lcdr2: 7
      lcdr3: 9
campaign:
  num_designs: 1
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            with self.assertRaisesRegex(ValueError, "must not collide"):
                load_campaign_config(config_path)

    def test_check_accepts_custom_scfv_sequence_with_explicit_cdr_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: cd45
binder:
  scaffold: scfv
  framework:
    name: lab_fixed_scfv
    sequence: EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSSGGGSGGGGSGGGGSGGGGSDIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK
    mutate: cdrs
    cdrs:
      hcdr1: 26-33
      hcdr2: 51-58
      hcdr3: 98-108
      lcdr1: 163-173
      lcdr2: 190-196
      lcdr3: 227-235
campaign:
  num_designs: 1
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            result = check_campaign_config(config_path)

            self.assertTrue(result.ok, result.errors)
            assert result.config is not None
            framework = result.config.binder.frameworks[0]
            self.assertEqual(framework.cdr_ranges["hcdr1"], (25, 33))
            self.assertEqual(framework.cdr_indices[:3], (25, 26, 27))
            self.assertEqual(
                result.config.to_resolved_dict()["binder"]["framework"]["cdrs"]["hcdr1"],
                "26-33",
            )

    def test_check_rejects_custom_scfv_sequence_without_explicit_cdrs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: cd45
binder:
  scaffold: scfv
  framework:
    name: lab_fixed_scfv
    sequence: EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSSGGGSGGGGSGGGGSGGGGSDIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK
    mutate: cdrs
campaign:
  num_designs: 1
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            result = check_campaign_config(config_path)

            self.assertFalse(result.ok)
            self.assertIn("cdrs must be a mapping", result.errors[0])

    def test_config_accepts_direct_target_sequence_without_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  sequence: acd
binder:
  scaffold: miniprotein
campaign:
  num_designs: 2
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            config = load_campaign_config(config_path)

            self.assertEqual(config.target_name, "sequence_target")
            self.assertEqual(config.target_sequence, "ACD")
            self.assertIsNone(config.target_structure)
            self.assertEqual(config.to_resolved_dict()["target"]["sequence"], "ACD")

    def test_config_rejects_target_sequence_with_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_test_pdb(target_path)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  structure: {target_path}
  chains: [A]
  sequence: ACD
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                load_campaign_config(config_path)

    def test_config_rejects_multiple_critics_for_current_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
    - ESMFold2-Experimental
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            with self.assertRaisesRegex(ValueError, "exactly one critic"):
                plan_campaign(config_path)

    def test_run_campaign_heartbeats_during_adapter_call(self) -> None:
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
                side_effect=_slow_fake_design_artifact,
            ):
                result = run_campaign(
                    campaign_dir,
                    worker_id="heartbeat-runner",
                    gpu_id="0",
                    heartbeat_interval_seconds=0.05,
                )

            self.assertEqual(result.completed_shards, 1)
            conn = connect_database(campaign_dir / "campaign.sqlite")
            row = conn.execute(
                """
                SELECT claimed_at, heartbeat_at, completed_at
                FROM shards
                WHERE shard_id = 'shard_000000'
                """
            ).fetchone()
            conn.close()
            self.assertIsNotNone(row["claimed_at"])
            self.assertIsNotNone(row["heartbeat_at"])
            self.assertNotEqual(row["claimed_at"], row["heartbeat_at"])
            self.assertLess(row["claimed_at"], row["heartbeat_at"])

    def test_run_campaign_passes_structure_target_to_adapter(self) -> None:
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
  conditioning:
    mode: distogram
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  inversion_model: ESMFold2-Experimental-Fast
  critics:
    - ESMFold2-Experimental
  steps: 1
loss:
  hotspot_contact_weight: 0.75
  hotspot_distogram_contact_cutoff_angstrom: 20.0
  hotspot_critic_contact_cutoff_angstrom: 4.5
  hotspot_num_contacts: 2
  hotspot_contact_probability_target: 0.7
  hotspot_loss_mode: entropy_hotspot
  target_geometry_drift:
    enabled: true
    weight: 0.5
    regions:
      A: all
output: {campaign_dir}
""".lstrip()
            )

            plan_campaign(config_path)

            def fake_artifact(**kwargs) -> DesignCandidateArtifact:
                self.assertIsNotNone(kwargs["structure_target"])
                self.assertEqual(kwargs["structure_target"].chains[0].sequence, "GS")
                self.assertEqual(kwargs["conditioning_mode"], "distogram")
                self.assertEqual(kwargs["inversion_model_name"], "ESMFold2-Experimental-Fast")
                self.assertEqual(kwargs["critic_name"], "ESMFold2-Experimental")
                self.assertEqual(kwargs["hotspot_contact_weight"], 0.75)
                self.assertEqual(
                    kwargs["hotspot_distogram_contact_cutoff_angstrom"],
                    20.0,
                )
                self.assertEqual(
                    kwargs["hotspot_critic_contact_cutoff_angstrom"],
                    4.5,
                )
                self.assertEqual(kwargs["hotspot_num_contacts"], 2)
                self.assertEqual(kwargs["hotspot_contact_probability_target"], 0.7)
                self.assertEqual(kwargs["hotspot_loss_mode"], "entropy_hotspot")
                self.assertTrue(kwargs["target_geometry_drift"].enabled)
                self.assertEqual(kwargs["target_geometry_drift"].weight, 0.5)
                self.assertEqual(
                    kwargs["target_geometry_drift"].tolerance_angstrom,
                    0.1,
                )
                self.assertEqual(
                    kwargs["target_geometry_drift"].stiffness_angstrom,
                    0.1,
                )
                self.assertEqual(
                    kwargs["target_geometry_drift"].regions,
                    {"A": ("all",)},
                )
                self.assertEqual(kwargs["binder_length_range"], (60, 200))
                return _fake_design_artifact(**kwargs)

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=fake_artifact,
            ):
                result = run_campaign(campaign_dir, worker_id="structure-runner")

            self.assertEqual(result.completed_shards, 1)

    def test_run_campaign_passes_assembly_conditioning_to_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_multichain_test_pdb(target_path)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  structure: {target_path}
  chains: [A, B]
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

            plan_campaign(config_path)

            def fake_artifact(**kwargs) -> DesignCandidateArtifact:
                self.assertIsNotNone(kwargs["structure_target"])
                self.assertEqual(
                    [chain.canonical_chain_id for chain in kwargs["structure_target"].chains],
                    ["A", "B"],
                )
                self.assertEqual(kwargs["conditioning_mode"], "distogram")
                self.assertTrue(kwargs["conditioning_assembly"])
                self.assertEqual(kwargs["conditioning_chain_pairs"], (("A", "B"),))
                return _fake_design_artifact(**kwargs)

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=fake_artifact,
            ):
                result = run_campaign(campaign_dir, worker_id="assembly-runner")

            self.assertEqual(result.completed_shards, 1)

    def test_multichain_distogram_conditioning_defaults_to_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_multichain_test_pdb(target_path)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  structure: {target_path}
  chains: [A, B]
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

            config = load_campaign_config(config_path)
            self.assertIsNotNone(config.target_structure)
            assert config.target_structure is not None
            self.assertEqual(config.target_structure.conditioning_mode, "distogram")
            self.assertTrue(config.target_structure.conditioning_assembly)
            self.assertIsNone(config.target_structure.conditioning_chain_pairs)
            self.assertTrue(config.target_geometry_drift.enabled)

    def test_structure_conditioning_and_geometry_drift_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_multichain_test_pdb(target_path)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  structure: {target_path}
  chains: [A, B]
  conditioning:
    mode: none
loss:
  target_geometry_drift:
    enabled: false
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            config = load_campaign_config(config_path)
            self.assertIsNotNone(config.target_structure)
            assert config.target_structure is not None
            self.assertEqual(config.target_structure.conditioning_mode, "none")
            self.assertFalse(config.target_structure.conditioning_assembly)
            self.assertFalse(config.target_geometry_drift.enabled)

    def test_explicit_multichain_assembly_false_is_honored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_multichain_test_pdb(target_path)
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  structure: {target_path}
  chains: [A, B]
  conditioning:
    mode: distogram
    assembly: false
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            config = load_campaign_config(config_path)
            self.assertIsNotNone(config.target_structure)
            assert config.target_structure is not None
            self.assertFalse(config.target_structure.conditioning_assembly)

    def test_run_campaign_passes_direct_target_sequence_to_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  name: custom_sequence_target
  sequence: ACDEFGHIK
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

            def fake_artifact(**kwargs) -> DesignCandidateArtifact:
                self.assertEqual(kwargs["target_name"], "custom_sequence_target")
                self.assertEqual(kwargs["target_sequence"], "ACDEFGHIK")
                self.assertIsNone(kwargs["structure_target"])
                self.assertEqual(kwargs["conditioning_mode"], "none")
                return _fake_design_artifact(**kwargs)

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=fake_artifact,
            ):
                result = run_campaign(campaign_dir, worker_id="sequence-runner")

            self.assertEqual(result.completed_shards, 1)

    def test_check_rejects_multichar_chain_ids_for_pdb_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.cif"
            _write_test_cif_with_multichar_chain(target_path)
            campaign_dir = root / "campaign"
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
target:
  structure: {target_path}
  chains: [AA]
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

            result = check_campaign_config(config_path)

            self.assertFalse(result.ok)
            self.assertIn("one-character IDs", result.errors[0])

    def test_plan_resolves_relative_structure_path_for_campaign_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_test_pdb(target_path)
            config_dir = root / "configs"
            config_dir.mkdir()
            campaign_dir = root / "campaign"
            config_path = config_dir / "config.yaml"
            config_path.write_text(
                f"""
target:
  structure: ../target.pdb
  chains: [A]
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

            plan_campaign(config_path)
            resolved = load_campaign_config(campaign_dir / "resolved_config.yaml")

            self.assertIsNotNone(resolved.target_structure)
            assert resolved.target_structure is not None
            self.assertEqual(resolved.target_structure.path, target_path.resolve())


def _fake_design_artifact(**kwargs) -> DesignCandidateArtifact:
    root = Path(kwargs["campaign_dir"])
    cid = kwargs["candidate_id"]
    critic_name = kwargs["critic_name"]
    sequence = f"BINDER{kwargs['seed']}"
    structure_path = structure_relpath(kwargs.get("artifact_stem", cid))
    write_text_atomic(root / structure_path, "HEADER    FAKE CAMPAIGN COMPLEX\nEND\n")
    return DesignCandidateArtifact(
        candidate_id=cid,
        designed_sequence=sequence,
        sequence_path=None,
        critic_name=critic_name,
        structure_path=structure_path.as_posix(),
        design_metrics={"steps": kwargs["steps"], "seed": kwargs["seed"]},
        critic_metrics={"iptm": 0.5 + kwargs["seed"], "steps": kwargs["steps"]},
    )


def _fake_scfv_design_artifact(**kwargs) -> DesignCandidateArtifact:
    root = Path(kwargs["campaign_dir"])
    seed = int(kwargs["seed"])
    cid = kwargs["candidate_id"]
    critic_name = kwargs["critic_name"]
    structure_path = structure_relpath(kwargs.get("artifact_stem", cid))
    write_text_atomic(
        root / structure_path,
        f"HEADER    FAKE SCFV COMPLEX {seed}\nEND\n",
    )
    return DesignCandidateArtifact(
        candidate_id=cid,
        designed_sequence=f"TARGET|SCFV{seed}",
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
            "framework": kwargs["binder_framework_name"],
            "framework_name": kwargs["binder_framework_name"],
            "framework_source": kwargs["binder_framework_source"],
            "final_loss": 0.4,
        },
        critic_metrics={
            "iptm": 0.7 + (0.01 * seed),
            "iptm_scope": "binder_target",
            "distogram_iptm_proxy": 0.5 + (0.01 * seed),
            "cdr_distogram_iptm_proxy": 0.6 + (0.01 * seed),
            "ptm": 0.5,
            "plddt_complex": 80.0,
            "plddt_binder": 75.0,
            "plddt_target": 85.0,
            "steps": kwargs["steps"],
        },
    )


def _fake_vhh_design_artifact(**kwargs) -> DesignCandidateArtifact:
    root = Path(kwargs["campaign_dir"])
    seed = int(kwargs["seed"])
    cid = kwargs["candidate_id"]
    critic_name = kwargs["critic_name"]
    structure_path = structure_relpath(kwargs.get("artifact_stem", cid))
    write_text_atomic(
        root / structure_path,
        f"HEADER    FAKE VHH COMPLEX {seed}\nEND\n",
    )
    return DesignCandidateArtifact(
        candidate_id=cid,
        designed_sequence=f"TARGET|VHH{seed}",
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
            "framework": kwargs["binder_framework_name"],
            "framework_name": kwargs["binder_framework_name"],
            "framework_source": kwargs["binder_framework_source"],
            "final_loss": 0.4,
        },
        critic_metrics={
            "iptm": 0.7 + (0.01 * seed),
            "iptm_scope": "binder_target",
            "distogram_iptm_proxy": 0.5 + (0.01 * seed),
            "cdr_distogram_iptm_proxy": 0.6 + (0.01 * seed),
            "ptm": 0.5,
            "plddt_complex": 80.0,
            "plddt_binder": 75.0,
            "plddt_target": 85.0,
            "steps": kwargs["steps"],
        },
    )


def _slow_fake_design_artifact(**kwargs) -> DesignCandidateArtifact:
    time.sleep(0.2)
    root = Path(kwargs["campaign_dir"])
    conn = connect_database(root / "campaign.sqlite")
    row = conn.execute(
        """
        SELECT status, claimed_at, heartbeat_at
        FROM shards
        WHERE shard_id = ?
        """,
        (kwargs["shard_id"],),
    ).fetchone()
    conn.close()
    assert row["status"] == "running"
    assert row["claimed_at"] is not None
    assert row["heartbeat_at"] is not None
    assert row["heartbeat_at"] > row["claimed_at"]
    return _fake_design_artifact(**kwargs)


def _write_test_pdb(path: Path) -> None:
    lines = [
        _pdb_atom_line(1, "N", "GLY", "A", 1, "", 0.0, 0.0, 0.0),
        _pdb_atom_line(2, "CA", "GLY", "A", 1, "", 1.0, 0.0, 0.0),
        _pdb_atom_line(3, "C", "GLY", "A", 1, "", 2.0, 0.0, 0.0),
        _pdb_atom_line(4, "O", "GLY", "A", 1, "", 2.5, 0.5, 0.0),
        _pdb_atom_line(5, "N", "SER", "A", 2, "", 3.0, 0.0, 0.0),
        _pdb_atom_line(6, "CA", "SER", "A", 2, "", 4.0, 0.0, 0.0),
        _pdb_atom_line(7, "CB", "SER", "A", 2, "", 4.0, 1.0, 0.0),
        _pdb_atom_line(8, "C", "SER", "A", 2, "", 5.0, 0.0, 0.0),
        _pdb_atom_line(9, "O", "SER", "A", 2, "", 5.5, 0.5, 0.0),
    ]
    path.write_text("".join(lines))


def _write_multichain_test_pdb(path: Path) -> None:
    lines = [
        _pdb_atom_line(1, "N", "GLY", "A", 1, "", 0.0, 0.0, 0.0),
        _pdb_atom_line(2, "CA", "GLY", "A", 1, "", 1.0, 0.0, 0.0),
        _pdb_atom_line(3, "C", "GLY", "A", 1, "", 2.0, 0.0, 0.0),
        _pdb_atom_line(4, "O", "GLY", "A", 1, "", 2.5, 0.5, 0.0),
        _pdb_atom_line(5, "N", "SER", "A", 2, "", 3.0, 0.0, 0.0),
        _pdb_atom_line(6, "CA", "SER", "A", 2, "", 4.0, 0.0, 0.0),
        _pdb_atom_line(7, "CB", "SER", "A", 2, "", 4.0, 1.0, 0.0),
        _pdb_atom_line(8, "C", "SER", "A", 2, "", 5.0, 0.0, 0.0),
        _pdb_atom_line(9, "O", "SER", "A", 2, "", 5.5, 0.5, 0.0),
        _pdb_atom_line(10, "N", "GLY", "B", 1, "", 0.0, 8.0, 0.0),
        _pdb_atom_line(11, "CA", "GLY", "B", 1, "", 1.0, 8.0, 0.0),
        _pdb_atom_line(12, "C", "GLY", "B", 1, "", 2.0, 8.0, 0.0),
        _pdb_atom_line(13, "O", "GLY", "B", 1, "", 2.5, 8.5, 0.0),
        _pdb_atom_line(14, "N", "THR", "B", 2, "", 3.0, 8.0, 0.0),
        _pdb_atom_line(15, "CA", "THR", "B", 2, "", 4.0, 8.0, 0.0),
        _pdb_atom_line(16, "CB", "THR", "B", 2, "", 4.0, 9.0, 0.0),
        _pdb_atom_line(17, "C", "THR", "B", 2, "", 5.0, 8.0, 0.0),
        _pdb_atom_line(18, "O", "THR", "B", 2, "", 5.5, 8.5, 0.0),
    ]
    path.write_text("".join(lines))


def _write_test_cif_with_multichar_chain(path: Path) -> None:
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
    for atom_name, element, x, y, z in [
        ("N", "N", 0.0, 0.0, 0.0),
        ("CA", "C", 1.0, 0.0, 0.0),
        ("C", "C", 2.0, 0.0, 0.0),
        ("O", "O", 2.5, 0.5, 0.0),
    ]:
        lines.append(
            " ".join(
                [
                    "ATOM",
                    str(atom_id),
                    element,
                    atom_name,
                    ".",
                    "GLY",
                    "AA",
                    "1",
                    "1",
                    "?",
                    f"{x:.3f}",
                    f"{y:.3f}",
                    f"{z:.3f}",
                    "1.00",
                    "20.00",
                    "?",
                    "1",
                    "GLY",
                    "AA",
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
