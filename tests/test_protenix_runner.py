from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from esmfold2_pipeline.artifact_layout import structure_relpath
from esmfold2_pipeline.artifacts import write_text_atomic
from esmfold2_pipeline.db import CampaignStore, connect_database
from esmfold2_pipeline.esm_adapter import DesignCandidateArtifact
from esmfold2_pipeline.execution import run_campaign
from esmfold2_pipeline.planning import plan_campaign
from esmfold2_pipeline.reports import report_validation
from esmfold2_pipeline.validation import (
    ProtenixRunnerConfig,
    ProtenixTaskInput,
    ProtenixTemplateSpec,
    ValidationPlanConfig,
    build_protenix_input_json,
    plan_validation_tasks,
    run_local_protenix_validation,
    run_multi_validation,
    scoped_pair_metric,
)
from esmfold2_pipeline.validation.protenix import (
    build_protenix_command,
    _resolve_ipsae_script,
    _run_protenix_command,
    _target_template_spec,
    _tasks_requiring_binder_msas,
)
from esmfold2_pipeline.validation.msa import MsaPair
from esmfold2_pipeline.validation.workers import (
    _cleanup_failed_worker_scratch,
    _should_cleanup_failed_worker_scratch,
)


class ProtenixRunnerTest(unittest.TestCase):
    def test_default_ipsae_script_is_bundled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            script = _resolve_ipsae_script(ProtenixRunnerConfig())

        self.assertIsNotNone(script)
        assert script is not None
        self.assertEqual(script.name, "ipsae.py")
        self.assertTrue(script.exists())
        self.assertEqual(script.parent.name, "validation")
        self.assertIn("script for calculating the ipSAE score", script.read_text())

    def test_build_protenix_input_json_places_binder_before_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task = ProtenixTaskInput(
                validation_id="val_one",
                candidate_id="cand_one",
                model_name="protenix-v2",
                selection_rank=1,
                designed_sequence="ACD",
                target_sequences=("GG", "SS"),
                target_labels=("chain_x", "chain_y"),
                seed=101,
                binder_scaffold="miniprotein",
                framework=None,
            )

            input_json, sample_names, chain_maps = build_protenix_input_json(
                [task],
                Path(tmpdir) / "input",
            )

            payload = json.loads(input_json.read_text())
            self.assertEqual(sample_names, {"val_one": "val_one_pred"})
            self.assertEqual(chain_maps["val_one"], {"binder": ["A"], "target": ["B", "C"]})
            self.assertEqual(
                [
                    item["proteinChain"]["sequence"]
                    for item in payload[0]["sequences"]
                ],
                ["ACD", "GG", "SS"],
            )
            self.assertEqual(
                [
                    item["proteinChain"]["id"]
                    for item in payload[0]["sequences"]
                ],
                [["A"], ["B"], ["C"]],
            )

    def test_build_protenix_input_json_attaches_structural_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_template = root / "target_template.cif"
            framework_template = root / "framework_template.cif"
            target_template.write_text("data_target\n")
            framework_template.write_text("data_framework\n")
            task = ProtenixTaskInput(
                validation_id="val_template",
                candidate_id="cand_template",
                model_name="protenix-v2",
                selection_rank=1,
                designed_sequence="ACD",
                target_sequences=("GG",),
                target_labels=("chain_x",),
                seed=101,
                binder_scaffold="scfv",
                framework="anifrolumab_framework_vhvl",
                framework_source="builtin",
            )

            input_json, _sample_names, _chain_maps = build_protenix_input_json(
                [task],
                root / "input",
                template_specs={
                    "val_template": (
                        ProtenixTemplateSpec(
                            path=framework_template,
                            chain_ids=("A",),
                            template_ids=("A",),
                            source="framework:anifrolumab",
                        ),
                        ProtenixTemplateSpec(
                            path=target_template,
                            chain_ids=("B",),
                            template_ids=("A",),
                            source="target",
                        ),
                    )
                },
            )

            sample = json.loads(input_json.read_text())[0]
            self.assertEqual(len(sample["templates"]), 2)
            self.assertEqual(
                sample["templates"][0]["chain_id"],
                ["A"],
            )
            self.assertEqual(
                sample["templates"][1]["chain_id"],
                ["B"],
            )
            self.assertEqual(sample["templates"][1]["template_id"], ["A"])
            for template in sample["templates"]:
                path = Path(template["cif"])
                self.assertTrue(path.exists())
                self.assertEqual(path.parent.name, "templates")

    def test_build_protenix_command_enables_template_flag_when_requested(self) -> None:
        command = build_protenix_command(
            input_json="/tmp/input.json",
            output_dir="/tmp/out",
            config=ProtenixRunnerConfig(
                protenix_command=("python", "-m", "runner.inference"),
            ),
            use_msa=False,
            use_template=True,
        )

        self.assertEqual(command[command.index("--use_msa") + 1], "false")
        self.assertEqual(command[command.index("--use_template") + 1], "true")

    def test_target_template_rejects_sequence_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_dir = root / "target"
            target_dir.mkdir()
            (target_dir / "normalized_target.cif").write_text("data_target\n")
            (target_dir / "chain_summary.json").write_text(
                json.dumps(
                    {
                        "chains": [
                            {
                                "canonical_chain_id": "A",
                                "sequence": "MISMATCH",
                            }
                        ]
                    }
                )
            )
            task = ProtenixTaskInput(
                validation_id="val_template",
                candidate_id="cand_template",
                model_name="protenix-v2",
                selection_rank=1,
                designed_sequence="ACD",
                target_sequences=("GG",),
                target_labels=("A",),
                seed=101,
                binder_scaffold="miniprotein",
                framework=None,
            )

            with self.assertRaisesRegex(ValueError, "sequence does not match"):
                _target_template_spec(root, task)

    def test_framework_template_suppresses_binder_msa_even_when_target_msa_active(self) -> None:
        task = ProtenixTaskInput(
            validation_id="val_vhh",
            candidate_id="cand_vhh",
            model_name="protenix-v2",
            selection_rank=1,
            designed_sequence="ACD",
            target_sequences=("GG",),
            target_labels=("target",),
            seed=101,
            binder_scaffold="vhh",
            framework="caplacizumab_framework_vhh",
            framework_source="builtin",
        )
        template_specs = {
            "val_vhh": (
                ProtenixTemplateSpec(
                    path=Path("/tmp/framework.cif"),
                    chain_ids=("A",),
                    template_ids=("A",),
                    source="framework:caplacizumab",
                ),
            )
        }

        self.assertEqual(
            _tasks_requiring_binder_msas(
                [task],
                config=ProtenixRunnerConfig(
                    target_msa_mode="server",
                    msa_server_url="https://msa.example",
                ),
                template_specs=template_specs,
                target_msas={"val_vhh": (None,)},
            ),
            (),
        )
        self.assertEqual(
            _tasks_requiring_binder_msas(
                [task],
                config=ProtenixRunnerConfig(
                    target_msa_mode="server",
                    msa_server_url="https://msa.example",
                    use_msa=True,
                ),
                template_specs=template_specs,
                target_msas={"val_vhh": (None,)},
            ),
            (task,),
        )

    def test_vhh_and_scfv_templates_allow_provided_target_msas_without_binder_msas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_template = root / "target_template.cif"
            vhh_template = root / "vhh_framework.cif"
            scfv_template = root / "scfv_framework.cif"
            target_template.write_text("data_target\n")
            vhh_template.write_text("data_vhh\n")
            scfv_template.write_text("data_scfv\n")
            tasks = (
                ProtenixTaskInput(
                    validation_id="val_vhh",
                    candidate_id="cand_vhh",
                    model_name="protenix-v2",
                    selection_rank=1,
                    designed_sequence="ACD",
                    target_sequences=("GGGG",),
                    target_labels=("target",),
                    seed=101,
                    binder_scaffold="vhh",
                    framework="caplacizumab_framework_vhh",
                    framework_source="builtin",
                ),
                ProtenixTaskInput(
                    validation_id="val_scfv",
                    candidate_id="cand_scfv",
                    model_name="protenix-v2",
                    selection_rank=2,
                    designed_sequence="EFGH",
                    target_sequences=("GGGG",),
                    target_labels=("target",),
                    seed=102,
                    binder_scaffold="scfv",
                    framework="trastuzumab_framework_vhvl",
                    framework_source="builtin",
                ),
            )
            template_specs = {
                "val_vhh": (
                    ProtenixTemplateSpec(
                        path=target_template,
                        chain_ids=("B",),
                        template_ids=("A",),
                        source="target",
                    ),
                    ProtenixTemplateSpec(
                        path=vhh_template,
                        chain_ids=("A",),
                        template_ids=("A",),
                        source="framework:caplacizumab",
                    ),
                ),
                "val_scfv": (
                    ProtenixTemplateSpec(
                        path=target_template,
                        chain_ids=("B",),
                        template_ids=("A",),
                        source="target",
                    ),
                    ProtenixTemplateSpec(
                        path=scfv_template,
                        chain_ids=("A",),
                        template_ids=("A",),
                        source="framework:trastuzumab",
                    ),
                ),
            }
            target_msas = {
                task.validation_id: (
                    MsaPair(
                        pairing=">query\nGGGG\n",
                        non_pairing=">query\nGGGG\n>hit\nGGGA\n",
                        source="provided",
                    ),
                )
                for task in tasks
            }

            input_json, _sample_names, _chain_maps = build_protenix_input_json(
                tasks,
                root / "input",
                target_msas=target_msas,
                template_specs=template_specs,
            )

            payload = json.loads(input_json.read_text())
            self.assertEqual(len(payload), 2)
            self.assertEqual(
                _tasks_requiring_binder_msas(
                    tasks,
                    config=ProtenixRunnerConfig(
                        target_msa_mode="provided",
                        target_msa_dir=root,
                    ),
                    template_specs=template_specs,
                    target_msas=target_msas,
                ),
                (),
            )
            for sample in payload:
                binder_chain = sample["sequences"][0]["proteinChain"]
                target_chain = sample["sequences"][1]["proteinChain"]
                self.assertNotIn("pairedMsaPath", binder_chain)
                self.assertNotIn("unpairedMsaPath", binder_chain)
                self.assertTrue(Path(target_chain["pairedMsaPath"]).exists())
                self.assertTrue(Path(target_chain["unpairedMsaPath"]).exists())
                self.assertIn(
                    ">hit\nGGGA\n",
                    Path(target_chain["unpairedMsaPath"]).read_text(),
                )
                self.assertEqual(len(sample["templates"]), 2)
                self.assertEqual(
                    sorted(
                        tuple(template["chain_id"])
                        for template in sample["templates"]
                    ),
                    [("A",), ("B",)],
                )

    def test_scoped_pair_metric_ignores_global_and_target_target_pairs(self) -> None:
        metric = scoped_pair_metric(
            {
                "iptm": 0.99,
                "chain_pair_iptm": [
                    [0.0, 0.80, 0.40],
                    [0.80, 0.0, 0.97],
                    [0.40, 0.97, 0.0],
                ],
            },
            keys=("chain_pair_iptm",),
            chain_role_map={"binder": ["A"], "target": ["B", "C"]},
        )

        assert metric is not None
        self.assertAlmostEqual(metric["value"], 0.40)
        self.assertAlmostEqual(metric["mean"], 0.60)
        self.assertAlmostEqual(metric["max"], 0.80)
        self.assertEqual(len(metric["pairs"]), 2)

    def test_local_runner_promotes_outputs_and_records_scoped_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = _run_fake_campaign(root)
            fake_protenix = _write_fake_protenix(root, missing_scoped_iptm=False)
            plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(
                    model_name="protenix-v2",
                    min_validation_iptm=0.75,
                ),
            )

            result = run_local_protenix_validation(
                campaign_dir,
                worker_id="protenix-test",
                gpu_id="0",
                config=ProtenixRunnerConfig(
                    protenix_command=(sys.executable, str(fake_protenix)),
                    seeds=(101,),
                    n_sample=2,
                    min_validation_iptm=0.75,
                    timeout_seconds=10,
                ),
            )

            self.assertEqual(result.completed_tasks, 1)
            self.assertEqual(result.failed_tasks, 0)
            self.assertEqual(result.recorded_structures, 2)
            self.assertFalse((campaign_dir / ".scratch").exists())
            self.assertFalse((campaign_dir / ".scratch" / "protenix_validation").exists())

            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                task = conn.execute("SELECT * FROM validation_tasks").fetchone()
                self.assertEqual(task["status"], "completed")
                self.assertAlmostEqual(task["iptm"], 0.82)
                self.assertAlmostEqual(task["ipsae"], 0.61)
                self.assertTrue(task["output_structure_path"].startswith(
                    "validation/protenix_v2/structures/passing/"
                ))
                self.assertTrue((campaign_dir / task["output_structure_path"]).exists())

                task_metrics = json.loads(task["metrics_json"])
                self.assertEqual(task_metrics["validation_global_iptm"], 0.99)
                self.assertEqual(task_metrics["validation_metric_scope"], "binder_target")
                self.assertEqual(
                    task_metrics["validation_chain_role_map"],
                    {"binder": ["A"], "target": ["B"]},
                )

                structures = conn.execute(
                    """
                    SELECT *
                    FROM validation_structures
                    ORDER BY sample_rank
                    """
                ).fetchall()
                self.assertEqual(
                    [row["status"] for row in structures],
                    ["passing", "rejected"],
                )
                self.assertEqual(
                    [row["scoped_iptm"] for row in structures],
                    [0.82, 0.21],
                )
                self.assertEqual(
                    [row["scoped_ipsae"] for row in structures],
                    [0.61, 0.12],
                )
                for row in structures:
                    self.assertTrue((campaign_dir / row["structure_path"]).exists())
                    self.assertNotIn("/.staging/", row["structure_path"])

                attempts = conn.execute(
                    "SELECT status, exit_code FROM attempts WHERE stage = 'validation'"
                ).fetchall()
                self.assertEqual([(row["status"], row["exit_code"]) for row in attempts], [("completed", 0)])
            finally:
                conn.close()

    def test_local_runner_reports_retryable_attempt_failure_as_nonterminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = _run_fake_campaign(root)
            fake_protenix = _write_flaky_fake_protenix(root)
            plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(
                    model_name="protenix-v2",
                    max_attempts=3,
                ),
            )

            result = run_local_protenix_validation(
                campaign_dir,
                worker_id="protenix-flaky-test",
                gpu_id="0",
                config=ProtenixRunnerConfig(
                    protenix_command=(sys.executable, str(fake_protenix)),
                    seeds=(101,),
                    n_sample=1,
                    timeout_seconds=10,
                ),
            )

            self.assertEqual(result.completed_tasks, 1)
            self.assertEqual(result.failed_tasks, 0)
            self.assertEqual(result.retryable_failed_attempts, 1)
            self.assertEqual(result.recorded_structures, 1)

            report = report_validation(campaign_dir)
            self.assertEqual(report.task_rows, 1)
            self.assertEqual(report.structure_rows, 1)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                task = conn.execute(
                    """
                    SELECT status, attempt_count, error_message
                    FROM validation_tasks
                    """
                ).fetchone()
                self.assertEqual(task["status"], "completed")
                self.assertEqual(task["attempt_count"], 2)
                self.assertIsNone(task["error_message"])

                attempts = conn.execute(
                    """
                    SELECT status, exit_code
                    FROM attempts
                    WHERE stage = 'validation'
                    ORDER BY attempt_id
                    """
                ).fetchall()
                self.assertEqual(
                    [(row["status"], row["exit_code"]) for row in attempts],
                    [("failed", 1), ("completed", 0)],
                )
            finally:
                conn.close()

    def test_keep_debug_records_validation_attempt_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = _run_fake_campaign(root)
            fake_protenix = _write_fake_protenix(root, missing_scoped_iptm=False)
            plan_validation_tasks(campaign_dir)

            result = run_local_protenix_validation(
                campaign_dir,
                worker_id="debug-protenix-test",
                gpu_id="0",
                config=ProtenixRunnerConfig(
                    protenix_command=(sys.executable, str(fake_protenix)),
                    seeds=(101,),
                    n_sample=1,
                    timeout_seconds=10,
                    keep_debug=True,
                ),
            )

            self.assertEqual(result.completed_tasks, 1)
            debug_paths = sorted(
                (campaign_dir / "validation" / "protenix_v2" / "debug").glob("*.json")
            )
            self.assertEqual(len(debug_paths), 1)
            payload = json.loads(debug_paths[0].read_text())
            self.assertEqual(payload["worker_id"], "debug-protenix-test")
            self.assertEqual(payload["gpu_id"], "0")
            self.assertTrue(Path(payload["scratch_dir"]).exists())
            self.assertTrue(Path(payload["input_json_path"]).exists())
            self.assertTrue(Path(payload["output_dir"]).exists())
            self.assertEqual(payload["protenix_returncode"], 0)
            self.assertEqual(len(payload["attempts"]), 1)
            self.assertTrue(payload["attempts"][0]["candidate_id"])
            self.assertGreaterEqual(payload["attempts"][0]["attempt_id"], 1)

    def test_local_runner_rejects_structure_below_min_validation_ipsae(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = _run_fake_campaign(root)
            fake_protenix = _write_fake_protenix(root, missing_scoped_iptm=False)
            plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(
                    model_name="protenix-v2",
                    min_validation_iptm=0.75,
                    min_validation_ipsae=0.70,
                ),
            )

            result = run_local_protenix_validation(
                campaign_dir,
                worker_id="protenix-test",
                gpu_id="0",
                config=ProtenixRunnerConfig(
                    protenix_command=(sys.executable, str(fake_protenix)),
                    seeds=(101,),
                    n_sample=1,
                    min_validation_iptm=0.75,
                    min_validation_ipsae=0.70,
                    timeout_seconds=10,
                ),
            )

            self.assertEqual(result.completed_tasks, 1)
            self.assertEqual(result.failed_tasks, 0)
            self.assertEqual(result.recorded_structures, 1)
            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                task = conn.execute("SELECT * FROM validation_tasks").fetchone()
                self.assertEqual(task["status"], "completed")
                self.assertTrue(task["output_structure_path"].startswith(
                    "validation/protenix_v2/structures/rejected/"
                ))
                task_metrics = json.loads(task["metrics_json"])
                self.assertAlmostEqual(task_metrics["validation_iptm"], 0.82)
                self.assertAlmostEqual(task_metrics["validation_ipSAE"], 0.61)
                self.assertAlmostEqual(task_metrics["min_validation_ipSAE"], 0.70)
                self.assertFalse(task_metrics["validation_ipSAE_pass"])
                self.assertFalse(task_metrics["validation_passed"])
                self.assertIn("validation_ipSAE", task_metrics["fail_reason"])
                self.assertIn("below threshold 0.7000", task_metrics["fail_reason"])

                structure = conn.execute("SELECT * FROM validation_structures").fetchone()
                self.assertEqual(structure["status"], "rejected")
                self.assertAlmostEqual(structure["scoped_ipsae"], 0.61)
                structure_metrics = json.loads(structure["metrics_json"])
                self.assertFalse(structure_metrics["validation_ipSAE_pass"])
            finally:
                conn.close()

    def test_local_runner_selects_lower_iptm_sample_that_passes_min_validation_ipsae(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = _run_fake_campaign(root)
            fake_protenix = _write_fake_ipsae_threshold_protenix(root)
            plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(
                    model_name="protenix-v2",
                    min_validation_iptm=0.75,
                    min_validation_ipsae=0.60,
                ),
            )

            result = run_local_protenix_validation(
                campaign_dir,
                worker_id="protenix-test",
                gpu_id="0",
                config=ProtenixRunnerConfig(
                    protenix_command=(sys.executable, str(fake_protenix)),
                    seeds=(101,),
                    n_sample=2,
                    min_validation_iptm=0.75,
                    min_validation_ipsae=0.60,
                    timeout_seconds=10,
                ),
            )

            self.assertEqual(result.completed_tasks, 1)
            self.assertEqual(result.failed_tasks, 0)
            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                task = conn.execute("SELECT * FROM validation_tasks").fetchone()
                self.assertEqual(task["status"], "completed")
                self.assertAlmostEqual(task["iptm"], 0.80)
                self.assertAlmostEqual(task["ipsae"], 0.64)
                self.assertTrue(
                    task["output_structure_path"].endswith("sample01.cif")
                )
                task_metrics = json.loads(task["metrics_json"])
                self.assertTrue(task_metrics["validation_ipSAE_pass"])
                self.assertTrue(task_metrics["validation_passed"])

                structures = conn.execute(
                    """
                    SELECT status, sample_rank, scoped_iptm, scoped_ipsae, metrics_json
                    FROM validation_structures
                    ORDER BY sample_rank
                    """
                ).fetchall()
                self.assertEqual(
                    [row["status"] for row in structures],
                    ["rejected", "passing"],
                )
                first_metrics = json.loads(structures[0]["metrics_json"])
                second_metrics = json.loads(structures[1]["metrics_json"])
                self.assertFalse(first_metrics["validation_ipSAE_pass"])
                self.assertTrue(second_metrics["validation_ipSAE_pass"])
            finally:
                conn.close()

            report = report_validation(campaign_dir)
            with report.structures_manifest_csv.open() as handle:
                rows = list(csv.DictReader(handle))
            rows_by_sample = {row["sample_rank"]: row for row in rows}
            self.assertEqual(
                [
                    rows_by_sample["0"]["validator_ipsae_pass"],
                    rows_by_sample["1"]["validator_ipsae_pass"],
                ],
                ["false", "true"],
            )
            self.assertEqual(
                [
                    rows_by_sample["0"]["min_validator_ipsae"],
                    rows_by_sample["1"]["min_validator_ipsae"],
                ],
                ["0.6", "0.6"],
            )

    def test_local_runner_calculates_ipsae_from_full_data_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = _run_fake_campaign(root)
            fake_protenix = _write_fake_protenix_without_summary_ipsae(root)
            fake_ipsae = _write_fake_ipsae_script(root)
            plan_validation_tasks(campaign_dir)

            result = run_local_protenix_validation(
                campaign_dir,
                worker_id="protenix-test",
                gpu_id="0",
                config=ProtenixRunnerConfig(
                    protenix_command=(sys.executable, str(fake_protenix)),
                    ipsae_script_path=fake_ipsae,
                    seeds=(101,),
                    n_sample=1,
                    timeout_seconds=10,
                ),
            )

            self.assertEqual(result.completed_tasks, 1)
            self.assertEqual(result.failed_tasks, 0)
            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                task = conn.execute("SELECT * FROM validation_tasks").fetchone()
                self.assertEqual(task["status"], "completed")
                self.assertAlmostEqual(task["ipsae"], 0.73)
                metrics = json.loads(task["metrics_json"])
                self.assertAlmostEqual(metrics["validation_ipSAE"], 0.73)
                self.assertAlmostEqual(metrics["validation_ipSAE_min"], 0.35)
                self.assertAlmostEqual(metrics["validation_ipSAE_max"], 0.45)
                self.assertEqual(metrics["validation_ipSAE_source_key"], "ipsae.py")
                self.assertEqual(metrics["validation_ipSAE_adapter"], "ipsae.py")
                self.assertAlmostEqual(metrics["validation_ipSAE_d0chn"], 0.55)
                self.assertAlmostEqual(metrics["validation_pDockQ2"], 0.67)
                self.assertNotIn("validation_ipSAE_fallback", metrics)

                structure = conn.execute(
                    "SELECT scoped_ipsae, metrics_json FROM validation_structures"
                ).fetchone()
                self.assertAlmostEqual(structure["scoped_ipsae"], 0.73)
                structure_metrics = json.loads(structure["metrics_json"])
                self.assertEqual(
                    structure_metrics["validation_ipSAE_pairs"][0]["binder_chain"],
                    "A",
                )
                self.assertEqual(
                    structure_metrics["validation_ipSAE_pairs"][0]["target_chain"],
                    "B",
                )
            finally:
                conn.close()

    def test_validation_task_cannot_complete_before_selected_cif_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = _run_fake_campaign(Path(tmpdir))
            plan_validation_tasks(campaign_dir)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            try:
                claim = store.claim_next_pending_validation_tasks(
                    worker_id="validation-test",
                    batch_size=1,
                    hostname="localhost",
                    pid=1234,
                    gpu_id="0",
                )[0]
                with self.assertRaisesRegex(
                    ValueError,
                    "selected CIF is promoted and recorded",
                ):
                    store.complete_validation_task(
                        validation_id=claim.validation_id,
                        attempt_id=claim.attempt_id,
                        output_structure_path=(
                            "validation/protenix_v2/structures/passing/not_recorded.cif"
                        ),
                        metrics={"validation_iptm": 0.8, "validation_ipSAE": 0.7},
                    )
            finally:
                conn.close()

    def test_promoted_structure_survives_stale_claim_before_task_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = _run_fake_campaign(Path(tmpdir))
            plan_validation_tasks(campaign_dir)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            try:
                first_claim = store.claim_next_pending_validation_tasks(
                    worker_id="validation-test",
                    batch_size=1,
                    hostname="localhost",
                    pid=1234,
                    gpu_id="0",
                )[0]
                promoted_path = (
                    Path("validation")
                    / "protenix_v2"
                    / "structures"
                    / "passing"
                    / "crash_window__protenix_v2__sample00.cif"
                )
                write_text_atomic(campaign_dir / promoted_path, "data_fake\n#\n")
                store.record_validation_structure(
                    validation_id=first_claim.validation_id,
                    structure_id="seed101_sample0",
                    candidate_id=first_claim.candidate_id,
                    model_name=first_claim.model_name,
                    seed=101,
                    sample_rank=0,
                    status="passing",
                    structure_path=promoted_path.as_posix(),
                    metrics={
                        "validation_iptm": 0.82,
                        "validation_ipSAE": 0.61,
                        "validation_passed": True,
                    },
                )

                recovered = store.recover_stale_validation_tasks(
                    stale_before="9999-01-01T00:00:00.000Z",
                    error_message="validation interrupted after structure promotion",
                )
                self.assertEqual(recovered, 1)
                task = conn.execute(
                    "SELECT status, attempt_count FROM validation_tasks"
                ).fetchone()
                self.assertEqual(task["status"], "pending")
                self.assertEqual(task["attempt_count"], 1)
                structure_count = conn.execute(
                    "SELECT COUNT(*) FROM validation_structures"
                ).fetchone()[0]
                self.assertEqual(structure_count, 1)

                retry_claim = store.claim_next_pending_validation_tasks(
                    worker_id="validation-retry",
                    batch_size=1,
                    hostname="localhost",
                    pid=5678,
                    gpu_id="0",
                )[0]
                self.assertEqual(retry_claim.validation_id, first_claim.validation_id)
                store.complete_validation_task(
                    validation_id=retry_claim.validation_id,
                    attempt_id=retry_claim.attempt_id,
                    output_structure_path=promoted_path.as_posix(),
                    metrics={
                        "validation_iptm": 0.82,
                        "validation_ipSAE": 0.61,
                        "best_structure_id": "seed101_sample0",
                    },
                    runtime_seconds=12.3,
                )

                task = conn.execute(
                    """
                    SELECT status, attempt_count, output_structure_path, iptm, ipsae
                    FROM validation_tasks
                    """
                ).fetchone()
                self.assertEqual(task["status"], "completed")
                self.assertEqual(task["attempt_count"], 2)
                self.assertEqual(task["output_structure_path"], promoted_path.as_posix())
                self.assertAlmostEqual(task["iptm"], 0.82)
                self.assertAlmostEqual(task["ipsae"], 0.61)

                attempts = conn.execute(
                    """
                    SELECT status
                    FROM attempts
                    WHERE stage = 'validation'
                    ORDER BY attempt_id
                    """
                ).fetchall()
                self.assertEqual(
                    [row["status"] for row in attempts],
                    ["stale", "completed"],
                )
            finally:
                conn.close()

    def test_local_runner_rejects_missing_scoped_iptm_without_global_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = _run_fake_campaign(root)
            fake_protenix = _write_fake_protenix(root, missing_scoped_iptm=True)
            plan_validation_tasks(campaign_dir)

            result = run_local_protenix_validation(
                campaign_dir,
                config=ProtenixRunnerConfig(
                    protenix_command=(sys.executable, str(fake_protenix)),
                    seeds=(101,),
                    n_sample=1,
                    timeout_seconds=10,
                ),
            )

            self.assertEqual(result.completed_tasks, 1)
            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                task = conn.execute("SELECT * FROM validation_tasks").fetchone()
                self.assertEqual(task["status"], "completed")
                self.assertIsNone(task["iptm"])
                self.assertTrue(task["output_structure_path"].startswith(
                    "validation/protenix_v2/structures/rejected/"
                ))
                metrics = json.loads(task["metrics_json"])
                self.assertEqual(metrics["validation_global_iptm"], 0.99)
                self.assertNotIn("validation_iptm", metrics)
                self.assertIn("missing scoped binder-target", metrics["fail_reason"])

                structure = conn.execute("SELECT * FROM validation_structures").fetchone()
                self.assertEqual(structure["status"], "rejected")
                self.assertIsNone(structure["scoped_iptm"])
            finally:
                conn.close()

    def test_local_runner_skips_tasks_over_token_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = _run_fake_campaign(Path(tmpdir))
            plan_validation_tasks(campaign_dir)

            result = run_local_protenix_validation(
                campaign_dir,
                config=ProtenixRunnerConfig(
                    protenix_command=(sys.executable, "-c", "raise SystemExit(99)"),
                    token_limit=1,
                    max_tasks=1,
                ),
            )

            self.assertEqual(result.skipped_tasks, 1)
            self.assertEqual(result.failed_tasks, 0)
            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                task = conn.execute("SELECT status, error_message FROM validation_tasks").fetchone()
                self.assertEqual(task["status"], "skipped")
                self.assertIn("exceeding protenix-v2 limit", task["error_message"])
                attempt = conn.execute(
                    "SELECT status, exit_code FROM attempts WHERE stage = 'validation'"
                ).fetchone()
                self.assertEqual(attempt["status"], "completed")
                self.assertEqual(attempt["exit_code"], 0)
            finally:
                conn.close()

    def test_local_runner_attaches_provided_target_msa_to_input_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = _run_fake_campaign(root)
            fake_protenix = _write_fake_protenix(root, missing_scoped_iptm=False)
            msa_dir = root / "target_msa"
            msa_dir.mkdir()
            (msa_dir / "non_pairing.a3m").write_text(">query\nGGGG\n>hit\nGGGA\n")
            (msa_dir / "pairing.a3m").write_text(">query\nGGGG\n")
            plan_validation_tasks(campaign_dir)

            result = run_local_protenix_validation(
                campaign_dir,
                config=ProtenixRunnerConfig(
                    protenix_command=(sys.executable, str(fake_protenix)),
                    seeds=(101,),
                    n_sample=1,
                    timeout_seconds=10,
                    keep_debug=True,
                    target_msa_mode="provided",
                    target_msa_dir=msa_dir,
                ),
            )

            self.assertEqual(result.completed_tasks, 1)
            input_jsons = sorted(
                (campaign_dir / ".scratch" / "protenix_validation").glob(
                    "**/input/input.json"
                )
            )
            self.assertEqual(len(input_jsons), 1)
            payload = json.loads(input_jsons[0].read_text())
            binder_chain = payload[0]["sequences"][0]["proteinChain"]
            self.assertTrue(Path(binder_chain["pairedMsaPath"]).exists())
            self.assertTrue(Path(binder_chain["unpairedMsaPath"]).exists())
            self.assertEqual(
                Path(binder_chain["unpairedMsaPath"]).read_text(),
                ">query\nACDEFGHIK\n",
            )
            target_chain = payload[0]["sequences"][1]["proteinChain"]
            self.assertTrue(Path(target_chain["pairedMsaPath"]).exists())
            self.assertTrue(Path(target_chain["unpairedMsaPath"]).exists())
            self.assertEqual(
                Path(target_chain["unpairedMsaPath"]).read_text(),
                ">query\nGGGG\n>hit\nGGGA\n",
            )
            cache_files = list(
                (campaign_dir / "validation" / "protenix_v2" / "msa_cache" / "target").glob(
                    "**/metadata.json"
                )
            )
            self.assertEqual(len(cache_files), 1)
            binder_cache_files = list(
                (campaign_dir / "validation" / "protenix_v2" / "msa_cache" / "binder").glob(
                    "**/metadata.json"
                )
            )
            self.assertEqual(len(binder_cache_files), 1)

    def test_local_runner_rejects_hotspot_miss_and_selects_passing_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = _run_fake_campaign(root)
            _write_validation_chain_summary(campaign_dir, hotspot_indices=[1])
            fake_protenix = _write_fake_hotspot_protenix(root)
            plan_validation_tasks(campaign_dir)

            result = run_local_protenix_validation(
                campaign_dir,
                config=ProtenixRunnerConfig(
                    protenix_command=(sys.executable, str(fake_protenix)),
                    seeds=(101,),
                    n_sample=2,
                    timeout_seconds=10,
                    validation_hotspot_cutoff_angstrom=5.0,
                ),
            )

            self.assertEqual(result.completed_tasks, 1)
            self.assertEqual(result.failed_tasks, 0)
            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                task = conn.execute("SELECT * FROM validation_tasks").fetchone()
                self.assertEqual(task["status"], "completed")
                self.assertAlmostEqual(task["iptm"], 0.76)
                self.assertTrue(
                    task["output_structure_path"].endswith("sample01.cif")
                )
                task_metrics = json.loads(task["metrics_json"])
                self.assertTrue(task_metrics["validation_hotspot_pass"])
                self.assertAlmostEqual(
                    task_metrics["validation_hotspot_distance_angstrom"],
                    3.0,
                )

                structures = conn.execute(
                    """
                    SELECT status, sample_rank, scoped_iptm, metrics_json
                    FROM validation_structures
                    ORDER BY sample_rank
                    """
                ).fetchall()
                self.assertEqual(
                    [row["status"] for row in structures],
                    ["rejected", "passing"],
                )
                first_metrics = json.loads(structures[0]["metrics_json"])
                second_metrics = json.loads(structures[1]["metrics_json"])
                self.assertFalse(first_metrics["validation_hotspot_pass"])
                self.assertIn("exceeds cutoff", first_metrics["fail_reason"])
                self.assertTrue(second_metrics["validation_hotspot_pass"])
            finally:
                conn.close()

    def test_subprocess_runner_invokes_heartbeat_callback_while_running(self) -> None:
        calls: list[float] = []

        _run_protenix_command(
            [sys.executable, "-c", "import time; time.sleep(0.2)"],
            cwd=None,
            env=None,
            gpu_id=None,
            timeout_seconds=5,
            heartbeat_interval_seconds=0.05,
            on_heartbeat=lambda: calls.append(1.0),
        )

        self.assertGreaterEqual(len(calls), 2)

    def test_subprocess_runner_prepends_executable_dir_to_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = Path(tmpdir) / "protenix-venv" / "bin"
            bin_dir.mkdir(parents=True)
            helper = bin_dir / "protenix-helper"
            helper.write_text("#!/bin/sh\nprintf helper-ok\n")
            fake_python = bin_dir / "python"
            fake_python.write_text(
                "#!/bin/sh\n"
                "if command -v protenix-helper >/dev/null 2>&1; then\n"
                "  protenix-helper\n"
                "else\n"
                "  printf helper-missing >&2\n"
                "  exit 17\n"
                "fi\n"
            )
            os.chmod(helper, 0o755)
            os.chmod(fake_python, 0o755)

            result = _run_protenix_command(
                [str(fake_python), "-m", "runner.inference"],
                cwd=None,
                env={"PATH": "/usr/bin:/bin"},
                gpu_id=None,
                timeout_seconds=5,
                heartbeat_interval_seconds=0.05,
            )

        self.assertEqual(result.returncode, 0)
        self.assertIn("helper-ok", result.stdout_tail)

    def test_subprocess_runner_removes_esm_repo_from_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            esm_repo = Path(tmpdir) / "biohub-esm"
            keep_path = Path(tmpdir) / "keep"
            esm_repo.mkdir()
            keep_path.mkdir()

            result = _run_protenix_command(
                [
                    sys.executable,
                    "-c",
                    "import os; print(os.environ.get('PYTHONPATH', ''))",
                ],
                cwd=None,
                env={
                    "ESM_REPO": str(esm_repo),
                    "PYTHONPATH": os.pathsep.join(
                        [str(keep_path), str(esm_repo), str(esm_repo) + "/"]
                    ),
                },
                gpu_id=None,
                timeout_seconds=5,
                heartbeat_interval_seconds=0.05,
            )

        self.assertEqual(result.returncode, 0)
        self.assertIn(str(keep_path), result.stdout_tail)
        self.assertNotIn(str(esm_repo), result.stdout_tail)

    def test_multi_validation_workers_share_pending_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = _run_fake_campaign(root)
            fake_protenix = _write_fake_protenix(root, missing_scoped_iptm=False)
            plan_validation_tasks(campaign_dir)

            result = run_multi_validation(
                campaign_dir,
                gpu_ids=["0", "1"],
                worker_prefix="validation-test",
                max_tasks_per_worker=1,
                poll_interval_seconds=0.05,
                python_executable=sys.executable,
                worker_args=[
                    "--protenix-command",
                    f"{sys.executable} {fake_protenix}",
                    "--n-sample",
                    "1",
                    "--seeds",
                    "101",
                    "--timeout-seconds",
                    "10",
                ],
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.completed_tasks, 1)
            self.assertEqual(result.failed_workers, 0)
            self.assertEqual(len(result.worker_results), 2)
            for worker in result.worker_results:
                self.assertTrue(worker.log_path.exists())

            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                task = conn.execute("SELECT status FROM validation_tasks").fetchone()
                self.assertEqual(task["status"], "completed")
            finally:
                conn.close()

    def test_failed_worker_cleanup_removes_default_scratch_unless_debug_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            scratch_dir = (
                campaign_dir
                / ".scratch"
                / "protenix_validation"
                / "validation_gpu_gpu0_run_123"
                / "batch_20260619T000000_deadbeef"
            )
            scratch_dir.mkdir(parents=True)
            (scratch_dir / "input.json").write_text("{}")

            self.assertTrue(_should_cleanup_failed_worker_scratch(()))
            self.assertFalse(
                _should_cleanup_failed_worker_scratch(("--keep-validation-debug",))
            )

            _cleanup_failed_worker_scratch(
                campaign_dir,
                "validation-gpu-gpu0-run-123",
            )

            self.assertFalse((campaign_dir / ".scratch").exists())


def _run_fake_campaign(root: Path) -> Path:
    campaign_dir = root / "campaign"
    config_path = root / "config.yaml"
    config_path.write_text(
        f"""
target:
  name: sequence_target
  sequence: GGGG
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
        run_campaign(campaign_dir, worker_id="protenix-plan-test", gpu_id="0")
    return campaign_dir


def _write_validation_chain_summary(
    campaign_dir: Path,
    *,
    hotspot_indices: list[int],
) -> None:
    write_text_atomic(
        campaign_dir / "target" / "chain_summary.json",
        json.dumps(
            {
                "chains": [
                    {
                        "canonical_chain_id": "T",
                        "sequence": "GGGG",
                        "hotspot_indices": hotspot_indices,
                    }
                ]
            }
        ),
    )


def _fake_design_artifact(**kwargs) -> DesignCandidateArtifact:
    root = Path(kwargs["campaign_dir"])
    structure_path = structure_relpath(kwargs.get("artifact_stem", kwargs["candidate_id"]))
    write_text_atomic(root / structure_path, "HEADER    FAKE PROTENIX SOURCE\nEND\n")
    return DesignCandidateArtifact(
        candidate_id=kwargs["candidate_id"],
        designed_sequence="ACDEFGHIK",
        sequence_path=None,
        critic_name=kwargs["critic_name"],
        structure_path=structure_path.as_posix(),
        design_metrics={
            "target_name": "sequence_target",
            "binder_scaffold": "miniprotein",
            "binder_chain_id": "B",
        },
        critic_metrics={
            "iptm": 0.72,
            "iptm_scope": "binder_target",
            "ptm": 0.4,
            "plddt_complex": 80.0,
            "distogram_iptm_proxy": 0.68,
        },
    )


def _write_fake_protenix(root: Path, *, missing_scoped_iptm: bool) -> Path:
    script = root / ("fake_protenix_missing.py" if missing_scoped_iptm else "fake_protenix.py")
    script.write_text(
        f"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_json_path")
parser.add_argument("--dump_dir")
parser.add_argument("--seeds")
parser.add_argument("--model_name")
parser.add_argument("--sample_diffusion.N_sample", dest="n_sample", type=int)
parser.add_argument("--sample_diffusion.N_step", dest="n_step")
parser.add_argument("--model.N_cycle", dest="n_cycle")
parser.add_argument("--need_atom_confidence")
parser.add_argument("--use_msa")
parser.add_argument("--use_template")
parser.add_argument("--load_checkpoint_dir", default=None)
args = parser.parse_args()

samples = json.loads(Path(args.input_json_path).read_text())
seed = int(args.seeds.split(",")[0])
for sample in samples:
    name = sample["name"]
    pred_dir = Path(args.dump_dir) / name / f"seed_{{seed}}" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(args.n_sample):
        if rank == 0:
            iptm = 0.82
            ipsae = 0.61
            ranking = 0.52
        else:
            iptm = 0.21
            ipsae = 0.12
            ranking = 0.18
        summary = {{
            "iptm": 0.99,
            "ptm": 0.44,
            "ranking_score": ranking,
            "chain_pair_ipsae": [[0.0, ipsae], [ipsae, 0.0]],
        }}
        if {str(not missing_scoped_iptm)}:
            summary["chain_pair_iptm"] = [[0.0, iptm], [iptm, 0.0]]
        (pred_dir / f"{{name}}_summary_confidence_sample_{{rank}}.json").write_text(
            json.dumps(summary)
        )
        (pred_dir / f"{{name}}_full_data_sample_{{rank}}.json").write_text(
            json.dumps({{"token_pair_pae": [[1.0]], "atom_plddt": [0.9]}})
        )
        (pred_dir / f"{{name}}_sample_{{rank}}.cif").write_text(
            "data_fake\\n#\\n"
        )
""".lstrip()
    )
    return script


def _write_flaky_fake_protenix(root: Path) -> Path:
    success_script = _write_fake_protenix(root, missing_scoped_iptm=False)
    script = root / "fake_protenix_fails_once.py"
    script.write_text(
        f"""
import runpy
import sys
from pathlib import Path

marker = Path(__file__).with_suffix(".failed_once")
if not marker.exists():
    marker.write_text("failed")
    print("simulated transient Protenix startup failure", file=sys.stderr)
    raise SystemExit(17)

runpy.run_path({str(success_script)!r}, run_name="__main__")
""".lstrip()
    )
    return script


def _write_fake_ipsae_threshold_protenix(root: Path) -> Path:
    script = root / "fake_protenix_ipsae_threshold.py"
    script.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_json_path")
parser.add_argument("--dump_dir")
parser.add_argument("--seeds")
parser.add_argument("--model_name")
parser.add_argument("--sample_diffusion.N_sample", dest="n_sample", type=int)
parser.add_argument("--sample_diffusion.N_step", dest="n_step")
parser.add_argument("--model.N_cycle", dest="n_cycle")
parser.add_argument("--need_atom_confidence")
parser.add_argument("--use_msa")
parser.add_argument("--use_template")
parser.add_argument("--load_checkpoint_dir", default=None)
args = parser.parse_args()

samples = json.loads(Path(args.input_json_path).read_text())
seed = int(args.seeds.split(",")[0])
for sample in samples:
    name = sample["name"]
    pred_dir = Path(args.dump_dir) / name / f"seed_{seed}" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(args.n_sample):
        if rank == 0:
            iptm = 0.90
            ipsae = 0.50
            ranking = 0.80
        else:
            iptm = 0.80
            ipsae = 0.64
            ranking = 0.50
        summary = {
            "iptm": 0.99,
            "ptm": 0.44,
            "ranking_score": ranking,
            "chain_pair_iptm": [[0.0, iptm], [iptm, 0.0]],
            "chain_pair_ipsae": [[0.0, ipsae], [ipsae, 0.0]],
        }
        (pred_dir / f"{name}_summary_confidence_sample_{rank}.json").write_text(
            json.dumps(summary)
        )
        (pred_dir / f"{name}_full_data_sample_{rank}.json").write_text("{}")
        (pred_dir / f"{name}_sample_{rank}.cif").write_text("data_fake\\n#\\n")
""".lstrip()
    )
    return script


def _write_fake_protenix_without_summary_ipsae(root: Path) -> Path:
    script = root / "fake_protenix_without_summary_ipsae.py"
    script.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_json_path")
parser.add_argument("--dump_dir")
parser.add_argument("--seeds")
parser.add_argument("--model_name")
parser.add_argument("--sample_diffusion.N_sample", dest="n_sample", type=int)
parser.add_argument("--sample_diffusion.N_step", dest="n_step")
parser.add_argument("--model.N_cycle", dest="n_cycle")
parser.add_argument("--need_atom_confidence")
parser.add_argument("--use_msa")
parser.add_argument("--use_template")
parser.add_argument("--load_checkpoint_dir", default=None)
args = parser.parse_args()

samples = json.loads(Path(args.input_json_path).read_text())
seed = int(args.seeds.split(",")[0])
for sample in samples:
    name = sample["name"]
    pred_dir = Path(args.dump_dir) / name / f"seed_{seed}" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "iptm": 0.99,
        "ptm": 0.44,
        "ranking_score": 0.52,
        "chain_pair_iptm": [[0.0, 0.82], [0.82, 0.0]],
    }
    (pred_dir / f"{name}_summary_confidence_sample_0.json").write_text(
        json.dumps(summary)
    )
    (pred_dir / f"{name}_full_data_sample_0.json").write_text(
        json.dumps({"token_pair_pae": [[1.0, 2.0], [3.0, 4.0]], "atom_plddt": [0.9, 0.8]})
    )
    (pred_dir / f"{name}_sample_0.cif").write_text("data_fake\\n#\\n")
""".lstrip()
    )
    return script


def _write_fake_ipsae_script(root: Path) -> Path:
    script = root / "fake_ipsae.py"
    script.write_text(
        """
import json
import sys
from pathlib import Path

adapted_full_path = Path(sys.argv[1])
cif_path = Path(sys.argv[2])
pae_cutoff = int(sys.argv[3])
dist_cutoff = int(sys.argv[4])
payload = json.loads(adapted_full_path.read_text())
assert payload["pae"] == [[1.0, 2.0], [3.0, 4.0]]
assert payload["atom_plddts"] == [90.0, 80.0]
suffix = f"_{pae_cutoff:02d}_{dist_cutoff:02d}.txt"
if pae_cutoff >= 10 and dist_cutoff >= 10:
    suffix = f"_{pae_cutoff}_{dist_cutoff}.txt"
out = Path(str(cif_path).replace(".cif", suffix))
out.write_text(
    '''
Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model
A    B     15   15   asym  0.450000    0.510000    0.450000    0.820    0.100000      0.1200     0.3400     0.5600       3    20      9    1.04    7.28    1.04      2       7       2       7   model
B    A     15   15   asym  0.350000    0.410000    0.350000    0.820    0.100000      0.1200     0.3300     0.5600       2    20      8    1.04    7.28    1.04      2       7       2       7   model
A    B     15   15   max   0.730000    0.550000    0.440000    0.820    0.100000      0.4500     0.6700     0.8900       5    21      9    1.04    7.28    1.04      2       7       2       7   model
'''
)
""".lstrip()
    )
    return script


def _write_fake_hotspot_protenix(root: Path) -> Path:
    script = root / "fake_protenix_hotspot.py"
    script.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_json_path")
parser.add_argument("--dump_dir")
parser.add_argument("--seeds")
parser.add_argument("--model_name")
parser.add_argument("--sample_diffusion.N_sample", dest="n_sample", type=int)
parser.add_argument("--sample_diffusion.N_step", dest="n_step")
parser.add_argument("--model.N_cycle", dest="n_cycle")
parser.add_argument("--need_atom_confidence")
parser.add_argument("--use_msa")
parser.add_argument("--use_template")
parser.add_argument("--load_checkpoint_dir", default=None)
args = parser.parse_args()

samples = json.loads(Path(args.input_json_path).read_text())
seed = int(args.seeds.split(",")[0])
for sample in samples:
    name = sample["name"]
    pred_dir = Path(args.dump_dir) / name / f"seed_{seed}" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(args.n_sample):
        if rank == 0:
            iptm = 0.90
            ipsae = 0.70
            ranking = 0.80
            hotspot_distance = 12.0
        else:
            iptm = 0.76
            ipsae = 0.62
            ranking = 0.50
            hotspot_distance = 3.0
        summary = {
            "iptm": 0.99,
            "ptm": 0.44,
            "ranking_score": ranking,
            "chain_pair_iptm": [[0.0, iptm], [iptm, 0.0]],
            "chain_pair_ipsae": [[0.0, ipsae], [ipsae, 0.0]],
        }
        (pred_dir / f"{name}_summary_confidence_sample_{rank}.json").write_text(
            json.dumps(summary)
        )
        (pred_dir / f"{name}_full_data_sample_{rank}.json").write_text("{}")
        (pred_dir / f"{name}_sample_{rank}.cif").write_text(
            f'''data_fake
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
ATOM 2 C CA GLY B 1 40.000 0.000 0.000
ATOM 3 C CA GLY B 2 {hotspot_distance:.3f} 0.000 0.000
#
'''
        )
""".lstrip()
    )
    return script


if __name__ == "__main__":
    unittest.main()
