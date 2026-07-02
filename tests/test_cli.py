from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from esmfold2_pipeline import cli as cli_module
from esmfold2_pipeline.artifact_layout import structure_relpath
from esmfold2_pipeline.artifacts import write_text_atomic
from esmfold2_pipeline.db import initialize_database
from esmfold2_pipeline.esm_adapter import DesignCandidateArtifact
from esmfold2_pipeline.frameworks import all_scfv_framework_names, all_vhh_framework_names


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliTest(unittest.TestCase):
    def test_check_env_reports_failures(self) -> None:
        result = _run_cli(
            "check-env",
            "--esm-repo",
            "/definitely/missing/esm",
            "--no-tutorial",
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("ok: false", result.stdout)
        self.assertIn("ESM repo does not exist", result.stdout)

    def test_check_protenix_accepts_fake_runner_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "protenix"
            runner = root / "runner"
            runner.mkdir(parents=True)
            (runner / "__init__.py").write_text("")
            (runner / "inference.py").write_text("")

            result = _run_cli("check-protenix", "--protenix-root", root)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok: true", result.stdout)
        self.assertIn("runner.inference", result.stdout)

    def test_check_protenix_reports_missing_accelerate_when_protenix_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "protenix"
            runner = root / "runner"
            package = root / "protenix"
            runner.mkdir(parents=True)
            package.mkdir()
            (runner / "__init__.py").write_text("")
            (runner / "inference.py").write_text("")
            (package / "__init__.py").write_text("")

            result = _run_cli("check-protenix", "--protenix-root", root)

        self.assertEqual(result.returncode, 1)
        self.assertIn("ok: false", result.stdout)
        self.assertIn("Protenix environment cannot import accelerate", result.stdout)

    def test_check_protenix_uses_protenix_python_env_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "protenix"
            runner = root / "runner"
            runner.mkdir(parents=True)
            (runner / "__init__.py").write_text("")
            (runner / "inference.py").write_text("")
            missing_python = root / "venv" / "bin" / "python"

            with patch.dict(os.environ, {"PROTENIX_PYTHON": str(missing_python)}):
                result = _run_cli("check-protenix", "--protenix-root", root)

        self.assertEqual(result.returncode, 1)
        self.assertIn("ok: false", result.stdout)
        self.assertIn("Protenix executable does not exist", result.stdout)

    def test_check_protenix_requires_checkpoint_file_when_dir_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "protenix"
            runner = root / "runner"
            runner.mkdir(parents=True)
            (runner / "__init__.py").write_text("")
            (runner / "inference.py").write_text("")
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()

            result = _run_cli(
                "check-protenix",
                "--protenix-root",
                root,
                "--protenix-checkpoint-dir",
                checkpoint_dir,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("ok: false", result.stdout)
        self.assertIn("Protenix checkpoint file does not exist", result.stdout)

    def test_check_protenix_accepts_checkpoint_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "protenix"
            runner = root / "runner"
            runner.mkdir(parents=True)
            (runner / "__init__.py").write_text("")
            (runner / "inference.py").write_text("")
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "protenix-v2.pt").write_text("fake checkpoint")

            result = _run_cli(
                "check-protenix",
                "--protenix-root",
                root,
                "--protenix-checkpoint-dir",
                checkpoint_dir,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok: true", result.stdout)
        self.assertIn("checkpoint_file", result.stdout)

    def test_validation_defaults_use_protenix_env_without_overriding_command(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PROTENIX_PYTHON": "/opt/protenix/bin/python",
                "PROTENIX_CHECKPOINT_DIR": "/opt/protenix/checkpoints",
            },
        ):
            defaults = cli_module._validation_hash_builtin_defaults()

        self.assertEqual(defaults["protenix_python"], "/opt/protenix/bin/python")
        self.assertEqual(
            defaults["protenix_checkpoint_dir"],
            Path("/opt/protenix/checkpoints"),
        )

        args = argparse.Namespace(
            campaign_dir=Path("/tmp/campaign"),
            protenix_command="python fake_protenix.py",
            protenix_python=None,
        )
        with patch.dict(os.environ, {"PROTENIX_PYTHON": "/opt/protenix/bin/python"}):
            merged = cli_module._with_validation_yaml_defaults(args, mode="run")

        self.assertEqual(merged.protenix_command, "python fake_protenix.py")
        self.assertIsNone(merged.protenix_python)

    def test_validate_check_env_alias_reports_missing_protenix_root(self) -> None:
        result = _run_cli(
            "validate-check-env",
            "--protenix-root",
            "/definitely/missing/protenix",
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("ok: false", result.stdout)
        self.assertIn("Protenix root does not exist", result.stdout)

    def test_validate_check_env_reports_missing_ipsae_script(self) -> None:
        result = _run_cli(
            "validate-check-env",
            "--ipsae-script",
            "/definitely/missing/ipsae.py",
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("ok: false", result.stdout)
        self.assertIn("ipSAE script does not exist", result.stdout)

    def test_validate_check_env_reports_hidden_requested_gpu(self) -> None:
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "-1"}):
            result = _run_cli(
                "validate-check-env",
                "--gpu-id",
                "0",
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("ok: false", result.stdout)
        self.assertIn("hides requested GPU 0", result.stdout)

    def test_plan_gpu_smoke_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "gpu-campaign"
            result = _run_cli("plan-gpu-smoke", campaign_dir)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("planned gpu smoke campaign", result.stdout)
            self.assertIn("shard_000000", result.stdout)

    def test_plan_command_from_minimal_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: miniprotein
  length: 80-120
campaign:
  num_designs: 2
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {campaign_dir}
""".lstrip()
            )
            result = _run_cli("plan", config)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("planned campaign", result.stdout)
            self.assertIn("shards: 2", result.stdout)
            self.assertIn("next single GPU:", result.stdout)
            self.assertIn(f"esmfold2-pipeline run {campaign_dir}", result.stdout)
            self.assertIn("--gpus all", result.stdout)
            self.assertTrue((campaign_dir / "campaign.sqlite").exists())

    def test_plan_preserves_validation_yaml_in_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            msa_dir = root / "msa"
            msa_dir.mkdir()
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  steps: 1
validation:
  min_validation_ipsae: 0.62
  protenix:
    target_msa_dir: msa
output: {campaign_dir}
""".lstrip()
            )

            result = _run_cli("plan", config)

            self.assertEqual(result.returncode, 0, result.stderr)
            resolved = yaml.safe_load(
                (campaign_dir / "resolved_config.yaml").read_text()
            )
            self.assertEqual(resolved["validation"]["min_validation_ipsae"], 0.62)
            self.assertEqual(
                resolved["validation"]["protenix"]["target_msa_dir"],
                str(msa_dir.resolve()),
            )

    def test_check_config_command_reports_resolved_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config = root / "config.yaml"
            config.write_text(
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

            result = _run_cli("check", config)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ok: true", result.stdout)
            self.assertIn("target: ctla4", result.stdout)
            self.assertIn("binder_scaffold: miniprotein", result.stdout)
            self.assertIn("binder_length: 60-200", result.stdout)
            self.assertIn("critic: ESMFold2-Experimental-Fast", result.stdout)
            self.assertIn("designs: 3", result.stdout)
            self.assertNotIn("seeds:", result.stdout)
            self.assertIn("errors: none", result.stdout)
            self.assertFalse((campaign_dir / "campaign.sqlite").exists())

    def test_check_config_validates_validation_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  steps: 1
validation:
  require_hotspot_contact: sometimes
output: {campaign_dir}
""".lstrip()
            )

            result = _run_cli("check", config)

        self.assertEqual(result.returncode, 1)
        self.assertIn("validation_config: invalid", result.stdout)
        self.assertIn("require_hotspot_contact", result.stdout)

    def test_check_config_command_returns_nonzero_for_invalid_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "config.yaml"
            config.write_text(
                """
target:
  name: ctla4
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  critics:
    - ESMFold2-Experimental-Fast
""".lstrip()
            )

            result = _run_cli("check", config)
            self.assertEqual(result.returncode, 1)
            self.assertIn("ok: false", result.stdout)
            self.assertIn("output is required", result.stdout)

    def test_check_env_uses_local_runtime_for_builtin_scfv_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: scfv
  framework: trastuzumab
campaign:
  num_designs: 1
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            env_result = types.SimpleNamespace(ok=True, checks={}, errors=[])
            with patch.object(
                cli_module,
                "check_environment",
                return_value=env_result,
            ) as check_environment:
                args = cli_module.build_parser().parse_args(
                    ["check", str(config), "--env"]
                )
                result = args.func(args)

        self.assertEqual(result, 0)
        self.assertEqual(
            set(check_environment.call_args.kwargs),
            {
                "esm_repo",
                "require_cuda",
                "require_tutorial",
                "require_local_runtime",
            },
        )
        self.assertFalse(check_environment.call_args.kwargs["require_tutorial"])
        self.assertTrue(check_environment.call_args.kwargs["require_local_runtime"])

    def test_check_env_uses_local_runtime_for_sequence_scfv_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  name: ctla4
binder:
  scaffold: scfv
  framework:
    name: lab_fixed_scfv
    sequence: QVQLKQSGPGLVQPSQSLSITCTVSGFSLTNYGVHWVRQSPGKGLEWLGVIWSGGNTDYNTPFTSRLSINKDNSKSQVFFKMNSLQSNDTAIYYCARALTYYDYEFAYWGQGTLVTVSGGGGSGGGGSGGGGSGGGGSDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGISRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELK
    mutate: cdrs
    cdrs:
      hcdr1: 26-35
      hcdr2: 51-65
      hcdr3: 98-108
      lcdr1: 162-172
      lcdr2: 188-194
      lcdr3: 226-234
campaign:
  num_designs: 1
  steps: 1
output: {root / "campaign"}
""".lstrip()
            )

            env_result = types.SimpleNamespace(ok=True, checks={}, errors=[])
            with patch.object(
                cli_module,
                "check_environment",
                return_value=env_result,
            ) as check_environment:
                args = cli_module.build_parser().parse_args(
                    ["check", str(config), "--env"]
                )
                result = args.func(args)

        self.assertEqual(result, 0)
        self.assertEqual(
            set(check_environment.call_args.kwargs),
            {
                "esm_repo",
                "require_cuda",
                "require_tutorial",
                "require_local_runtime",
            },
        )
        self.assertFalse(check_environment.call_args.kwargs["require_tutorial"])
        self.assertTrue(check_environment.call_args.kwargs["require_local_runtime"])

    def test_check_env_local_runtime_uses_local_runtime_api_check(self) -> None:
        env_result = types.SimpleNamespace(ok=True, checks={}, errors=[])
        with patch.object(
            cli_module,
            "check_environment",
            return_value=env_result,
        ) as check_environment:
            args = cli_module.build_parser().parse_args(
                ["check-env", "--local-runtime"]
            )
            result = args.func(args)

        self.assertEqual(result, 0)
        self.assertFalse(check_environment.call_args.kwargs["require_tutorial"])
        self.assertTrue(check_environment.call_args.kwargs["require_local_runtime"])

    def test_check_env_default_uses_local_runtime_api_check(self) -> None:
        env_result = types.SimpleNamespace(ok=True, checks={}, errors=[])
        with patch.dict(os.environ, {}, clear=True), patch.object(
            cli_module,
            "check_environment",
            return_value=env_result,
        ) as check_environment:
            args = cli_module.build_parser().parse_args(["check-env"])
            result = args.func(args)

        self.assertEqual(result, 0)
        self.assertFalse(check_environment.call_args.kwargs["require_tutorial"])
        self.assertTrue(check_environment.call_args.kwargs["require_local_runtime"])

    def test_check_env_tutorial_backend_env_uses_tutorial_check(self) -> None:
        env_result = types.SimpleNamespace(ok=True, checks={}, errors=[])
        with patch.dict(
            os.environ,
            {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "tutorial"},
        ), patch.object(
            cli_module,
            "check_environment",
            return_value=env_result,
        ) as check_environment:
            args = cli_module.build_parser().parse_args(["check-env"])
            result = args.func(args)

        self.assertEqual(result, 0)
        self.assertTrue(check_environment.call_args.kwargs["require_tutorial"])
        self.assertFalse(check_environment.call_args.kwargs["require_local_runtime"])

    def test_check_models_help_is_available(self) -> None:
        result = _run_cli("check-models", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("load the model set", result.stdout)

    def test_run_multi_help_is_available(self) -> None:
        result = _run_cli("run-multi", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run one local worker process per GPU id", result.stdout)
        self.assertIn("--gpus", result.stdout)
        self.assertIn("ranges like 0-3", result.stdout)
        self.assertIn("all", result.stdout)
        self.assertIn("--validation-msa-workers", result.stdout)
        self.assertIn("--msa-max-requests-per-minute", result.stdout)

    def test_run_multi_starts_validation_msa_worker_pool_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            captured: dict[str, object] = {}

            class FakeMsaPool:
                def __init__(
                    self,
                    *,
                    campaign_dir: Path,
                    count: int,
                    max_requests_per_minute: float,
                    poll_interval_seconds: float,
                ) -> None:
                    captured["campaign_dir"] = campaign_dir
                    captured["count"] = count
                    captured["max_requests_per_minute"] = max_requests_per_minute
                    captured["poll_interval_seconds"] = poll_interval_seconds
                    self.count = count
                    self.completed_jobs = 0
                    self.failed_jobs = 0
                    self.skipped_jobs = 0
                    self.errors: list[str] = []

                def __enter__(self) -> "FakeMsaPool":
                    captured["entered"] = True
                    return self

                def __exit__(self, exc_type, exc_value, traceback) -> None:
                    captured["exited"] = True

            run_result = types.SimpleNamespace(
                ok=True,
                run_id="test-run",
                completed_shards=0,
                recovered_shards=0,
                failed_workers=0,
                worker_results=[],
            )
            args = argparse.Namespace(
                campaign_dir=campaign_dir,
                esm_repo=None,
                gpus=["0", "1"],
                worker_prefix="local-gpu",
                max_shards_per_worker=None,
                poll_interval=0.05,
                heartbeat_interval=30.0,
                stale_timeout=None,
                enable_hf_xet=False,
                disable_local_runtime_cache=True,
                validation_msa_workers=None,
                validation_msa_poll_interval=0.25,
                msa_max_requests_per_minute=7.0,
            )

            with patch.object(cli_module, "_LaunchValidationMsaWorkerPool", FakeMsaPool):
                with patch.object(
                    cli_module,
                    "run_multi_campaign",
                    return_value=run_result,
                ) as run_multi_campaign:
                    result = cli_module._run_multi_campaign(args)

            self.assertEqual(result, 0)
            self.assertEqual(captured["campaign_dir"], campaign_dir)
            self.assertEqual(captured["count"], 1)
            self.assertEqual(captured["max_requests_per_minute"], 7.0)
            self.assertEqual(captured["poll_interval_seconds"], 0.25)
            self.assertTrue(captured["entered"])
            self.assertTrue(captured["exited"])
            run_multi_campaign.assert_called_once()
            self.assertTrue(
                run_multi_campaign.call_args.kwargs["disable_local_runtime_cache"]
            )

    def test_validation_run_defaults_to_max_batch_size_ten(self) -> None:
        args = argparse.Namespace(campaign_dir=Path("/tmp/campaign"))
        merged = cli_module._with_validation_yaml_defaults(args, mode="run")

        self.assertEqual(merged.validation_batch_size, 10)

    def test_effective_validation_batch_size_spreads_small_campaigns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            _write_ready_validation_tasks(campaign_dir, count=12)
            args = argparse.Namespace(
                campaign_dir=campaign_dir,
                validation_batch_size=10,
            )

            merged = cli_module._with_effective_validation_batch_size(
                args,
                worker_count=2,
            )

            self.assertEqual(merged.validation_batch_size, 6)
            self.assertEqual(merged.validation_max_batch_size, 10)
            self.assertEqual(merged.validation_ready_task_count, 12)
            self.assertEqual(merged.validation_worker_count, 2)

    def test_effective_validation_batch_size_ignores_msa_blocked_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            _write_ready_validation_tasks(campaign_dir, count=3)
            _write_ready_validation_tasks(
                campaign_dir,
                count=7,
                start_index=3,
                blocked_by_msa=True,
            )
            args = argparse.Namespace(
                campaign_dir=campaign_dir,
                validation_batch_size=10,
            )

            merged = cli_module._with_effective_validation_batch_size(
                args,
                worker_count=2,
            )

            self.assertEqual(merged.validation_batch_size, 2)
            self.assertEqual(merged.validation_ready_task_count, 3)

    def test_launch_from_config_plans_then_runs_existing_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
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
    - fast
  steps: 1
output: {campaign_dir}
""".lstrip()
            )

            result = _run_cli("launch", config, "--max-shards", "0")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ok: true", result.stdout)
            self.assertIn("planned campaign", result.stdout)
            self.assertIn("no pending shards", result.stdout)
            self.assertTrue((campaign_dir / "campaign.sqlite").exists())

    def test_launch_runs_validation_msa_worker_for_validation_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config = root / "config.yaml"
            config.write_text(
                f"""
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
validation:
  model: protenix-v2
  msa:
    binder: single_sequence
output: {campaign_dir}
""".lstrip()
            )
            args = _launch_args(
                validation_msa_workers=1,
                validation_msa_poll_interval=0.01,
                msa_max_requests_per_minute=5.0,
                skip_validation=True,
            )

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_cli_design_artifact,
            ):
                result = cli_module._launch_config_path(config, args)

            self.assertEqual(result, 0)
            self.assertTrue(
                any(
                    (
                        campaign_dir
                        / "validation"
                        / "protenix_v2"
                        / "msa_cache"
                        / "binder"
                    ).glob("**/metadata.json")
                )
            )
            conn = sqlite3.connect(campaign_dir / "campaign.sqlite")
            conn.row_factory = sqlite3.Row
            try:
                job = conn.execute(
                    "SELECT status FROM validation_msa_jobs"
                ).fetchone()
                self.assertEqual(job["status"], "ready")
            finally:
                conn.close()

    def test_launch_from_config_writes_final_outputs_and_runs_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            fake_protenix = _write_cli_fake_protenix(root)
            config = root / "config.yaml"
            config.write_text(
                f"""
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
validation:
  model: protenix-v2
  top_k: all
  msa:
    binder: single_sequence
  protenix:
    command: "{sys.executable} {fake_protenix}"
    seeds: [101]
    n_sample: 1
    timeout_seconds: 10
output: {campaign_dir}
""".lstrip()
            )
            args = _launch_args(
                validation_msa_workers=0,
                poll_interval=0.05,
                analysis_top_k=1,
            )

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_cli_design_artifact,
            ):
                result = cli_module._launch_config_path(config, args)

            self.assertEqual(result, 0)
            self.assertTrue((campaign_dir / "esmfold2" / "metrics_all.csv").exists())
            self.assertTrue(
                (campaign_dir / "esmfold2" / "selected_designs.csv").exists()
            )
            self.assertTrue(
                (
                    campaign_dir
                    / "esmfold2"
                    / "selected_structures"
                    / "selected_manifest.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    campaign_dir
                    / "validation"
                    / "protenix_v2"
                    / "validation_summary.json"
                ).exists()
            )
            self.assertTrue(
                (
                    campaign_dir
                    / "ranked_results"
                    / "combined_ranking.csv"
                ).exists()
            )

    def test_launch_campaign_dir_resumes_existing_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config = root / "config.yaml"
            config.write_text(
                f"""
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
            )
            args = _launch_args(skip_validation=True)
            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_cli_design_artifact,
            ):
                self.assertEqual(cli_module._launch_config_path(config, args), 0)

            result = _run_cli("launch", campaign_dir, "--skip-validation")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("resuming campaign", result.stdout)
            self.assertIn("no pending shards", result.stdout)
            self.assertTrue(
                (
                    campaign_dir
                    / "esmfold2"
                    / "selected_structures"
                    / "selected_manifest.csv"
                ).exists()
            )

            rejected = _run_cli(
                "launch",
                campaign_dir,
                "--out",
                root / "other-campaign",
                "--skip-validation",
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("config-generation flags", rejected.stdout)
            self.assertIn("--out", rejected.stdout)

    def test_launch_same_config_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config = root / "config.yaml"
            config.write_text(
                f"""
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
            )
            args = _launch_args(skip_validation=True)
            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_cli_design_artifact,
            ):
                self.assertEqual(cli_module._launch_config_path(config, args), 0)
                self.assertEqual(cli_module._launch_config_path(config, args), 0)

            conn = sqlite3.connect(campaign_dir / "campaign.sqlite")
            conn.row_factory = sqlite3.Row
            try:
                shard_count = conn.execute("SELECT COUNT(*) FROM shards").fetchone()[0]
                candidate_count = conn.execute(
                    "SELECT COUNT(*) FROM candidates"
                ).fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(shard_count, 1)
            self.assertEqual(candidate_count, 1)

    def test_launch_config_rejects_existing_campaign_with_different_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config = root / "config.yaml"
            config.write_text(
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
            self.assertEqual(_run_cli("plan", config).returncode, 0)
            config.write_text(
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

            result = _run_cli("launch", config, "--max-shards", "0")

            self.assertEqual(result.returncode, 2)
            self.assertIn(
                "existing campaign was planned from a different config",
                result.stdout,
            )

    def test_launch_rejects_invalid_validation_config_before_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  name: ctla4
  sequence: GGGG
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  steps: 1
validation:
  require_hotspot_contact: maybe
output: {campaign_dir}
""".lstrip()
            )
            args = _launch_args(skip_validation=True)

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_cli_design_artifact,
            ) as design:
                result = cli_module._launch_config_path(config, args)

            self.assertEqual(result, 2)
            design.assert_not_called()
            self.assertFalse((campaign_dir / "campaign.sqlite").exists())

    def test_launch_skip_export_still_aggregates_and_selects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            config = root / "config.yaml"
            config.write_text(
                f"""
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
            )
            args = _launch_args(skip_validation=True, skip_export=True)

            with patch(
                "esmfold2_pipeline.execution.local.run_binder_design_artifact",
                side_effect=_fake_cli_design_artifact,
            ):
                result = cli_module._launch_config_path(config, args)

            self.assertEqual(result, 0)
            self.assertTrue((campaign_dir / "esmfold2" / "metrics_all.csv").exists())
            self.assertTrue(
                (campaign_dir / "esmfold2" / "selected_designs.csv").exists()
            )
            self.assertFalse(
                (
                    campaign_dir
                    / "esmfold2"
                    / "selected_structures"
                    / "selected_manifest.csv"
                ).exists()
            )

    def test_launch_without_yaml_generates_miniprotein_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"

            result = _run_cli(
                "launch",
                "--target-name",
                "custom_ctla4",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--num-designs",
                "1",
                "--length",
                "80-140",
                "--model",
                "fast",
                "--steps",
                "1",
                "--out",
                campaign_dir,
                "--max-shards",
                "0",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ok: true", result.stdout)
            self.assertIn("planned campaign", result.stdout)
            self.assertTrue((campaign_dir / "config.yaml").exists())
            resolved = yaml.safe_load((campaign_dir / "resolved_config.yaml").read_text())
            self.assertEqual(resolved["target"]["name"], "custom_ctla4")
            self.assertEqual(resolved["target"]["sequence"], "ACDEFGHIKLMNPQRSTVWY")
            self.assertEqual(resolved["binder"]["scaffold"], "miniprotein")
            self.assertEqual(resolved["binder"]["length"], {"min": 80, "max": 140})
            self.assertEqual(
                resolved["campaign"]["inversion_model"],
                "ESMFold2-Experimental-Fast",
            )
            self.assertEqual(
                resolved["campaign"]["critics"],
                ["ESMFold2-Experimental-Fast"],
            )

    def test_launch_without_yaml_generates_structure_target_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_cli_test_pdb(target_path)
            campaign_dir = root / "campaign"

            result = _run_cli(
                "launch",
                "--target-name",
                "target_from_structure",
                "--target-structure",
                target_path,
                "--chains",
                "A",
                "--hotspot",
                "A:2",
                "--num-designs",
                "1",
                "--out",
                campaign_dir,
                "--max-shards",
                "0",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ok: true", result.stdout)
            resolved = yaml.safe_load((campaign_dir / "resolved_config.yaml").read_text())
            self.assertEqual(resolved["target"]["name"], "target_from_structure")
            self.assertEqual(resolved["target"]["structure"], str(target_path.resolve()))
            self.assertEqual(resolved["target"]["chains"], ["A"])
            self.assertEqual(resolved["target"]["hotspots"], {"A": ["2"]})
            self.assertEqual(resolved["target"]["conditioning"]["mode"], "distogram")
            self.assertFalse(resolved["target"]["conditioning"]["assembly"])
            self.assertTrue(resolved["loss"]["target_geometry_drift"]["enabled"])
            self.assertTrue(
                (
                    campaign_dir
                    / "target"
                    / "conditioning"
                    / "chain_A_distogram.npy"
                ).exists()
            )
            self.assertTrue((campaign_dir / "target" / "chain_summary.json").exists())

    def test_launch_rejects_missing_explicit_target_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_cli(
                "launch",
                "--target-name",
                "missing_source",
                "--num-designs",
                "1",
                "--out",
                Path(tmpdir) / "campaign",
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("--target-sequence or --target-structure is required", result.stdout)

    def test_launch_without_yaml_generates_scfv_all_frameworks_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"

            result = _run_cli(
                "launch",
                "--target-name",
                "custom_scfv_target",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--scaffold",
                "scfv",
                "--frameworks",
                "all",
                "--num-designs",
                "3",
                "--out",
                campaign_dir,
                "--max-shards",
                "0",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            resolved = yaml.safe_load((campaign_dir / "resolved_config.yaml").read_text())
            self.assertEqual(resolved["binder"]["scaffold"], "scfv")
            expected_frameworks = list(all_scfv_framework_names())
            self.assertEqual(resolved["binder"]["frameworks"], expected_frameworks)
            self.assertEqual(
                resolved["campaign"]["framework_schedule"],
                expected_frameworks[:3],
            )

    def test_launch_without_yaml_accepts_scfv_framework_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"

            result = _run_cli(
                "launch",
                "--target-name",
                "custom_scfv_target",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--scaffold",
                "scfv",
                "--frameworks",
                "trastuzumab,atezolizumab",
                "--num-designs",
                "2",
                "--out",
                campaign_dir,
                "--max-shards",
                "0",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            generated = yaml.safe_load((campaign_dir / "config.yaml").read_text())
            resolved = yaml.safe_load((campaign_dir / "resolved_config.yaml").read_text())
            self.assertEqual(
                generated["binder"]["frameworks"],
                [
                    "trastuzumab_framework_vhvl",
                    "atezolizumab_framework_vhvl",
                ],
            )
            self.assertEqual(
                resolved["binder"]["frameworks"],
                [
                    "trastuzumab_framework_vhvl",
                    "atezolizumab_framework_vhvl",
                ],
            )

    def test_launch_without_yaml_generates_vhh_all_frameworks_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"

            result = _run_cli(
                "launch",
                "--target-name",
                "custom_vhh_target",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--scaffold",
                "vhh",
                "--frameworks",
                "all",
                "--num-designs",
                "3",
                "--out",
                campaign_dir,
                "--max-shards",
                "0",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            resolved = yaml.safe_load((campaign_dir / "resolved_config.yaml").read_text())
            self.assertEqual(resolved["binder"]["scaffold"], "vhh")
            expected_frameworks = list(all_vhh_framework_names())
            self.assertEqual(resolved["binder"]["frameworks"], expected_frameworks)
            self.assertEqual(
                resolved["campaign"]["framework_schedule"],
                expected_frameworks[:3],
            )

    def test_launch_without_yaml_accepts_vhh_framework_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"

            result = _run_cli(
                "launch",
                "--target-name",
                "custom_vhh_target",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--scaffold",
                "vhh",
                "--frameworks",
                "caplacizumab,vobarilizumab_il6r",
                "--num-designs",
                "2",
                "--out",
                campaign_dir,
                "--max-shards",
                "0",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            generated = yaml.safe_load((campaign_dir / "config.yaml").read_text())
            self.assertEqual(
                generated["binder"]["frameworks"],
                [
                    "caplacizumab_framework_vhh",
                    "vobarilizumab_il6r_framework_vhh",
                ],
            )

    def test_launch_without_yaml_generates_mosaic_cdr_mode_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"

            result = _run_cli(
                "launch",
                "--target-name",
                "custom_vhh_target",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--scaffold",
                "vhh",
                "--frameworks",
                "caplacizumab",
                "--num-designs",
                "1",
                "--binder-target-contact-mode",
                "mosaic_cdr",
                "--mosaic-cdr-contact-weight",
                "0.7",
                "--mosaic-cdr-contact-cutoff-angstrom",
                "18.0",
                "--mosaic-cdr-num-target-contacts",
                "2",
                "--mosaic-framework-contact-penalty-weight",
                "0.25",
                "--mosaic-framework-contact-penalty-scope",
                "target_all",
                "--out",
                campaign_dir,
                "--max-shards",
                "0",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            resolved = yaml.safe_load((campaign_dir / "resolved_config.yaml").read_text())
            self.assertEqual(
                resolved["loss"]["binder_target_contact_mode"],
                "mosaic_cdr",
            )
            self.assertEqual(resolved["loss"]["mosaic_cdr_contact_weight"], 0.7)
            self.assertEqual(
                resolved["loss"]["mosaic_cdr_contact_cutoff_angstrom"],
                18.0,
            )
            self.assertEqual(resolved["loss"]["mosaic_cdr_num_target_contacts"], 2)
            self.assertEqual(
                resolved["loss"]["mosaic_framework_contact_penalty_weight"],
                0.25,
            )
            self.assertEqual(
                resolved["loss"]["mosaic_framework_contact_penalty_scope"],
                "target_all",
            )

    def test_launch_rejects_mosaic_cdr_mode_for_miniprotein(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_cli(
                "launch",
                "--target-name",
                "custom_miniprotein_target",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--num-designs",
                "1",
                "--binder-target-contact-mode",
                "mosaic_cdr",
                "--out",
                Path(tmpdir) / "campaign",
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("mosaic_cdr contact mode requires", result.stdout)

    def test_launch_rejects_mosaic_tuning_flags_without_mosaic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_cli(
                "launch",
                "--target-name",
                "custom_vhh_target",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--scaffold",
                "vhh",
                "--frameworks",
                "caplacizumab",
                "--num-designs",
                "1",
                "--mosaic-cdr-contact-weight",
                "0.7",
                "--out",
                Path(tmpdir) / "campaign",
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn(
                "--mosaic-cdr-contact-weight requires "
                "--binder-target-contact-mode mosaic_cdr",
                result.stdout,
            )

    def test_launch_rejects_mosaic_penalty_scope_without_mosaic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_cli(
                "launch",
                "--target-name",
                "custom_vhh_target",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--scaffold",
                "vhh",
                "--frameworks",
                "caplacizumab",
                "--num-designs",
                "1",
                "--mosaic-framework-contact-penalty-scope",
                "target_all",
                "--out",
                Path(tmpdir) / "campaign",
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn(
                "--mosaic-framework-contact-penalty-scope requires "
                "--binder-target-contact-mode mosaic_cdr",
                result.stdout,
            )

    def test_launch_requires_frameworks_for_scfv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_cli(
                "launch",
                "--target-name",
                "custom_scfv_target",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--scaffold",
                "scfv",
                "--num-designs",
                "1",
                "--out",
                Path(tmpdir) / "campaign",
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("--frameworks is required when --scaffold scfv", result.stdout)

    def test_launch_requires_frameworks_for_vhh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_cli(
                "launch",
                "--target-name",
                "custom_vhh_target",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--scaffold",
                "vhh",
                "--num-designs",
                "1",
                "--out",
                Path(tmpdir) / "campaign",
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("--frameworks is required when --scaffold vhh", result.stdout)

    def test_launch_rejects_length_for_vhh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_cli(
                "launch",
                "--target-name",
                "custom_vhh_target",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--scaffold",
                "vhh",
                "--frameworks",
                "caplacizumab",
                "--length",
                "80",
                "--num-designs",
                "1",
                "--out",
                Path(tmpdir) / "campaign",
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("--length is only valid when --scaffold miniprotein", result.stdout)

    def test_launch_rejects_frameworks_for_miniprotein(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_cli(
                "launch",
                "--target-name",
                "custom_miniprotein_target",
                "--target-sequence",
                "ACDEFGHIKLMNPQRSTVWY",
                "--frameworks",
                "all",
                "--num-designs",
                "1",
                "--out",
                Path(tmpdir) / "campaign",
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn(
                "--frameworks is only valid when --scaffold scfv or vhh",
                result.stdout,
            )

    def test_plan_run_status_mock_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"

            planned = _run_cli("plan-mock", campaign_dir)
            self.assertEqual(planned.returncode, 0, planned.stderr)
            self.assertIn("planned mock campaign", planned.stdout)
            self.assertIn("shard_000000", planned.stdout)

            completed = _run_cli(
                "run-mock",
                campaign_dir,
                "--worker-id",
                "cli-worker",
                "--gpu-id",
                "0",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("completed shard: shard_000000", completed.stdout)
            self.assertIn("candidate: cand_000000_0000", completed.stdout)
            self.assertIn("sequence_path:", completed.stdout)
            self.assertIn("structure_path:", completed.stdout)

            status = _run_cli("status", campaign_dir)
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertIn("shards: completed=1", status.stdout)
            self.assertIn("candidates: completed=1", status.stdout)
            self.assertIn("critics: completed=1", status.stdout)
            self.assertIn("attempts: completed=1", status.stdout)
            self.assertIn("issues: none", status.stdout)

            second_run = _run_cli("run-mock", campaign_dir)
            self.assertEqual(second_run.returncode, 0, second_run.stderr)
            self.assertIn("no pending mock shard", second_run.stdout)

    def test_aggregate_select_export_mock_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"

            self.assertEqual(_run_cli("plan-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)

            aggregate = _run_cli("aggregate", campaign_dir)
            self.assertEqual(aggregate.returncode, 0, aggregate.stderr)
            self.assertIn("metric_rows: 1", aggregate.stdout)
            self.assertTrue((campaign_dir / "esmfold2" / "metrics_all.csv").exists())

            select = _run_cli("select", campaign_dir, "--max-designs", "1")
            self.assertEqual(select.returncode, 0, select.stderr)
            self.assertIn("candidate_pool: 1", select.stdout)
            self.assertIn("selected: 1", select.stdout)
            self.assertTrue((campaign_dir / "esmfold2" / "selected_designs.csv").exists())

            export = _run_cli("export", campaign_dir)
            self.assertEqual(export.returncode, 0, export.stderr)
            self.assertIn("selected: 1", export.stdout)
            self.assertIn("copied_files: 1", export.stdout)
            self.assertTrue(
                (campaign_dir / "esmfold2" / "selected_structures" / "selected_manifest.csv").exists()
            )

    def test_status_returns_nonzero_when_reconciliation_finds_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            self.assertEqual(_run_cli("plan-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)

            structure_path = (
                campaign_dir
                / "esmfold2" / "structures"
                / "s000_seed000_c000.pdb"
            )
            structure_path.unlink()

            status = _run_cli("status", campaign_dir)
            self.assertEqual(status.returncode, 1)
            self.assertIn("issues: 1", status.stdout)
            self.assertIn("missing_structure_artifact", status.stdout)

    def test_validate_plan_command_plans_from_completed_mock_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            self.assertEqual(_run_cli("plan-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)

            result = _run_cli(
                "validate-plan",
                campaign_dir,
                "--validate-model",
                "protenix-v2",
                "--validate-top-k",
                "all",
                "--min-validation-ipsae",
                "0.6",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("validation_model: protenix-v2", result.stdout)
            self.assertIn("candidate_pool: 1", result.stdout)
            self.assertIn("selected: 1", result.stdout)
            self.assertIn("created: 1", result.stdout)

            status = _run_cli("status", campaign_dir)
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertIn("validation_tasks: pending=1", status.stdout)

    def test_auto_template_preflight_only_requires_support_for_discovered_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            self.assertEqual(_run_cli("plan-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("validate-plan", campaign_dir).returncode, 0)
            config = cli_module.ProtenixRunnerConfig(use_template="auto")

            self.assertFalse(
                cli_module._validation_run_requires_template_support(
                    campaign_dir,
                    config=config,
                )
            )

            target_dir = campaign_dir / "target"
            target_dir.mkdir()
            (target_dir / "normalized_target.cif").write_text("data_target\n")
            self.assertTrue(
                cli_module._validation_run_requires_template_support(
                    campaign_dir,
                    config=config,
                )
            )

    def test_validate_plan_rejects_scfv_for_protenix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            self.assertEqual(_run_cli("plan-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)
            conn = sqlite3.connect(campaign_dir / "campaign.sqlite")
            try:
                conn.execute(
                    """
                    UPDATE candidates
                    SET design_metrics_json = ?
                    """,
                    (
                        json.dumps(
                            {
                                "target_name": "mock_target",
                                "binder_scaffold": "scfv",
                                "binder_type": "scfv",
                                "binder_chain_id": "B",
                            }
                        ),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            result = _run_cli("validate-plan", campaign_dir)

            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(
                "requires a bundled scFv framework structural template",
                result.stdout,
            )
            conn = sqlite3.connect(campaign_dir / "campaign.sqlite")
            try:
                count = conn.execute("SELECT COUNT(*) FROM validation_tasks").fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(count, 0)

    def test_validate_run_mock_command_promotes_validation_structures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            self.assertEqual(_run_cli("plan-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)
            self.assertEqual(
                _run_cli(
                    "validate-plan",
                    campaign_dir,
                    "--validate-model",
                    "mock-protenix-v2",
                ).returncode,
                0,
            )

            result = _run_cli(
                "validate-run-mock",
                campaign_dir,
                "--worker-id",
                "cli-mock-validator",
                "--gpu-id",
                "0",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("completed validation tasks: 1", result.stdout)
            self.assertIn("recorded structures: 2", result.stdout)

            status = _run_cli("status", campaign_dir)
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertIn("validation_tasks: completed=1", status.stdout)
            self.assertIn(
                "validation_structures: passing=1, rejected=1",
                status.stdout,
            )
            self.assertTrue(
                any(
                    (
                        campaign_dir
                        / "validation"
                        / "mock_protenix_v2"
                        / "structures"
                        / "passing"
                    ).glob("*.cif")
                )
            )
            self.assertTrue(
                any(
                    (
                        campaign_dir
                        / "validation"
                        / "mock_protenix_v2"
                        / "structures"
                        / "rejected"
                    ).glob("*.cif")
                )
            )

    def test_validate_report_command_writes_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            self.assertEqual(_run_cli("plan-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)
            self.assertEqual(
                _run_cli(
                    "validate-plan",
                    campaign_dir,
                    "--validate-model",
                    "mock-protenix-v2",
                ).returncode,
                0,
            )
            self.assertEqual(_run_cli("validate-run-mock", campaign_dir).returncode, 0)

            result = _run_cli("validate-report", campaign_dir)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("validation_results_csv:", result.stdout)
            self.assertIn("structure_samples_csv:", result.stdout)
            self.assertIn("validation_summary_json:", result.stdout)
            self.assertIn("validation_tasks: 1", result.stdout)
            self.assertIn("validation_structures: 2", result.stdout)
            self.assertTrue(
                (
                    campaign_dir
                    / "validation"
                    / "mock_protenix_v2"
                    / "validation_results.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    campaign_dir
                    / "validation"
                    / "mock_protenix_v2"
                    / "structure_samples.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    campaign_dir
                    / "validation"
                    / "mock_protenix_v2"
                    / "validation_summary.json"
                ).exists()
            )

    def test_validate_run_command_promotes_fake_protenix_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            fake_protenix = _write_cli_fake_protenix(root)
            self.assertEqual(_run_cli("plan-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)
            self.assertEqual(
                _run_cli("validate-plan", campaign_dir).returncode,
                0,
            )

            result = _run_cli(
                "validate-run",
                campaign_dir,
                "--worker-id",
                "cli-protenix-validator",
                "--gpu-id",
                "0",
                "--protenix-command",
                f"{sys.executable} {fake_protenix}",
                "--n-sample",
                "1",
                "--seeds",
                "101",
                "--min-validation-ipsae",
                "0.6",
                "--timeout-seconds",
                "10",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("completed validation tasks: 1", result.stdout)
            self.assertIn("failed validation tasks: 0", result.stdout)
            self.assertIn("recorded structures: 1", result.stdout)

            status = _run_cli("status", campaign_dir)
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertIn("validation_tasks: completed=1", status.stdout)
            self.assertIn("validation_structures: passing=1", status.stdout)
            self.assertTrue(
                any(
                    (
                        campaign_dir
                        / "validation"
                        / "protenix_v2"
                        / "structures"
                        / "passing"
                    ).glob("*.cif")
                )
            )

    def test_validate_run_preflight_fails_before_claiming_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            self.assertEqual(_run_cli("plan-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("validate-plan", campaign_dir).returncode, 0)

            result = _run_cli(
                "validate-run",
                campaign_dir,
                "--protenix-command",
                str(root / "missing_protenix"),
                "--max-tasks",
                "1",
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("Protenix validation preflight: failed", result.stdout)
            self.assertIn("Protenix executable does not exist", result.stdout)

            conn = sqlite3.connect(campaign_dir / "campaign.sqlite")
            conn.row_factory = sqlite3.Row
            try:
                task = conn.execute(
                    "SELECT status, attempt_count FROM validation_tasks"
                ).fetchone()
                self.assertEqual(task["status"], "pending")
                self.assertEqual(task["attempt_count"], 0)
                validation_attempts = conn.execute(
                    "SELECT COUNT(*) FROM attempts WHERE stage = 'validation'"
                ).fetchone()[0]
                self.assertEqual(validation_attempts, 0)
            finally:
                conn.close()

    def test_validate_msa_retry_resets_failed_job_by_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            self.assertEqual(_run_cli("plan-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)

            conn = sqlite3.connect(campaign_dir / "campaign.sqlite")
            conn.row_factory = sqlite3.Row
            try:
                candidate_id = conn.execute(
                    "SELECT candidate_id FROM candidates LIMIT 1"
                ).fetchone()["candidate_id"]
                conn.execute(
                    """
                    INSERT INTO validation_msa_jobs (
                        msa_job_id,
                        scope,
                        cache_key,
                        msa_context_hash,
                        status,
                        attempt_count,
                        max_attempts,
                        representative_sequence,
                        member_sequences_json,
                        error_message,
                        completed_at
                    )
                    VALUES (
                        'msa_failed_cli',
                        'miniprotein_single_sequence',
                        'miniprotein:test',
                        'ctx',
                        'failed',
                        1,
                        1,
                        'ACDEFGHIK',
                        '["ACDEFGHIK"]',
                        'server failed',
                        '2026-01-01T00:00:00.000Z'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO validation_msa_job_candidates (
                        candidate_id,
                        msa_job_id,
                        reason
                    )
                    VALUES (?, 'msa_failed_cli', 'test')
                    """,
                    (candidate_id,),
                )
                conn.commit()
            finally:
                conn.close()

            result = _run_cli(
                "validate-msa-retry",
                campaign_dir,
                "--candidate-id",
                candidate_id,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("retried_msa_jobs: 1", result.stdout)
            self.assertIn("attempt_counts_reset: true", result.stdout)

            conn = sqlite3.connect(campaign_dir / "campaign.sqlite")
            conn.row_factory = sqlite3.Row
            try:
                job = conn.execute(
                    """
                    SELECT status, attempt_count, error_message, completed_at
                    FROM validation_msa_jobs
                    WHERE msa_job_id = 'msa_failed_cli'
                    """
                ).fetchone()
                self.assertEqual(job["status"], "pending")
                self.assertEqual(job["attempt_count"], 0)
                self.assertIsNone(job["error_message"])
                self.assertIsNone(job["completed_at"])
            finally:
                conn.close()

    def test_validation_commands_use_yaml_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            fake_protenix = _write_cli_fake_protenix(root)
            self.assertEqual(_run_cli("plan-mock", campaign_dir).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)
            (campaign_dir / "config.yaml").write_text(
                f"""
validation:
  model: protenix-v2
  top_k: all
  min_validation_iptm: 0.75
  min_validation_ipsae: 0.60
  msa:
    binder: single_sequence
  protenix:
    command: "{sys.executable} {fake_protenix}"
    seeds: [101]
    n_sample: 1
    timeout_seconds: 10
""".lstrip()
            )

            plan = _run_cli("validate-plan", campaign_dir)
            self.assertEqual(plan.returncode, 0, plan.stderr)
            self.assertIn("validation_model: protenix-v2", plan.stdout)
            self.assertIn("selected: 1", plan.stdout)

            run = _run_cli("validate-run", campaign_dir)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("completed validation tasks: 1", run.stdout)
            self.assertIn("recorded structures: 1", run.stdout)
            self.assertTrue(
                any(
                    (
                        campaign_dir
                        / "validation"
                        / "protenix_v2"
                        / "msa_cache"
                        / "binder"
                    ).glob("**/metadata.json")
                )
            )

            report = _run_cli("validate-report", campaign_dir)
            self.assertEqual(report.returncode, 0, report.stderr)
            manifest = (
                campaign_dir
                / "validation"
                / "protenix_v2"
                / "structure_samples.csv"
            ).read_text()
            self.assertIn("min_validator_ipsae", manifest)
            self.assertIn("validator_ipsae_pass", manifest)
            self.assertIn("0.6", manifest)

    def test_validate_wrapper_runs_msa_plan_validation_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            fake_protenix = _write_cli_fake_protenix(root)
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  name: mock_target
  sequence: GGGG
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  steps: 1
validation:
  model: protenix-v2
  top_k: all
  min_validation_iptm: 0.75
  min_validation_ipsae: 0.60
  msa:
    binder: single_sequence
  protenix:
    command: "{sys.executable} {fake_protenix}"
    seeds: [101]
    n_sample: 1
    timeout_seconds: 10
output: {campaign_dir}
""".lstrip()
            )
            self.assertEqual(_run_cli("plan", config).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)

            result = _run_cli("validate", campaign_dir)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("existing_msa_jobs: 1", result.stdout)
            self.assertIn("completed_msa_jobs: 1", result.stdout)
            self.assertIn("validation_model: protenix-v2", result.stdout)
            self.assertIn("completed validation tasks: 1", result.stdout)
            self.assertIn("validation_summary_json:", result.stdout)
            self.assertTrue(
                (
                    campaign_dir
                    / "validation"
                    / "protenix_v2"
                    / "validation_summary.json"
                ).exists()
            )
            self.assertTrue(
                any(
                    (
                        campaign_dir
                        / "validation"
                        / "protenix_v2"
                        / "msa_cache"
                        / "binder"
                    ).glob("**/metadata.json")
                )
            )

    def test_validate_wrapper_uses_cli_msa_overrides_for_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "campaign"
            fake_protenix = _write_cli_fake_protenix(root)
            config = root / "config.yaml"
            config.write_text(
                f"""
target:
  name: mock_target
  sequence: GGGG
binder:
  scaffold: miniprotein
campaign:
  num_designs: 1
  steps: 1
output: {campaign_dir}
""".lstrip()
            )
            self.assertEqual(_run_cli("plan", config).returncode, 0)
            self.assertEqual(_run_cli("run-mock", campaign_dir).returncode, 0)

            result = _run_cli(
                "validate",
                campaign_dir,
                "--protenix-command",
                f"{sys.executable} {fake_protenix}",
                "--use-msa",
                "--target-msa-mode",
                "none",
                "--binder-msa-mode",
                "single_sequence",
                "--n-sample",
                "1",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("queued_msa_jobs: 1", result.stdout)
            self.assertIn("completed_msa_jobs: 1", result.stdout)
            self.assertIn("validation_model: protenix-v2", result.stdout)
            self.assertIn("completed validation tasks: 1", result.stdout)


def _write_cli_test_pdb(path: Path) -> None:
    lines = []
    serial = 1
    for index, (res_name, res_id) in enumerate((("GLY", 1), ("SER", 2))):
        x = float(index * 3)
        for atom_name, dx, dy, dz in (
            ("N", 0.0, 0.0, 0.0),
            ("CA", 1.0, 0.0, 0.0),
            ("C", 2.0, 0.0, 0.0),
            ("O", 2.5, 0.5, 0.0),
        ):
            lines.append(
                _pdb_atom_line(serial, atom_name, res_name, "A", res_id, x + dx, dy, dz)
            )
            serial += 1
        if res_name != "GLY":
            lines.append(
                _pdb_atom_line(serial, "CB", res_name, "A", res_id, x + 1.0, 1.0, 0.0)
            )
            serial += 1
    path.write_text("".join(lines))


def _fake_cli_design_artifact(**kwargs) -> DesignCandidateArtifact:
    root = Path(kwargs["campaign_dir"])
    candidate_id = kwargs["candidate_id"]
    structure_path = structure_relpath(kwargs.get("artifact_stem", candidate_id))
    write_text_atomic(
        root / structure_path,
        "HEADER    FAKE CLI LAUNCH STRUCTURE\nEND\n",
    )
    return DesignCandidateArtifact(
        candidate_id=candidate_id,
        designed_sequence="ACDEFGHIK",
        sequence_path=None,
        critic_name=kwargs["critic_name"],
        structure_path=structure_path.as_posix(),
        design_metrics={
            "target_name": kwargs.get("target_name", "target"),
            "binder_scaffold": kwargs.get("binder_scaffold", "miniprotein"),
            "binder_type": kwargs.get("binder_scaffold", "miniprotein"),
            "binder_chain_id": "B",
        },
        critic_metrics={
            "iptm": 0.82,
            "ptm": 0.5,
            "distogram_iptm_proxy": 0.75,
        },
    )


def _launch_args(**overrides) -> argparse.Namespace:
    values = {
        "out": None,
        "esm_repo": None,
        "gpu_id": None,
        "gpus": None,
        "worker_prefix": "local-gpu",
        "max_shards": None,
        "max_shards_per_worker": None,
        "poll_interval": 0.05,
        "heartbeat_interval": 30.0,
        "stale_timeout": None,
        "enable_hf_xet": False,
        "disable_local_runtime_cache": False,
        "validation_msa_workers": None,
        "validation_msa_poll_interval": 0.01,
        "msa_max_requests_per_minute": 5.0,
        "max_designs": 50,
        "min_iptm": None,
        "require_hotspot_contact": None,
        "skip_export": False,
        "skip_validation": False,
        "skip_analysis": False,
        "analysis_top_k": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_ready_validation_tasks(
    campaign_dir: Path,
    *,
    count: int,
    start_index: int = 0,
    blocked_by_msa: bool = False,
) -> None:
    conn = initialize_database(campaign_dir / "campaign.sqlite")
    try:
        for offset in range(count):
            index = start_index + offset
            shard_id = f"shard_{index:06d}"
            candidate_id = f"candidate_{index:06d}"
            validation_id = f"validation_{index:06d}"
            conn.execute(
                """
                INSERT INTO shards (
                    shard_id,
                    seed,
                    batch_index,
                    status
                )
                VALUES (?, ?, ?, 'completed')
                """,
                (shard_id, index, index),
            )
            conn.execute(
                """
                INSERT INTO candidates (
                    candidate_id,
                    shard_id,
                    candidate_index,
                    designed_sequence,
                    status
                )
                VALUES (?, ?, 0, 'ACDE', 'completed')
                """,
                (candidate_id, shard_id),
            )
            conn.execute(
                """
                INSERT INTO validation_tasks (
                    validation_id,
                    candidate_id,
                    model_name,
                    validation_config_hash,
                    selection_rank,
                    status
                )
                VALUES (?, ?, 'protenix-v2', 'hash', ?, 'pending')
                """,
                (validation_id, candidate_id, index + 1),
            )
            if blocked_by_msa:
                msa_job_id = f"msa_{index:06d}"
                conn.execute(
                    """
                    INSERT INTO validation_msa_jobs (
                        msa_job_id,
                        scope,
                        cache_key,
                        msa_context_hash,
                        status,
                        representative_sequence
                    )
                    VALUES (?, 'target', ?, 'context', 'pending', 'ACDE')
                    """,
                    (msa_job_id, msa_job_id),
                )
                conn.execute(
                    """
                    INSERT INTO validation_msa_job_candidates (
                        candidate_id,
                        msa_job_id,
                        validation_config_hash,
                        reason
                    )
                    VALUES (?, ?, 'hash', 'test')
                    """,
                    (candidate_id, msa_job_id),
                )
        conn.commit()
    finally:
        conn.close()


def _write_cli_fake_protenix(root: Path) -> Path:
    script = root / "fake_cli_protenix.py"
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
        "chain_pair_ipsae": [[0.0, 0.61], [0.61, 0.0]],
    }
    (pred_dir / f"{name}_summary_confidence_sample_0.json").write_text(
        json.dumps(summary)
    )
    (pred_dir / f"{name}_full_data_sample_0.json").write_text("{}")
    (pred_dir / f"{name}_sample_0.cif").write_text("data_fake\\n#\\n")
""".lstrip()
    )
    return script


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


if __name__ == "__main__":
    unittest.main()
