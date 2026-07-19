from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from esmfold2_pipeline.db import CampaignStore, initialize_database
from esmfold2_pipeline.reports import inspect_campaign
from esmfold2_pipeline.validation import (
    DEFAULT_MSA_MAX_REQUESTS_PER_MINUTE,
    enqueue_msa_prefetch_for_candidate,
    plan_msa_prefetch,
    run_msa_prefetch_worker,
)


class ValidationMsaPrefetchTest(unittest.TestCase):
    def test_critic_prefetch_skips_design_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(
                root,
                validation={
                    "min_esm_iptm": 0.8,
                    "msa": {"binder": "single_sequence"},
                },
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.72)
            messages: list[str] = []

            result = enqueue_msa_prefetch_for_candidate(
                root,
                store=store,
                candidate_id="cand_000000_0000",
                critic_metrics={"iptm": 0.72},
                log=messages.append,
            )

            self.assertTrue(result.skipped)
            self.assertEqual(result.queued_jobs, 0)
            self.assertIn("below prefetch threshold", messages[0])
            count = conn.execute("SELECT COUNT(*) FROM validation_msa_jobs").fetchone()[0]
            self.assertEqual(count, 0)
            conn.close()

    def test_msa_plan_and_worker_materialize_miniprotein_single_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(
                root,
                validation={
                    "min_esm_iptm": 0.7,
                    "msa": {"binder": "single_sequence"},
                },
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.82)
            conn.close()

            plan = plan_msa_prefetch(root, log=lambda _message: None)
            self.assertEqual(plan.candidate_count, 1)
            self.assertEqual(plan.queued_jobs, 1)

            run = run_msa_prefetch_worker(
                root,
                max_requests_per_minute=DEFAULT_MSA_MAX_REQUESTS_PER_MINUTE,
                log=lambda _message: None,
            )
            self.assertEqual(run.completed_jobs, 1)
            self.assertEqual(run.failed_jobs, 0)

            conn = initialize_database(root / "campaign.sqlite")
            try:
                job = conn.execute("SELECT * FROM validation_msa_jobs").fetchone()
                self.assertEqual(job["status"], "ready")
                paths = json.loads(job["cache_paths_json"])
                self.assertEqual(
                    Path(paths["non_pairing_path"]).read_text(),
                    ">query\nACDEFGHIK\n",
                )
            finally:
                conn.close()

    def test_msa_worker_repairs_ready_job_with_missing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(
                root,
                validation={
                    "min_esm_iptm": 0.7,
                    "msa": {"binder": "single_sequence"},
                },
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.82)
            conn.close()

            plan_msa_prefetch(root)
            first = run_msa_prefetch_worker(root)
            self.assertEqual(first.completed_jobs, 1)

            conn = initialize_database(root / "campaign.sqlite")
            try:
                row = conn.execute(
                    "SELECT cache_paths_json FROM validation_msa_jobs"
                ).fetchone()
                paths = json.loads(row["cache_paths_json"])
            finally:
                conn.close()
            Path(paths["non_pairing_path"]).unlink()

            messages: list[str] = []
            repaired = run_msa_prefetch_worker(root, log=messages.append)

            self.assertEqual(repaired.completed_jobs, 1)
            self.assertTrue(Path(paths["non_pairing_path"]).is_file())
            self.assertIn(
                "reopened ready MSA jobs with missing cache: 1",
                messages,
            )

    def test_target_template_suppresses_target_msa_prefetch_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_target_template(root)
            conn = _campaign_db(
                root,
                validation={
                    "msa": {
                        "target": "server",
                        "server_url": "https://msa.example",
                    },
                },
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.82)
            messages: list[str] = []

            result = enqueue_msa_prefetch_for_candidate(
                root,
                store=store,
                candidate_id="cand_000000_0000",
                critic_metrics={"iptm": 0.82},
                log=messages.append,
            )

            self.assertFalse(result.skipped)
            self.assertEqual(result.queued_jobs, 0)
            self.assertIn("target structural template available", messages[0])
            count = conn.execute("SELECT COUNT(*) FROM validation_msa_jobs").fetchone()[0]
            self.assertEqual(count, 0)
            conn.close()

    def test_explicit_msa_overrides_target_template_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_target_template(root)
            conn = _campaign_db(
                root,
                validation={
                    "msa": {
                        "use_msa": True,
                        "target": "server",
                        "server_url": "https://msa.example",
                    },
                },
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.82)

            result = enqueue_msa_prefetch_for_candidate(
                root,
                store=store,
                candidate_id="cand_000000_0000",
                critic_metrics={"iptm": 0.82},
                log=lambda _message: None,
            )

            self.assertEqual(result.queued_jobs, 2)
            scopes = [
                row["scope"]
                for row in conn.execute(
                    "SELECT scope FROM validation_msa_jobs ORDER BY scope"
                ).fetchall()
            ]
            self.assertEqual(scopes, ["miniprotein_single_sequence", "target"])
            conn.close()

    def test_structure_target_chain_summary_provides_target_msa_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_target_template(root)
            conn = _campaign_db(
                root,
                target={
                    "name": "structure_target",
                    "structure": "target.pdb",
                    "chains": ["A"],
                    "sequences": {},
                },
                validation={
                    "msa": {
                        "use_msa": True,
                        "target": "server",
                        "server_url": "https://msa.example",
                    },
                },
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.82)

            result = enqueue_msa_prefetch_for_candidate(
                root,
                store=store,
                candidate_id="cand_000000_0000",
                critic_metrics={"iptm": 0.82},
                log=lambda _message: None,
            )

            self.assertEqual(result.queued_jobs, 2)
            rows = conn.execute(
                """
                SELECT scope, representative_sequence
                FROM validation_msa_jobs
                ORDER BY scope
                """
            ).fetchall()
            self.assertEqual(
                [(row["scope"], row["representative_sequence"]) for row in rows],
                [
                    ("miniprotein_single_sequence", "ACDEFGHIK"),
                    ("target", "GGGG"),
                ],
            )
            conn.close()

    def test_nested_explicit_msa_overrides_top_level_false_for_template_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_target_template(root)
            conn = _campaign_db(
                root,
                validation={
                    "use_msa": False,
                    "msa": {
                        "use_msa": True,
                        "target": "server",
                        "server_url": "https://msa.example",
                    },
                },
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.82)

            result = enqueue_msa_prefetch_for_candidate(
                root,
                store=store,
                candidate_id="cand_000000_0000",
                critic_metrics={"iptm": 0.82},
                log=lambda _message: None,
            )

            self.assertEqual(result.queued_jobs, 2)
            conn.close()

    def test_target_template_mismatch_does_not_suppress_target_msa_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_target_template(root, sequence="MISMATCH")
            conn = _campaign_db(
                root,
                validation={
                    "msa": {
                        "target": "server",
                        "server_url": "https://msa.example",
                    },
                },
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.82)

            result = enqueue_msa_prefetch_for_candidate(
                root,
                store=store,
                candidate_id="cand_000000_0000",
                critic_metrics={"iptm": 0.82},
                log=lambda _message: None,
            )

            self.assertEqual(result.queued_jobs, 2)
            scopes = [
                row["scope"]
                for row in conn.execute(
                    "SELECT scope FROM validation_msa_jobs ORDER BY scope"
                ).fetchall()
            ]
            self.assertEqual(scopes, ["miniprotein_single_sequence", "target"])
            conn.close()

    def test_builtin_vhh_framework_template_suppresses_binder_msa_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(root, validation={"msa": {"binder": "auto"}})
            store = CampaignStore(conn)
            _insert_completed_candidate(
                store,
                iptm=0.82,
                binder_scaffold="vhh",
                framework="caplacizumab_framework_vhh",
                framework_source="builtin",
            )
            messages: list[str] = []

            result = enqueue_msa_prefetch_for_candidate(
                root,
                store=store,
                candidate_id="cand_000000_0000",
                critic_metrics={"iptm": 0.82},
                log=messages.append,
            )

            self.assertFalse(result.skipped)
            self.assertEqual(result.queued_jobs, 0)
            self.assertTrue(
                any("VHH framework structural template available" in item for item in messages)
            )
            count = conn.execute("SELECT COUNT(*) FROM validation_msa_jobs").fetchone()[0]
            self.assertEqual(count, 0)
            conn.close()

    def test_explicit_msa_overrides_vhh_framework_template_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(root, validation={"msa": {"use_msa": True}})
            store = CampaignStore(conn)
            _insert_completed_candidate(
                store,
                iptm=0.82,
                binder_scaffold="vhh",
                framework="caplacizumab_framework_vhh",
                framework_source="builtin",
            )

            with patch(
                "esmfold2_pipeline.validation.msa_prefetch.analyze_vhh_sequence",
                return_value={"lengths_only_template_key_hash": "fake-vhh"},
            ):
                result = enqueue_msa_prefetch_for_candidate(
                    root,
                    store=store,
                    candidate_id="cand_000000_0000",
                    critic_metrics={"iptm": 0.82},
                    log=lambda _message: None,
                )

            self.assertEqual(result.queued_jobs, 1)
            job = conn.execute("SELECT scope FROM validation_msa_jobs").fetchone()
            self.assertEqual(job["scope"], "vhh_binder_group")
            conn.close()

    def test_builtin_scfv_framework_template_avoids_unsupported_prefetch_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(root, validation={"msa": {"binder": "auto"}})
            store = CampaignStore(conn)
            _insert_completed_candidate(
                store,
                iptm=0.82,
                binder_scaffold="scfv",
                framework="anifrolumab_framework_vhvl",
                framework_source="builtin",
            )

            result = enqueue_msa_prefetch_for_candidate(
                root,
                store=store,
                candidate_id="cand_000000_0000",
                critic_metrics={"iptm": 0.82},
                log=lambda _message: None,
            )

            self.assertFalse(result.skipped)
            self.assertEqual(result.queued_jobs, 0)
            conn.close()

    def test_explicit_scfv_binder_msa_is_skipped_with_clear_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(root, validation={"msa": {"use_msa": True}})
            store = CampaignStore(conn)
            _insert_completed_candidate(
                store,
                iptm=0.82,
                binder_scaffold="scfv",
                framework="anifrolumab_framework_vhvl",
                framework_source="builtin",
            )
            messages: list[str] = []

            result = enqueue_msa_prefetch_for_candidate(
                root,
                store=store,
                candidate_id="cand_000000_0000",
                critic_metrics={"iptm": 0.82},
                log=messages.append,
            )

            self.assertTrue(result.skipped)
            self.assertEqual(result.queued_jobs, 0)
            self.assertTrue(
                any("scFv binder MSA support is not implemented" in item for item in messages)
            )
            conn.close()

    def test_validation_claim_waits_for_dependent_msa_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(
                root,
                validation={"msa": {"binder": "single_sequence"}},
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.9)
            store.create_validation_task(
                validation_id="val_cand",
                candidate_id="cand_000000_0000",
                model_name="protenix-v2",
                validation_config_hash="hash",
                selection_rank=1,
            )
            store.create_or_update_msa_job(
                msa_job_id="msa_blocking",
                scope="miniprotein_single_sequence",
                cache_key="miniprotein:test",
                msa_context_hash="ctx",
                candidate_id="cand_000000_0000",
                representative_sequence="ACDEFGHIK",
                member_sequences=("ACDEFGHIK",),
                metadata={},
                validation_config_hash="hash",
            )

            blocked = store.claim_next_pending_validation_tasks(
                worker_id="validator",
                batch_size=1,
            )
            self.assertEqual(blocked, [])

            store.complete_msa_job(msa_job_id="msa_blocking")
            ready = store.claim_next_pending_validation_tasks(
                worker_id="validator",
                batch_size=1,
            )
            self.assertEqual([claim.validation_id for claim in ready], ["val_cand"])
            task = conn.execute(
                "SELECT attempt_count FROM validation_tasks WHERE validation_id = 'val_cand'"
            ).fetchone()
            self.assertEqual(task["attempt_count"], 1)
            conn.close()

    def test_validation_claim_ignores_unscoped_stale_msa_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(
                root,
                validation={"msa": {"binder": "single_sequence"}},
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.9)
            store.create_validation_task(
                validation_id="val_cand",
                candidate_id="cand_000000_0000",
                model_name="protenix-v2",
                validation_config_hash="new_hash",
                selection_rank=1,
            )
            store.create_or_update_msa_job(
                msa_job_id="msa_old_unscoped",
                scope="miniprotein_single_sequence",
                cache_key="miniprotein:test",
                msa_context_hash="ctx",
                candidate_id="cand_000000_0000",
                representative_sequence="ACDEFGHIK",
                member_sequences=("ACDEFGHIK",),
                metadata={},
            )

            ready = store.claim_next_pending_validation_tasks(
                worker_id="validator",
                batch_size=1,
            )

            self.assertEqual([claim.validation_id for claim in ready], ["val_cand"])
            conn.close()

    def test_ready_msa_job_reopens_when_required_cache_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(
                root,
                validation={"msa": {"binder": "single_sequence"}},
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.9)
            store.create_validation_task(
                validation_id="val_cand",
                candidate_id="cand_000000_0000",
                model_name="protenix-v2",
                validation_config_hash="hash",
                selection_rank=1,
            )
            cache_dir = (
                root
                / "validation"
                / "protenix_v2"
                / "msa_cache"
                / "binder"
                / "missing"
            )
            cache_dir.mkdir(parents=True)
            (cache_dir / "metadata.json").write_text("{}")
            store.create_or_update_msa_job(
                msa_job_id="msa_missing_cache",
                scope="miniprotein_single_sequence",
                cache_key="miniprotein:test",
                msa_context_hash="ctx",
                candidate_id="cand_000000_0000",
                representative_sequence="ACDEFGHIK",
                member_sequences=("ACDEFGHIK",),
                metadata={},
                validation_config_hash="hash",
            )
            store.complete_msa_job(
                msa_job_id="msa_missing_cache",
                cache_paths={
                    "cache_dir": str(cache_dir),
                    "non_pairing_path": str(cache_dir / "non_pairing.a3m"),
                    "metadata_path": str(cache_dir / "metadata.json"),
                },
            )
            conn.execute(
                """
                UPDATE validation_msa_jobs
                SET attempt_count = 3,
                    max_attempts = 3
                WHERE msa_job_id = 'msa_missing_cache'
                """
            )
            conn.commit()

            reopened = store.reopen_ready_msa_jobs_with_missing_cache(base_dir=root)
            self.assertEqual(reopened, 1)
            job = conn.execute(
                """
                SELECT status, attempt_count, error_message
                FROM validation_msa_jobs
                WHERE msa_job_id = 'msa_missing_cache'
                """
            ).fetchone()
            self.assertEqual(job["status"], "pending")
            self.assertEqual(job["attempt_count"], 0)
            self.assertIn("non_pairing.a3m", job["error_message"])
            claim = store.claim_next_pending_msa_job(worker_id="msa-worker")
            self.assertIsNotNone(claim)
            assert claim is not None
            self.assertEqual(claim.msa_job_id, "msa_missing_cache")
            self.assertEqual(claim.attempt_count, 1)

            blocked = store.claim_next_pending_validation_tasks(
                worker_id="validator",
                batch_size=1,
            )
            self.assertEqual(blocked, [])
            task = conn.execute(
                "SELECT attempt_count FROM validation_tasks WHERE validation_id = 'val_cand'"
            ).fetchone()
            self.assertEqual(task["attempt_count"], 0)
            conn.close()

    def test_ready_msa_job_keeps_campaign_prefixed_relative_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_dir = root / "gpu_runs" / "campaign"
            campaign_dir.mkdir(parents=True)
            conn = _campaign_db(
                campaign_dir,
                validation={"msa": {"binder": "single_sequence"}},
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.9)
            store.create_validation_task(
                validation_id="val_cand",
                candidate_id="cand_000000_0000",
                model_name="protenix-v2",
                validation_config_hash="hash",
                selection_rank=1,
            )
            cache_dir = (
                campaign_dir
                / "validation"
                / "protenix_v2"
                / "msa_cache"
                / "binder"
                / "ready"
            )
            cache_dir.mkdir(parents=True)
            (cache_dir / "non_pairing.a3m").write_text(">query\nACDEFGHIK\n")
            (cache_dir / "metadata.json").write_text("{}")
            campaign_prefixed_cache_dir = cache_dir.relative_to(root)
            store.create_or_update_msa_job(
                msa_job_id="msa_campaign_prefixed_cache",
                scope="miniprotein_single_sequence",
                cache_key="miniprotein:test",
                msa_context_hash="ctx",
                candidate_id="cand_000000_0000",
                representative_sequence="ACDEFGHIK",
                member_sequences=("ACDEFGHIK",),
                metadata={},
                validation_config_hash="hash",
            )
            store.complete_msa_job(
                msa_job_id="msa_campaign_prefixed_cache",
                cache_paths={
                    "cache_dir": str(campaign_prefixed_cache_dir),
                    "non_pairing_path": str(
                        campaign_prefixed_cache_dir / "non_pairing.a3m"
                    ),
                    "metadata_path": str(campaign_prefixed_cache_dir / "metadata.json"),
                },
            )

            cwd = Path.cwd()
            try:
                os.chdir(root)
                reopened = store.reopen_ready_msa_jobs_with_missing_cache(
                    base_dir=campaign_dir
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(reopened, 0)
            job = conn.execute(
                """
                SELECT status, error_message
                FROM validation_msa_jobs
                WHERE msa_job_id = 'msa_campaign_prefixed_cache'
                """
            ).fetchone()
            self.assertEqual(job["status"], "ready")
            self.assertIsNone(job["error_message"])
            ready = store.claim_next_pending_validation_tasks(
                worker_id="validator",
                batch_size=1,
            )
            self.assertEqual([claim.validation_id for claim in ready], ["val_cand"])
            conn.close()

    def test_status_reports_msa_blocked_and_failed_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(
                root,
                validation={"msa": {"binder": "single_sequence"}},
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.9)
            store.create_validation_task(
                validation_id="val_cand",
                candidate_id="cand_000000_0000",
                model_name="protenix-v2",
                validation_config_hash="hash",
                selection_rank=1,
            )
            store.create_or_update_msa_job(
                msa_job_id="msa_blocking",
                scope="miniprotein_single_sequence",
                cache_key="miniprotein:test",
                msa_context_hash="ctx",
                candidate_id="cand_000000_0000",
                representative_sequence="ACDEFGHIK",
                member_sequences=("ACDEFGHIK",),
                metadata={},
                validation_config_hash="hash",
            )
            conn.close()

            status = inspect_campaign(root)
            self.assertEqual(status.validation_msa_status_counts, {"pending": 1})
            self.assertEqual(status.validation_msa_blocked_counts, {"pending": 1})
            self.assertEqual(status.validation_msa_failures, [])

            conn = initialize_database(root / "campaign.sqlite")
            conn.execute(
                """
                UPDATE validation_msa_jobs
                SET status = 'failed',
                    error_message = 'mmseqs request failed'
                WHERE msa_job_id = 'msa_blocking'
                """
            )
            conn.commit()
            conn.close()

            failed = inspect_campaign(root)
            self.assertEqual(failed.validation_msa_status_counts, {"failed": 1})
            self.assertEqual(failed.validation_msa_blocked_counts, {"failed": 1})
            self.assertEqual(len(failed.validation_msa_failures), 1)
            self.assertEqual(
                failed.validation_msa_failures[0].candidate_ids,
                ("cand_000000_0000",),
            )
            self.assertIn(
                "mmseqs request failed",
                failed.validation_msa_failures[0].error_message,
            )

    def test_retry_failed_msa_jobs_resets_failed_dependency_for_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(
                root,
                validation={"msa": {"binder": "single_sequence"}},
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.9)
            store.create_validation_task(
                validation_id="val_cand",
                candidate_id="cand_000000_0000",
                model_name="protenix-v2",
                validation_config_hash="hash",
                selection_rank=1,
            )
            store.create_or_update_msa_job(
                msa_job_id="msa_blocking",
                scope="miniprotein_single_sequence",
                cache_key="miniprotein:test",
                msa_context_hash="ctx",
                candidate_id="cand_000000_0000",
                representative_sequence="ACDEFGHIK",
                member_sequences=("ACDEFGHIK",),
                metadata={},
                validation_config_hash="hash",
                max_attempts=1,
            )
            conn.execute(
                """
                UPDATE validation_msa_jobs
                SET status = 'failed',
                    attempt_count = 1,
                    error_message = 'exhausted retry budget',
                    completed_at = '2026-01-01T00:00:00.000Z'
                WHERE msa_job_id = 'msa_blocking'
                """
            )
            conn.commit()

            blocked = store.claim_next_pending_validation_tasks(
                worker_id="validator",
                batch_size=1,
            )
            self.assertEqual(blocked, [])

            retried = store.retry_failed_msa_jobs(
                candidate_ids=("cand_000000_0000",),
            )
            self.assertEqual(retried, 1)
            job = conn.execute(
                """
                SELECT status, attempt_count, error_message, completed_at
                FROM validation_msa_jobs
                WHERE msa_job_id = 'msa_blocking'
                """
            ).fetchone()
            self.assertEqual(job["status"], "pending")
            self.assertEqual(job["attempt_count"], 0)
            self.assertIsNone(job["error_message"])
            self.assertIsNone(job["completed_at"])

            claim = store.claim_next_pending_msa_job(worker_id="msa-worker")
            self.assertIsNotNone(claim)
            assert claim is not None
            self.assertEqual(claim.msa_job_id, "msa_blocking")
            conn.close()

    def test_worker_recovers_stale_running_msa_job_before_claiming(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(
                root,
                validation={"msa": {"binder": "single_sequence"}},
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.9)
            store.create_validation_task(
                validation_id="val_cand",
                candidate_id="cand_000000_0000",
                model_name="protenix-v2",
                validation_config_hash="hash",
                selection_rank=1,
            )
            store.create_or_update_msa_job(
                msa_job_id="msa_running",
                scope="miniprotein_single_sequence",
                cache_key="miniprotein:test",
                msa_context_hash="ctx",
                candidate_id="cand_000000_0000",
                representative_sequence="ACDEFGHIK",
                member_sequences=("ACDEFGHIK",),
                metadata={
                    "binder_sequence": "ACDEFGHIK",
                    "binder_scaffold": "miniprotein",
                    "msa_config": {
                        "target_mode": "none",
                        "binder_mode": "single_sequence",
                        "pairing_strategy": "greedy",
                    },
                },
                validation_config_hash="hash",
            )
            claimed = store.claim_next_pending_msa_job(worker_id="dead-msa-worker")
            self.assertIsNotNone(claimed)
            conn.execute(
                """
                UPDATE validation_msa_jobs
                SET heartbeat_at = '2026-01-01T00:00:00.000Z'
                WHERE msa_job_id = 'msa_running'
                """
            )
            conn.commit()

            blocked = store.claim_next_pending_validation_tasks(
                worker_id="validator",
                batch_size=1,
            )
            self.assertEqual(blocked, [])

            result = run_msa_prefetch_worker(
                root,
                worker_id="recovery-msa-worker",
                max_jobs=1,
                stale_timeout_seconds=1.0,
            )

            self.assertEqual(result.recovered_stale_jobs, 1)
            self.assertEqual(result.completed_jobs, 1)
            job = conn.execute(
                """
                SELECT status, attempt_count, error_message
                FROM validation_msa_jobs
                WHERE msa_job_id = 'msa_running'
                """
            ).fetchone()
            self.assertEqual(job["status"], "ready")
            self.assertEqual(job["attempt_count"], 2)
            self.assertIsNone(job["error_message"])

            unblocked = store.claim_next_pending_validation_tasks(
                worker_id="validator",
                batch_size=1,
            )
            self.assertEqual(len(unblocked), 1)
            self.assertEqual(unblocked[0].validation_id, "val_cand")
            conn.close()

    def test_worker_marks_exhausted_stale_msa_job_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(
                root,
                validation={"msa": {"binder": "single_sequence"}},
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.9)
            store.create_validation_task(
                validation_id="val_cand",
                candidate_id="cand_000000_0000",
                model_name="protenix-v2",
                validation_config_hash="hash",
                selection_rank=1,
            )
            store.create_or_update_msa_job(
                msa_job_id="msa_exhausted",
                scope="miniprotein_single_sequence",
                cache_key="miniprotein:test",
                msa_context_hash="ctx",
                candidate_id="cand_000000_0000",
                representative_sequence="ACDEFGHIK",
                member_sequences=("ACDEFGHIK",),
                metadata={
                    "binder_sequence": "ACDEFGHIK",
                    "binder_scaffold": "miniprotein",
                    "msa_config": {
                        "target_mode": "none",
                        "binder_mode": "single_sequence",
                        "pairing_strategy": "greedy",
                    },
                },
                validation_config_hash="hash",
                max_attempts=1,
            )
            claimed = store.claim_next_pending_msa_job(worker_id="dead-msa-worker")
            self.assertIsNotNone(claimed)
            conn.execute(
                """
                UPDATE validation_msa_jobs
                SET heartbeat_at = '2026-01-01T00:00:00.000Z'
                WHERE msa_job_id = 'msa_exhausted'
                """
            )
            conn.commit()

            result = run_msa_prefetch_worker(
                root,
                worker_id="recovery-msa-worker",
                max_jobs=1,
                stale_timeout_seconds=1.0,
            )

            self.assertEqual(result.recovered_stale_jobs, 1)
            self.assertEqual(result.completed_jobs, 0)
            self.assertEqual(result.failed_jobs, 0)
            self.assertTrue(result.no_pending)
            job = conn.execute(
                """
                SELECT status, attempt_count, error_message, completed_at
                FROM validation_msa_jobs
                WHERE msa_job_id = 'msa_exhausted'
                """
            ).fetchone()
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["attempt_count"], 1)
            self.assertIn("heartbeat exceeded 1s", job["error_message"])
            self.assertIsNotNone(job["completed_at"])

            blocked = store.claim_next_pending_validation_tasks(
                worker_id="validator",
                batch_size=1,
            )
            self.assertEqual(blocked, [])
            task = conn.execute(
                """
                SELECT status, attempt_count
                FROM validation_tasks
                WHERE validation_id = 'val_cand'
                """
            ).fetchone()
            self.assertEqual(task["status"], "pending")
            self.assertEqual(task["attempt_count"], 0)
            conn.close()

    def test_ready_msa_job_reopens_when_new_candidate_member_is_added(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conn = _campaign_db(
                root,
                validation={"msa": {"binder": "single_sequence"}},
            )
            store = CampaignStore(conn)
            _insert_completed_candidate(store, iptm=0.9)
            store.create_or_update_msa_job(
                msa_job_id="msa_shared",
                scope="vhh_binder_group",
                cache_key="vhh:template",
                msa_context_hash="ctx",
                candidate_id="cand_000000_0000",
                representative_sequence="ACDEFGHIK",
                member_sequences=("ACDEFGHIK",),
                metadata={
                    "binder_sequence": "ACDEFGHIK",
                    "candidate_id": "cand_000000_0000",
                },
            )
            store.complete_msa_job(msa_job_id="msa_shared")
            conn.execute(
                """
                UPDATE validation_msa_jobs
                SET attempt_count = 3,
                    max_attempts = 3,
                    heartbeat_at = '2026-01-01T00:00:00.000Z',
                    completed_at = '2026-01-01T00:00:00.000Z',
                    error_message = 'previous transient warning'
                WHERE msa_job_id = 'msa_shared'
                """
            )
            conn.commit()

            store.record_completed_candidate(
                candidate_id="cand_000000_0001",
                shard_id="shard_000000",
                candidate_index=1,
                designed_sequence="LMNPQRSTV",
                sequence_path=None,
                design_metrics={"binder_scaffold": "vhh"},
            )
            created = store.create_or_update_msa_job(
                msa_job_id="ignored_duplicate_id",
                scope="vhh_binder_group",
                cache_key="vhh:template",
                msa_context_hash="ctx",
                candidate_id="cand_000000_0001",
                representative_sequence="LMNPQRSTV",
                member_sequences=("LMNPQRSTV",),
                metadata={
                    "binder_sequence": "LMNPQRSTV",
                    "candidate_id": "cand_000000_0001",
                },
            )

            self.assertFalse(created)
            job = conn.execute(
                """
                SELECT status,
                       attempt_count,
                       heartbeat_at,
                       completed_at,
                       next_eligible_at,
                       error_message,
                       representative_sequence,
                       member_sequences_json,
                       metadata_json
                FROM validation_msa_jobs
                """
            ).fetchone()
            self.assertEqual(job["status"], "pending")
            self.assertEqual(job["attempt_count"], 0)
            self.assertIsNone(job["heartbeat_at"])
            self.assertIsNone(job["completed_at"])
            self.assertIsNone(job["next_eligible_at"])
            self.assertIsNone(job["error_message"])
            self.assertEqual(job["representative_sequence"], "ACDEFGHIK")
            self.assertEqual(
                json.loads(job["member_sequences_json"]),
                ["ACDEFGHIK", "LMNPQRSTV"],
            )
            self.assertEqual(
                json.loads(job["metadata_json"])["binder_sequence"],
                "ACDEFGHIK",
            )
            self.assertEqual(
                json.loads(job["metadata_json"])["candidate_id"],
                "cand_000000_0000",
            )
            claim = store.claim_next_pending_msa_job(worker_id="msa-worker")
            self.assertIsNotNone(claim)
            assert claim is not None
            self.assertEqual(claim.msa_job_id, "msa_shared")
            self.assertEqual(claim.attempt_count, 1)
            conn.close()


def _campaign_db(
    root: Path,
    *,
    validation: dict,
    target: dict | None = None,
) -> object:
    conn = initialize_database(
        root / "campaign.sqlite",
        config_hash="prefetch-test",
        resolved_config={
            "target": target or {"name": "target", "sequence": "GGGG"},
            "binder": {"scaffold": "miniprotein"},
            "campaign": {"num_designs": 1},
            "validation": validation,
        },
    )
    store = CampaignStore(conn)
    store.create_shard(
        shard_id="shard_000000",
        seed=0,
        batch_index=0,
        target_key="target",
        binder_key="binder",
        critic_set=["critic"],
    )
    return conn


def _insert_completed_candidate(
    store: CampaignStore,
    *,
    iptm: float,
    binder_scaffold: str = "miniprotein",
    framework: str | None = None,
    framework_source: str | None = None,
) -> None:
    store.record_completed_candidate(
        candidate_id="cand_000000_0000",
        shard_id="shard_000000",
        candidate_index=0,
        designed_sequence="ACDEFGHIK",
        sequence_path=None,
        design_metrics={
            "binder_scaffold": binder_scaffold,
            "framework": framework,
            "framework_name": framework,
            "framework_source": framework_source,
        },
    )
    store.record_completed_critic(
        candidate_id="cand_000000_0000",
        critic_name="critic",
        structure_path="esmfold2/structures/cand.pdb",
        metrics={"iptm": iptm, "ptm": 0.5},
    )


def _write_target_template(root: Path, *, sequence: str = "GGGG") -> None:
    target_dir = root / "target"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "normalized_target.cif").write_text("data_target\n")
    (target_dir / "chain_summary.json").write_text(
        json.dumps(
            {
                "chains": [
                    {
                        "canonical_chain_id": "A",
                        "sequence": sequence,
                    }
                ]
            }
        )
    )


if __name__ == "__main__":
    unittest.main()
