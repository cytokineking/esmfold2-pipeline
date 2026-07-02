from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from esmfold2_pipeline.db import CampaignStore, connect_database
from esmfold2_pipeline.execution import run_multi_campaign
from esmfold2_pipeline.execution.multi import _worker_command
from esmfold2_pipeline.execution.mock_worker import (
    plan_one_mock_shard,
    run_one_mock_shard,
)
from esmfold2_pipeline.reports import inspect_campaign


class MultiGPUExecutorTest(unittest.TestCase):
    def test_two_mock_workers_share_sqlite_claims_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            plan_one_mock_shard(campaign_dir)

            result = run_multi_campaign(
                campaign_dir,
                gpu_ids=["0", "1"],
                worker_prefix="test-worker",
                poll_interval_seconds=0.05,
                python_executable=sys.executable,
                worker_subcommand="run-mock",
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.failed_workers, 0)
            self.assertEqual(result.completed_shards, 1)
            self.assertEqual(len(result.worker_results), 2)
            self.assertEqual(
                sorted(worker.gpu_id for worker in result.worker_results),
                ["0", "1"],
            )
            for worker in result.worker_results:
                self.assertTrue(worker.log_path.exists())

            status = inspect_campaign(campaign_dir)
            self.assertEqual(status.shard_status_counts, {"completed": 1})
            self.assertEqual(status.candidate_status_counts, {"completed": 1})
            self.assertEqual(status.critic_status_counts, {"completed": 1})
            self.assertEqual(status.issues, [])

    def test_run_multi_recovers_stale_shards_before_starting_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            shard_id = plan_one_mock_shard(campaign_dir)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            claim = store.claim_next_pending_shard(worker_id="preempted-worker")
            assert claim is not None
            self.assertEqual(claim.shard_id, shard_id)
            conn.execute(
                """
                UPDATE shards
                SET heartbeat_at = '1970-01-01T00:00:00.000Z'
                WHERE shard_id = ?
                """,
                (shard_id,),
            )
            conn.commit()
            conn.close()

            result = run_multi_campaign(
                campaign_dir,
                gpu_ids=["0"],
                worker_prefix="resume-worker",
                poll_interval_seconds=0.05,
                python_executable=sys.executable,
                worker_subcommand="run-mock",
                stale_after_seconds=1.0,
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.recovered_shards, 1)
            self.assertEqual(result.startup_recovered_shards, 1)
            self.assertEqual(result.completed_shards, 1)
            status = inspect_campaign(campaign_dir)
            self.assertEqual(status.shard_status_counts, {"completed": 1})
            self.assertEqual(status.attempt_status_counts, {"completed": 1, "stale": 1})
            self.assertEqual(status.issues, [])

    def test_duplicate_gpu_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            plan_one_mock_shard(campaign_dir)

            with self.assertRaisesRegex(ValueError, "unique"):
                run_multi_campaign(
                    campaign_dir,
                    gpu_ids=["0,1", "1"],
                    worker_subcommand="run-mock",
                )

    def test_gpu_ranges_expand_to_worker_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            plan_one_mock_shard(campaign_dir)

            result = run_multi_campaign(
                campaign_dir,
                gpu_ids=["0-1"],
                worker_prefix="range-worker",
                poll_interval_seconds=0.05,
                python_executable=sys.executable,
                worker_subcommand="run-mock",
            )

            self.assertTrue(result.ok)
            self.assertEqual(
                sorted(worker.gpu_id for worker in result.worker_results),
                ["0", "1"],
            )

    def test_all_gpus_uses_cuda_visible_devices_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            plan_one_mock_shard(campaign_dir)

            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,3"}):
                result = run_multi_campaign(
                    campaign_dir,
                    gpu_ids=["all"],
                    worker_prefix="all-worker",
                    poll_interval_seconds=0.05,
                    python_executable=sys.executable,
                    worker_subcommand="run-mock",
                )

            self.assertTrue(result.ok)
            self.assertEqual(
                sorted(worker.gpu_id for worker in result.worker_results),
                ["2", "3"],
            )

    def test_failed_worker_process_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            plan_one_mock_shard(campaign_dir)

            result = run_multi_campaign(
                campaign_dir,
                gpu_ids=["0"],
                python_executable=shutil.which("false") or "/usr/bin/false",
                poll_interval_seconds=0.05,
                worker_subcommand="run-mock",
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.failed_workers, 1)
            self.assertEqual(result.worker_results[0].returncode, 1)
            self.assertEqual(result.worker_results[0].recovered_shards, 0)

    def test_worker_command_can_disable_local_runtime_cache(self) -> None:
        command = _worker_command(
            executable=sys.executable,
            worker_subcommand="run",
            campaign_dir=Path("/tmp/campaign"),
            worker_id="worker-0",
            gpu_id="0",
            esm_repo=Path("/tmp/esm"),
            max_shards_per_worker=1,
            heartbeat_interval_seconds=30.0,
            disable_hf_xet=True,
            disable_local_runtime_cache=True,
        )

        self.assertIn("--disable-local-runtime-cache", command)


class FailedWorkerRecoveryTest(unittest.TestCase):
    def test_failed_worker_releases_running_shard_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            shard_id = plan_one_mock_shard(campaign_dir)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            claim = store.claim_next_pending_shard(
                worker_id="dead-worker",
                pid=12345,
                gpu_id="0",
            )
            assert claim is not None

            recovered = store.recover_failed_worker_shards(
                worker_id="dead-worker",
                pid=12345,
                exit_code=9,
                error_message="test worker crash",
            )
            self.assertEqual(recovered, 1)

            shard = store.fetch_one(
                "SELECT * FROM shards WHERE shard_id = ?",
                (shard_id,),
            )
            self.assertEqual(shard["status"], "pending")
            self.assertEqual(shard["attempt_count"], 1)
            self.assertIsNone(shard["claim_worker_id"])

            attempt = store.fetch_one(
                "SELECT * FROM attempts WHERE attempt_id = ?",
                (claim.attempt_id,),
            )
            self.assertEqual(attempt["status"], "failed")
            self.assertEqual(attempt["exit_code"], 9)
            conn.close()

            retry = run_one_mock_shard(campaign_dir, worker_id="retry-worker")
            self.assertIsNotNone(retry)

            status = inspect_campaign(campaign_dir)
            self.assertEqual(status.shard_status_counts, {"completed": 1})
            self.assertEqual(status.attempt_status_counts, {"completed": 1, "failed": 1})
            self.assertEqual(status.issues, [])


if __name__ == "__main__":
    unittest.main()
