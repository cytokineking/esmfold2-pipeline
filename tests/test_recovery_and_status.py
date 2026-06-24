from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from esmfold2_pipeline.artifacts import FastaRecord, write_fasta
from esmfold2_pipeline.db import CampaignStore, connect_database
from esmfold2_pipeline.execution.mock_worker import (
    MOCK_CRITIC_NAME,
    MOCK_SEQUENCE,
    plan_one_mock_shard,
    run_one_mock_shard,
)
from esmfold2_pipeline.planning import candidate_id
from esmfold2_pipeline.reports import inspect_campaign


class RecoveryAndStatusTest(unittest.TestCase):
    def test_stale_claim_returns_to_pending_and_retries_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)
            shard_id = plan_one_mock_shard(campaign_dir)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            claim = store.claim_next_pending_shard(worker_id="stale-worker")
            self.assertIsNotNone(claim)
            assert claim is not None
            self.assertEqual(claim.shard_id, shard_id)

            recovered = store.recover_stale_shards(
                stale_before="9999-01-01T00:00:00.000Z",
                error_message="test stale timeout",
            )
            self.assertEqual(recovered, 1)
            shard = store.fetch_one("SELECT * FROM shards WHERE shard_id = ?", (shard_id,))
            self.assertEqual(shard["status"], "pending")
            self.assertEqual(shard["attempt_count"], 1)
            stale_attempt = store.fetch_one(
                "SELECT * FROM attempts WHERE attempt_id = ?",
                (claim.attempt_id,),
            )
            self.assertEqual(stale_attempt["status"], "stale")
            conn.close()

            result = run_one_mock_shard(campaign_dir, worker_id="retry-worker")
            self.assertIsNotNone(result)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            shard = store.fetch_one("SELECT * FROM shards WHERE shard_id = ?", (shard_id,))
            self.assertEqual(shard["status"], "completed")
            self.assertEqual(shard["attempt_count"], 2)
            self.assertEqual(shard["claim_worker_id"], "retry-worker")

            counts = {
                row["name"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT 'candidates' AS name, COUNT(*) AS count FROM candidates
                    UNION ALL
                    SELECT 'critic_metrics', COUNT(*) FROM critic_metrics
                    UNION ALL
                    SELECT 'attempts', COUNT(*) FROM attempts
                    """
                )
            }
            self.assertEqual(
                counts,
                {"candidates": 1, "critic_metrics": 1, "attempts": 2},
            )
            status = inspect_campaign(campaign_dir)
            self.assertEqual(status.shard_status_counts, {"completed": 1})
            self.assertEqual(status.attempt_status_counts, {"completed": 1, "stale": 1})
            self.assertEqual(status.issues, [])
            conn.close()

    def test_failed_attempt_retries_until_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)
            shard_id = plan_one_mock_shard(campaign_dir)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            claim = store.claim_next_pending_shard(worker_id="failing-worker")
            assert claim is not None
            next_status = store.fail_shard(
                shard_id=shard_id,
                attempt_id=claim.attempt_id,
                error_message="simulated worker failure",
                exit_code=17,
            )
            self.assertEqual(next_status, "pending")
            attempt = store.fetch_one(
                "SELECT * FROM attempts WHERE attempt_id = ?",
                (claim.attempt_id,),
            )
            self.assertEqual(attempt["status"], "failed")
            self.assertEqual(attempt["exit_code"], 17)
            conn.close()

        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)
            shard_id = plan_one_mock_shard(campaign_dir)
            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            conn.execute(
                """
                UPDATE shards
                SET max_attempts = 1
                WHERE shard_id = ?
                """,
                (shard_id,),
            )
            conn.commit()

            claim = store.claim_next_pending_shard(worker_id="last-worker")
            assert claim is not None
            next_status = store.fail_shard(
                shard_id=shard_id,
                attempt_id=claim.attempt_id,
                error_message="final failure",
            )
            self.assertEqual(next_status, "failed")
            shard = store.fetch_one("SELECT * FROM shards WHERE shard_id = ?", (shard_id,))
            self.assertEqual(shard["status"], "failed")
            self.assertEqual(shard["error_message"], "final failure")
            conn.close()

    def test_status_reports_untracked_artifact_before_db_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)
            shard_id = plan_one_mock_shard(campaign_dir)
            cid = candidate_id(0, 0)
            sequence_relpath = (
                Path("shards") / shard_id / "candidates" / cid / "sequence.fasta"
            )

            write_fasta(
                campaign_dir / sequence_relpath,
                [FastaRecord(identifier=cid, sequence=MOCK_SEQUENCE)],
            )
            status = inspect_campaign(campaign_dir)
            self.assertEqual(status.untracked_artifact_count, 1)
            self.assertEqual(status.issues[0].kind, "untracked_artifact")
            self.assertEqual(status.issues[0].path, sequence_relpath.as_posix())

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            store.record_completed_candidate(
                candidate_id=cid,
                shard_id=shard_id,
                candidate_index=0,
                designed_sequence=MOCK_SEQUENCE,
                sequence_path=sequence_relpath.as_posix(),
            )
            conn.close()

            status = inspect_campaign(campaign_dir)
            self.assertEqual(status.issues, [])

    def test_status_reports_missing_artifact_after_db_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)
            plan_one_mock_shard(campaign_dir)
            result = run_one_mock_shard(campaign_dir)
            assert result is not None

            (campaign_dir / result.structure_path).unlink()
            status = inspect_campaign(campaign_dir)

            self.assertEqual(status.missing_artifact_count, 1)
            self.assertEqual(status.issues[0].kind, "missing_structure_artifact")
            self.assertEqual(status.issues[0].table, "critic_metrics")
            self.assertEqual(
                status.issues[0].row_id,
                f"{result.candidate_id}:{MOCK_CRITIC_NAME}",
            )


if __name__ == "__main__":
    unittest.main()
