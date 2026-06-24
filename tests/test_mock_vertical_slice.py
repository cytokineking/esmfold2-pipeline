from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from esmfold2_pipeline.db import CampaignStore, connect_database
from esmfold2_pipeline.execution.mock_worker import (
    MOCK_CRITIC_NAME,
    MOCK_METRICS,
    MOCK_SEQUENCE,
    plan_one_mock_shard,
    run_one_mock_shard,
)


class MockVerticalSliceTest(unittest.TestCase):
    def test_one_mock_shard_claim_write_and_commit_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)

            shard_id = plan_one_mock_shard(campaign_dir)
            result = run_one_mock_shard(campaign_dir, worker_id="worker-test", gpu_id="0")

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.shard_id, shard_id)
            self.assertEqual(result.critic_name, MOCK_CRITIC_NAME)

            self.assertIsNone(result.sequence_path)
            structure_path = campaign_dir / result.structure_path
            self.assertTrue(structure_path.exists())
            self.assertEqual(result.structure_path, "esmfold2/structures/s000_seed000_c000.pdb")
            self.assertIn("MOCK ESMFOLD2 COMPLEX", structure_path.read_text())

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)

            shard = store.fetch_one(
                "SELECT * FROM shards WHERE shard_id = ?",
                (result.shard_id,),
            )
            self.assertEqual(shard["status"], "completed")
            self.assertEqual(shard["attempt_count"], 1)
            self.assertEqual(shard["claim_worker_id"], "worker-test")
            self.assertEqual(shard["claim_gpu_id"], "0")

            candidate = store.fetch_one(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (result.candidate_id,),
            )
            self.assertEqual(candidate["status"], "completed")
            self.assertEqual(candidate["designed_sequence"], MOCK_SEQUENCE)
            self.assertIsNone(candidate["sequence_path"])

            critic = store.fetch_one(
                """
                SELECT *
                FROM critic_metrics
                WHERE candidate_id = ? AND critic_name = ?
                """,
                (result.candidate_id, MOCK_CRITIC_NAME),
            )
            self.assertEqual(critic["status"], "completed")
            self.assertEqual(critic["structure_path"], result.structure_path)
            self.assertEqual(critic["iptm"], MOCK_METRICS["iptm"])
            self.assertEqual(
                critic["distogram_iptm_proxy"],
                MOCK_METRICS["distogram_iptm_proxy"],
            )

            attempt = store.fetch_one(
                "SELECT * FROM attempts WHERE shard_id = ?",
                (result.shard_id,),
            )
            self.assertEqual(attempt["status"], "completed")
            self.assertEqual(attempt["worker_id"], "worker-test")
            self.assertEqual(attempt["exit_code"], 0)
            conn.close()

            second_result = run_one_mock_shard(campaign_dir, worker_id="worker-test-2")
            self.assertIsNone(second_result)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            counts = {
                row["name"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT 'shards' AS name, COUNT(*) AS count FROM shards
                    UNION ALL
                    SELECT 'candidates', COUNT(*) FROM candidates
                    UNION ALL
                    SELECT 'critic_metrics', COUNT(*) FROM critic_metrics
                    UNION ALL
                    SELECT 'attempts', COUNT(*) FROM attempts
                    """
                )
            }
            conn.close()
            self.assertEqual(
                counts,
                {
                    "shards": 1,
                    "candidates": 1,
                    "critic_metrics": 1,
                    "attempts": 1,
                },
            )


if __name__ == "__main__":
    unittest.main()
