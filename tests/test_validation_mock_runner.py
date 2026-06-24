from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from esmfold2_pipeline.db import CampaignStore, connect_database
from esmfold2_pipeline.execution.mock_worker import (
    plan_one_mock_shard,
    run_one_mock_shard,
)
from esmfold2_pipeline.reports import inspect_campaign
from esmfold2_pipeline.validation import (
    MOCK_VALIDATION_MODEL,
    ValidationPlanConfig,
    plan_validation_tasks,
    run_mock_validation,
)


class MockValidationRunnerTest(unittest.TestCase):
    def test_mock_validation_promotes_all_cifs_before_task_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)
            plan_one_mock_shard(campaign_dir)
            design_result = run_one_mock_shard(campaign_dir)
            assert design_result is not None

            plan = plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(model_name=MOCK_VALIDATION_MODEL),
            )
            self.assertEqual(plan.created_count, 1)

            result = run_mock_validation(
                campaign_dir,
                worker_id="mock-validator",
                gpu_id="0",
                max_tasks=1,
            )
            self.assertEqual(result.completed_tasks, 1)
            self.assertEqual(result.recorded_structures, 2)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                task = conn.execute(
                    "SELECT * FROM validation_tasks"
                ).fetchone()
                self.assertEqual(task["status"], "completed")
                self.assertEqual(task["attempt_count"], 1)
                self.assertEqual(task["claim_worker_id"], "mock-validator")
                self.assertEqual(task["claim_gpu_id"], "0")
                self.assertEqual(task["iptm"], 0.8)
                self.assertEqual(task["ipsae"], 0.6)
                self.assertTrue(task["output_structure_path"].startswith(
                    "validation/mock_protenix_v2/structures/passing/"
                ))
                self.assertTrue((campaign_dir / task["output_structure_path"]).exists())

                attempts = conn.execute(
                    """
                    SELECT status, stage, exit_code
                    FROM attempts
                    WHERE stage = 'validation'
                    """
                ).fetchall()
                self.assertEqual(len(attempts), 1)
                self.assertEqual(attempts[0]["status"], "completed")
                self.assertEqual(attempts[0]["exit_code"], 0)

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
                    [0.8, 0.21],
                )
                self.assertEqual(
                    [row["scoped_ipsae"] for row in structures],
                    [0.6, 0.12],
                )
                for row in structures:
                    self.assertTrue((campaign_dir / row["structure_path"]).exists())
                    self.assertNotIn("/.staging/", row["structure_path"])
            finally:
                conn.close()

            status = inspect_campaign(campaign_dir)
            self.assertEqual(status.validation_status_counts, {"completed": 1})
            self.assertEqual(
                status.validation_structure_status_counts,
                {"passing": 1, "rejected": 1},
            )
            self.assertEqual(status.issues, [])

            second = run_mock_validation(campaign_dir)
            self.assertEqual(second.completed_tasks, 0)
            self.assertEqual(second.recorded_structures, 0)
            self.assertTrue(second.skipped_no_pending)

    def test_mock_validation_adds_config_suffix_for_same_candidate_model_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)
            plan_one_mock_shard(campaign_dir)
            design_result = run_one_mock_shard(campaign_dir)
            assert design_result is not None

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            try:
                for suffix in ("a", "b"):
                    store.create_validation_task(
                        validation_id=f"val_{suffix}",
                        candidate_id=design_result.candidate_id,
                        model_name=MOCK_VALIDATION_MODEL,
                        validation_config_hash=f"hash_{suffix}",
                        selection_rank=1,
                    )
            finally:
                conn.close()

            result = run_mock_validation(campaign_dir)
            self.assertEqual(result.completed_tasks, 2)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                paths = [
                    row["structure_path"]
                    for row in conn.execute(
                        """
                        SELECT structure_path
                        FROM validation_structures
                        WHERE sample_rank = 0
                        ORDER BY validation_id
                        """
                    )
                ]
            finally:
                conn.close()

            self.assertEqual(len(paths), 2)
            self.assertIn("__cfg-hash_a__sample00.cif", paths[0])
            self.assertIn("__cfg-hash_b__sample00.cif", paths[1])
            self.assertEqual(len(set(paths)), 2)

    def test_status_reports_missing_validation_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)
            plan_one_mock_shard(campaign_dir)
            run_one_mock_shard(campaign_dir)
            plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(model_name=MOCK_VALIDATION_MODEL),
            )
            run_mock_validation(campaign_dir)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            try:
                row = conn.execute(
                    """
                    SELECT structure_path
                    FROM validation_structures
                    WHERE status = 'rejected'
                    """
                ).fetchone()
            finally:
                conn.close()
            (campaign_dir / row["structure_path"]).unlink()

            status = inspect_campaign(campaign_dir)
            self.assertEqual(status.missing_artifact_count, 1)
            self.assertEqual(status.issues[0].kind, "missing_validation_artifact")
            self.assertEqual(status.issues[0].table, "validation_structures")


if __name__ == "__main__":
    unittest.main()
