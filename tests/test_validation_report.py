from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from esmfold2_pipeline.execution.mock_worker import (
    plan_one_mock_shard,
    run_one_mock_shard,
)
from esmfold2_pipeline.reports import analyze_campaign, report_validation
from esmfold2_pipeline.validation import (
    MOCK_VALIDATION_MODEL,
    ValidationPlanConfig,
    plan_validation_tasks,
    run_mock_validation,
)


class ValidationReportTest(unittest.TestCase):
    def test_report_validation_writes_task_structure_and_summary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)
            plan_one_mock_shard(campaign_dir)
            run_one_mock_shard(campaign_dir)
            plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(model_name=MOCK_VALIDATION_MODEL),
            )
            run_mock_validation(campaign_dir)

            result = report_validation(campaign_dir)

            self.assertEqual(result.task_rows, 1)
            self.assertEqual(result.structure_rows, 2)
            self.assertEqual(
                result.validated_dir,
                campaign_dir / "validation" / "mock_protenix_v2",
            )
            self.assertTrue(result.manifest_csv.exists())
            self.assertTrue(result.structures_manifest_csv.exists())
            self.assertTrue(result.summary_json.exists())

            with result.manifest_csv.open() as handle:
                task_rows = list(csv.DictReader(handle))
            self.assertEqual(len(task_rows), 1)
            task = task_rows[0]
            self.assertEqual(task["validator_model"], MOCK_VALIDATION_MODEL)
            self.assertEqual(task["validator_iptm"], "0.8")
            self.assertEqual(task["validator_ipsae"], "0.6")
            self.assertEqual(task["validator_metric_scope"], "binder_target")
            self.assertEqual(task["binder_length"], "20")
            self.assertIn("esm_hotspot_distance_angstrom", task)
            self.assertIn("pose_agreement_error", task)
            self.assertEqual(
                task["source_structure_path"],
                "esmfold2/structures/s000_seed000_c000.pdb",
            )
            self.assertTrue(
                task["validated_structure_path"].startswith(
                    "validation/mock_protenix_v2/structures/passing/"
                )
            )
            self.assertIn("binder_ca_rmsd_after_target_alignment", task)

            with result.structures_manifest_csv.open() as handle:
                structure_rows = list(csv.DictReader(handle))
            self.assertEqual(len(structure_rows), 2)
            self.assertEqual(
                [row["structure_status"] for row in structure_rows],
                ["passing", "rejected"],
            )
            self.assertEqual(
                [row["validator_ipsae"] for row in structure_rows],
                ["0.6", "0.12"],
            )
            for row in structure_rows:
                self.assertTrue((campaign_dir / row["structure_path"]).exists())

            summary = json.loads(result.summary_json.read_text())
            self.assertEqual(summary["counts"]["completed_count"], 1)
            self.assertEqual(summary["counts"]["passing_structure_count"], 1)
            self.assertEqual(summary["counts"]["rejected_structure_count"], 1)
            self.assertEqual(
                summary["validation"]["models"],
                {MOCK_VALIDATION_MODEL: 1},
            )
            self.assertEqual(summary["issues"], [])
            self.assertEqual(
                summary["top_validation_rows"][0]["validator_ipsae"],
                0.6,
            )
            self.assertEqual(summary["pose_agreement"]["rows"], 2)
            self.assertEqual(summary["pose_agreement"]["rows_with_binder_ca_rmsd"], 0)
            self.assertTrue(summary["pose_agreement"]["errors"])

    def test_report_validation_writes_one_report_per_validator_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)
            plan_one_mock_shard(campaign_dir)
            run_one_mock_shard(campaign_dir)
            plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(model_name="mock-protenix-v1"),
            )
            plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(model_name="mock-protenix-v2"),
            )
            run_mock_validation(campaign_dir)

            result = report_validation(campaign_dir)

            self.assertEqual(result.task_rows, 2)
            self.assertEqual(result.structure_rows, 4)
            self.assertEqual(
                [report.model_name for report in result.model_reports],
                ["mock-protenix-v1", "mock-protenix-v2"],
            )
            for report in result.model_reports:
                self.assertEqual(report.task_rows, 1)
                self.assertEqual(report.structure_rows, 2)
                self.assertTrue(report.manifest_csv.exists())
                self.assertTrue(report.structures_manifest_csv.exists())
                summary = json.loads(report.summary_json.read_text())
                self.assertEqual(summary["counts"]["planned_validation_count"], 1)
                self.assertEqual(summary["counts"]["completed_count"], 1)
                self.assertEqual(summary["counts"]["passing_structure_count"], 1)
                self.assertEqual(summary["counts"]["rejected_structure_count"], 1)
                self.assertEqual(
                    summary["validation"]["models"],
                    {report.model_name: 1},
                )

    def test_analyze_campaign_ranks_all_designs_and_copies_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)
            plan_one_mock_shard(campaign_dir)
            run_one_mock_shard(campaign_dir)
            plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(model_name=MOCK_VALIDATION_MODEL),
            )
            run_mock_validation(campaign_dir)
            long_candidate_id = "target_" + ("long_candidate_" * 8) + "seed0"
            with sqlite3.connect(campaign_dir / "campaign.sqlite") as conn:
                conn.execute("PRAGMA foreign_keys = OFF")
                for table in (
                    "candidates",
                    "critic_metrics",
                    "validation_tasks",
                    "validation_structures",
                    "attempts",
                    "validation_msa_job_candidates",
                ):
                    update = (
                        f"UPDATE {table} "
                        "SET candidate_id = ? "
                        "WHERE candidate_id IS NOT NULL"
                    )
                    conn.execute(update, (long_candidate_id,))
                conn.commit()
                conn.execute("PRAGMA foreign_keys = ON")

            result = analyze_campaign(campaign_dir, top_k=1)

            self.assertEqual(result.ranked_count, 1)
            self.assertEqual(result.copied_designs, 1)
            with result.combined_ranking_csv.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["analysis_rank"], "1")
            self.assertEqual(rows[0]["candidate_id"], long_candidate_id)
            self.assertEqual(rows[0]["validator_model"], MOCK_VALIDATION_MODEL)
            self.assertEqual(rows[0]["binder_length"], "20")
            self.assertIn("binder_ca_rmsd_after_target_alignment", rows[0])
            self.assertTrue(
                (
                    result.plots_dir
                    / "esmfold2_iptm_vs_mock_protenix_v2_iptm_colored_by_rmsd.png"
                ).exists()
            )
            self.assertTrue(
                (
                    result.plots_dir
                    / "esmfold2_iptm_vs_mock_protenix_v2_ipsae_colored_by_rmsd.png"
                ).exists()
            )
            esm_structures = sorted(
                (result.top_ranked_dir / "esmfold2").glob("rank0001_*_esmfold2.pdb")
            )
            validator_structures = sorted(
                (result.top_ranked_dir / "mock_protenix_v2").glob(
                    "rank0001_*_mock_protenix_v2.cif"
                )
            )
            self.assertEqual(len(esm_structures), 1)
            self.assertEqual(len(validator_structures), 1)
            self.assertIn(long_candidate_id, esm_structures[0].name)
            self.assertIn(long_candidate_id, validator_structures[0].name)
            self.assertEqual(
                rows[0]["copied_esmfold2_structure"],
                esm_structures[0].relative_to(campaign_dir).as_posix(),
            )
            self.assertEqual(
                rows[0]["copied_validator_structure"],
                validator_structures[0].relative_to(campaign_dir).as_posix(),
            )
            self.assertFalse(any(result.top_ranked_dir.glob("**/metadata.json")))
            summary = json.loads(result.summary_json.read_text())
            self.assertEqual(summary["counts"]["ranked_designs"], 1)
            self.assertEqual(summary["counts"]["top_k"], 1)
            self.assertEqual(summary["counts"]["copied_designs"], 1)
            self.assertEqual(summary["pose_agreement"]["rows"], 1)
            self.assertTrue(
                any(
                    warning.startswith("skipped plot mock_protenix_v2_iptm_vs_binder_rmsd")
                    for warning in summary["warnings"]
                )
            )


if __name__ == "__main__":
    unittest.main()
