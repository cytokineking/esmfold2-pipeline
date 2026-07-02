from __future__ import annotations

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
from esmfold2_pipeline.validation import ValidationPlanConfig, plan_validation_tasks


class ValidationPlanningTest(unittest.TestCase):
    def test_validation_config_hash_includes_runtime_msa_and_filter_settings(self) -> None:
        base = ValidationPlanConfig(
            model_name="protenix-v2",
            top_k=2,
            min_esm_iptm=0.6,
        )

        self.assertEqual(
            base.validation_config_hash,
            ValidationPlanConfig(
                model_name="protenix-v2",
                top_k=3,
                min_esm_iptm=0.6,
            ).validation_config_hash,
        )
        self.assertNotEqual(
            base.validation_config_hash,
            ValidationPlanConfig(
                model_name="protenix-v2",
                top_k=2,
                min_esm_iptm=0.7,
            ).validation_config_hash,
        )
        self.assertNotEqual(
            base.validation_config_hash,
            ValidationPlanConfig(
                model_name="protenix-v2",
                top_k=2,
                min_esm_iptm=0.6,
                seeds=(101, 202),
            ).validation_config_hash,
        )
        self.assertNotEqual(
            base.validation_config_hash,
            ValidationPlanConfig(
                model_name="protenix-v2",
                top_k=2,
                min_esm_iptm=0.6,
                n_sample=2,
            ).validation_config_hash,
        )
        self.assertNotEqual(
            base.validation_config_hash,
            ValidationPlanConfig(
                model_name="protenix-v2",
                top_k=2,
                min_esm_iptm=0.6,
                target_msa_mode="server",
                msa_server_url="https://msa.example/",
            ).validation_config_hash,
        )
        self.assertNotEqual(
            base.validation_config_hash,
            ValidationPlanConfig(
                model_name="protenix-v2",
                top_k=2,
                min_esm_iptm=0.6,
                binder_msa_mode="single_sequence",
                use_msa=True,
            ).validation_config_hash,
        )
        self.assertNotEqual(
            base.validation_config_hash,
            ValidationPlanConfig(
                model_name="protenix-v2",
                top_k=2,
                min_esm_iptm=0.6,
                validation_hotspot_cutoff_angstrom=4.0,
            ).validation_config_hash,
        )
        self.assertNotEqual(
            base.validation_config_hash,
            ValidationPlanConfig(
                model_name="protenix-v2",
                top_k=2,
                min_esm_iptm=0.6,
                use_template="false",
            ).validation_config_hash,
        )
        self.assertNotEqual(
            base.validation_config_hash,
            ValidationPlanConfig(
                model_name="protenix-v2",
                top_k=2,
                min_esm_iptm=0.6,
                ipsae_pae_cutoff=12.0,
            ).validation_config_hash,
        )

    def test_validate_plan_filters_deduplicates_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = _run_fake_campaign(Path(tmpdir), num_designs=4)

            result = plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(
                    model_name="protenix-v2",
                    top_k=2,
                    min_esm_iptm=0.6,
                ),
            )

            self.assertEqual(result.candidate_count, 3)
            self.assertEqual(result.selected_count, 2)
            self.assertEqual(result.created_count, 2)
            self.assertEqual(result.existing_count, 0)

            rows = _validation_rows(campaign_dir)
            self.assertEqual(
                [(row["selection_rank"], row["candidate_id"]) for row in rows],
                [
                    (1, "ctla4_mp_seed3"),
                    (2, "ctla4_mp_seed2"),
                ],
            )
            self.assertEqual(
                {row["validation_config_hash"] for row in rows},
                {result.validation_config_hash},
            )
            self.assertNotEqual(
                result.validation_config_hash,
                ValidationPlanConfig(
                    model_name="protenix-v2",
                    top_k=2,
                    min_esm_iptm=0.6,
                    min_validation_ipsae=0.6,
                ).validation_config_hash,
            )

            second = plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(
                    model_name="protenix-v2",
                    top_k=2,
                    min_esm_iptm=0.6,
                ),
            )
            self.assertEqual(second.created_count, 0)
            self.assertEqual(second.existing_count, 2)

            expanded = plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(
                    model_name="protenix-v2",
                    top_k=3,
                    min_esm_iptm=0.6,
                ),
            )
            self.assertEqual(expanded.created_count, 1)
            self.assertEqual(expanded.existing_count, 2)
            expanded_rows = _validation_rows(campaign_dir)
            self.assertEqual(
                [(row["selection_rank"], row["candidate_id"]) for row in expanded_rows],
                [
                    (1, "ctla4_mp_seed3"),
                    (2, "ctla4_mp_seed2"),
                    (3, "ctla4_mp_seed1"),
                ],
            )

    def test_changed_validation_runtime_config_creates_distinct_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = _run_fake_campaign(Path(tmpdir), num_designs=1)

            first = plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(n_sample=1),
            )
            second = plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(n_sample=2),
            )

            self.assertEqual(first.created_count, 1)
            self.assertEqual(second.created_count, 1)
            rows = _validation_rows(campaign_dir)
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                len({row["validation_config_hash"] for row in rows}),
                2,
            )

    def test_validation_claims_recover_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = _run_fake_campaign(Path(tmpdir), num_designs=3)
            plan_validation_tasks(campaign_dir)

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            claims = store.claim_next_pending_validation_tasks(
                worker_id="validation-worker",
                batch_size=2,
                hostname="localhost",
                pid=1234,
                gpu_id="0",
            )
            self.assertEqual(len(claims), 2)
            self.assertEqual({claim.selection_rank for claim in claims}, {1, 2})

            recovered = store.recover_stale_validation_tasks(
                stale_before="9999-01-01T00:00:00.000Z",
                error_message="validation stale timeout",
            )
            self.assertEqual(recovered, 2)

            rows = _validation_rows(campaign_dir)
            self.assertEqual({row["status"] for row in rows}, {"pending"})
            self.assertEqual({row["attempt_count"] for row in rows}, {1})

            attempts = conn.execute(
                """
                SELECT stage, status
                FROM attempts
                WHERE stage = 'validation'
                ORDER BY attempt_id
                """
            ).fetchall()
            self.assertEqual(
                [(row["stage"], row["status"]) for row in attempts],
                [("validation", "stale"), ("validation", "stale")],
            )
            conn.close()

    def test_protenix_validation_rejects_scfv_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = _run_fake_campaign(
                Path(tmpdir),
                num_designs=1,
                binder_scaffold="scfv",
            )

            with self.assertRaisesRegex(
                ValueError,
                "requires a bundled scFv framework structural template",
            ):
                plan_validation_tasks(
                    campaign_dir,
                    config=ValidationPlanConfig(model_name="protenix-v2"),
                )

            self.assertEqual(_validation_rows(campaign_dir), [])

    def test_protenix_validation_allows_builtin_scfv_template_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = _run_fake_campaign(
                Path(tmpdir),
                num_designs=1,
                binder_scaffold="scfv",
                framework="anifrolumab_framework_vhvl",
                framework_source="builtin",
            )

            result = plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(model_name="protenix-v2"),
            )

            self.assertEqual(result.created_count, 1)
            self.assertEqual(len(_validation_rows(campaign_dir)), 1)

            with self.assertRaisesRegex(
                ValueError,
                "requires a bundled scFv framework structural template",
            ):
                plan_validation_tasks(
                    campaign_dir,
                    config=ValidationPlanConfig(
                        model_name="protenix-v2",
                        use_template="false",
                    ),
                )

            with self.assertRaisesRegex(
                ValueError,
                "scFv binder MSA support is not implemented",
            ):
                plan_validation_tasks(
                    campaign_dir,
                    config=ValidationPlanConfig(
                        model_name="protenix-v2",
                        use_msa=True,
                    ),
                )

            target_msa_only = plan_validation_tasks(
                campaign_dir,
                config=ValidationPlanConfig(
                    model_name="protenix-v2",
                    use_msa=True,
                    binder_msa_mode="none",
                ),
            )
            self.assertEqual(target_msa_only.created_count, 1)


def _run_fake_campaign(
    root: Path,
    *,
    num_designs: int,
    binder_scaffold: str = "miniprotein",
    framework: str | None = None,
    framework_source: str | None = None,
) -> Path:
    campaign_dir = root / "campaign"
    config_path = root / "config.yaml"
    config_path.write_text(
        f"""
target:
  name: ctla4
binder:
  scaffold: miniprotein
campaign:
  num_designs: {num_designs}
  critics:
    - ESMFold2-Experimental-Fast
  steps: 1
output: {campaign_dir}
""".lstrip()
    )
    plan_campaign(config_path)
    with patch(
        "esmfold2_pipeline.execution.local.run_binder_design_artifact",
        side_effect=lambda **kwargs: _fake_design_artifact(
            **{
                **kwargs,
                "binder_scaffold": binder_scaffold,
                "framework": framework,
                "framework_source": framework_source,
            }
        ),
    ):
        run_campaign(campaign_dir, worker_id="validation-plan-test", gpu_id="0")
    return campaign_dir


def _fake_design_artifact(**kwargs) -> DesignCandidateArtifact:
    root = Path(kwargs["campaign_dir"])
    seed = int(kwargs["seed"])
    cid = kwargs["candidate_id"]
    structure_path = structure_relpath(kwargs.get("artifact_stem", cid))
    write_text_atomic(
        root / structure_path,
        f"HEADER    FAKE VALIDATION SOURCE {seed}\nEND\n",
    )
    sequence = (
        "DUPLICATESEQ"
        if seed in {0, 1}
        else f"UNIQUESEQ{seed}"
    )
    return DesignCandidateArtifact(
        candidate_id=cid,
        designed_sequence=sequence,
        sequence_path=None,
        critic_name=kwargs["critic_name"],
        structure_path=structure_path.as_posix(),
        design_metrics={
            "target_name": "ctla4",
            "binder_scaffold": kwargs.get("binder_scaffold", "miniprotein"),
            "binder_type": kwargs.get("binder_scaffold", "miniprotein"),
            "framework": kwargs.get("framework"),
            "framework_name": kwargs.get("framework"),
            "framework_source": kwargs.get("framework_source"),
            "binder_chain_id": "B",
            "final_loss": 1.0 - (0.1 * seed),
        },
        critic_metrics={
            "iptm": 0.5 + (0.1 * seed),
            "iptm_scope": "binder_target",
            "complex_iptm": 0.7 + (0.05 * seed),
            "ptm": 0.4 + (0.05 * seed),
            "plddt_complex": 80.0 + seed,
            "plddt_binder": 70.0 + seed,
            "plddt_target": 90.0 + seed,
            "distogram_iptm_proxy": 0.45 + (0.1 * seed),
        },
    )


def _validation_rows(campaign_dir: Path):
    conn = connect_database(campaign_dir / "campaign.sqlite")
    try:
        return conn.execute(
            """
            SELECT *
            FROM validation_tasks
            ORDER BY selection_rank
            """
        ).fetchall()
    finally:
        conn.close()


if __name__ == "__main__":
    unittest.main()
