from __future__ import annotations

import unittest

import numpy as np

from esmfold2_pipeline.validation.logical_scoring import (
    LOGICAL_IPTM_AGGREGATION,
    esmfold2_pae_bin_centers,
    ipsae_normalization,
    logical_cross_role_tm_score,
    logical_iptm_from_pae_logits,
    logical_ipsae_from_pae,
    logical_role_masks,
)
from esmfold2_pipeline.design.loop import StepResult, run_design_loop
from esmfold2_pipeline.validation.protenix import (
    _logical_iptm_metric_from_full_data,
    _validation_metrics_from_summary,
)


class LogicalScoringTests(unittest.TestCase):
    def test_esmfold2_bins_match_native_definition_for_nondefault_bin_count(self) -> None:
        bins = esmfold2_pae_bin_centers(8)

        np.testing.assert_allclose(
            bins,
            np.array([2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0]),
        )

    def test_esmfold2_torch_reduction_matches_native_reference(self) -> None:
        import torch

        torch.manual_seed(11)
        valid = torch.tensor([[True, True, True, True], [True, True, True, False]])
        binder = torch.tensor([True, True, False, False])
        target = ~binder
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.randn(2, 4, 4, 8, dtype=dtype)
                actual = logical_iptm_from_pae_logits(
                    logits,
                    binder_mask=binder,
                    target_mask=target,
                    valid_token_mask=valid,
                )

                bin_width = 32.0 / logits.shape[-1]
                bins = torch.arange(0.5 * bin_width, 32.0, bin_width)
                mask_f = valid.float()
                d0 = 1.24 * (mask_f.sum(dim=-1, keepdim=True).clamp(min=19) - 15) ** (1 / 3) - 1.8
                tm_per_bin = 1 / (1 + (bins / d0) ** 2)
                expected_tm = (torch.softmax(logits, dim=-1) * tm_per_bin[:, None, None, :]).sum(-1)
                cross_role = ((binder[:, None] & target[None, :]) | (target[:, None] & binder[None, :])).float()
                scoring_mask = cross_role[None, ...] * mask_f[:, :, None] * mask_f[:, None, :]
                reference = ((expected_tm * scoring_mask).sum(-1) / (scoring_mask.sum(-1) + 1e-5)).max(-1).values

                self.assertTrue(torch.allclose(actual, reference, atol=1e-7, rtol=1e-6))

    def test_esmfold2_rejects_invalid_roles_but_allows_unassigned_padding(self) -> None:
        import torch

        logits = torch.zeros(1, 3, 3, 8)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            logical_iptm_from_pae_logits(
                logits,
                binder_mask=[True, True, False],
                target_mask=[True, False, True],
                valid_token_mask=[True, True, False],
            )
        with self.assertRaisesRegex(ValueError, "every valid token"):
            logical_iptm_from_pae_logits(
                logits,
                binder_mask=[True, False, False],
                target_mask=[False, True, False],
                valid_token_mask=[True, True, True],
            )

        score = logical_iptm_from_pae_logits(
            logits,
            binder_mask=[True, False, False],
            target_mask=[False, True, False],
            valid_token_mask=[True, True, False],
        )
        self.assertEqual(score.shape, (1,))

    def test_native_cross_role_row_reduction_ignores_same_role_nonfinite_values(self) -> None:
        expected_tm = np.array(
            [
                [np.nan, np.inf, 0.5, 0.7],
                [-np.inf, np.nan, 0.2, 0.2],
                [0.1, 0.3, np.nan, np.inf],
                [0.8, 0.9, -np.inf, np.nan],
            ]
        )

        score = logical_cross_role_tm_score(
            expected_tm,
            binder_mask=[True, True, False, False],
            target_mask=[False, False, True, True],
        )

        self.assertAlmostEqual(score, 0.85)

    def test_missing_frame_excludes_only_source_row(self) -> None:
        expected_tm = np.array(
            [
                [0.0, 0.0, 0.5, 0.7],
                [0.0, 0.0, 0.2, 0.2],
                [0.1, 0.3, 0.0, 0.0],
                [0.8, 0.9, 0.0, 0.0],
            ]
        )

        score = logical_cross_role_tm_score(
            expected_tm,
            binder_mask=[True, True, False, False],
            target_mask=[False, False, True, True],
            source_valid_mask=[True, True, True, False],
        )

        self.assertAlmostEqual(score, 0.6)

    def test_split_reorder_and_rename_physical_chains_are_invariant(self) -> None:
        expected_tm = np.array(
            [
                [0.0, 0.0, 0.6, 0.8],
                [0.0, 0.0, 0.4, 0.2],
                [0.3, 0.7, 0.0, 0.0],
                [0.9, 0.5, 0.0, 0.0],
            ]
        )
        binder, target = logical_role_masks(
            [10, 11, 20, 21],
            asym_id_to_chain_id={10: "binder-left", 11: "binder-right", 20: "target-x", 21: "target-y"},
            chain_role_map={
                "binder": ["binder-left", "binder-right"],
                "target": ["target-x", "target-y"],
            },
        )
        baseline = logical_cross_role_tm_score(
            expected_tm,
            binder_mask=binder,
            target_mask=target,
        )

        permutation = np.array([2, 0, 3, 1])
        permuted = expected_tm[np.ix_(permutation, permutation)]
        renamed_binder, renamed_target = logical_role_masks(
            [7, 3, 8, 4],
            asym_id_to_chain_id={7: "renamed-target-a", 3: "renamed-binder-a", 8: "renamed-target-b", 4: "renamed-binder-b"},
            chain_role_map={
                "binder": ["renamed-binder-a", "renamed-binder-b"],
                "target": ["renamed-target-a", "renamed-target-b"],
            },
        )
        reordered = logical_cross_role_tm_score(
            permuted,
            binder_mask=renamed_binder,
            target_mask=renamed_target,
        )
        merged_binder, merged_target = logical_role_masks(
            [10, 10, 20, 20],
            asym_id_to_chain_id={10: "binder-merged", 20: "target-merged"},
            chain_role_map={
                "binder": ["binder-merged"],
                "target": ["target-merged"],
            },
        )
        merged = logical_cross_role_tm_score(
            expected_tm,
            binder_mask=merged_binder,
            target_mask=merged_target,
        )

        self.assertAlmostEqual(baseline, 0.7)
        self.assertAlmostEqual(reordered, baseline)
        self.assertAlmostEqual(merged, baseline)

    def test_ipsae_matches_native_per_row_normalization_and_directional_max(self) -> None:
        pae = np.array(
            [
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 20.0, 20.0],
                [4.0, 2.0, 0.0, 0.0],
                [20.0, 20.0, 0.0, 0.0],
            ]
        )

        metric = logical_ipsae_from_pae(
            pae,
            binder_mask=[True, True, False, False],
            target_mask=[False, False, True, True],
            pae_cutoff=15.0,
        )

        d0 = ipsae_normalization(2)
        binder_to_target = np.mean(1.0 / (1.0 + (np.array([1.0, 3.0]) / d0) ** 2))
        target_to_binder = np.mean(1.0 / (1.0 + (np.array([4.0, 2.0]) / d0) ** 2))
        self.assertAlmostEqual(metric["value"], max(binder_to_target, target_to_binder))
        self.assertEqual(metric["best_direction"], "binder_to_target")

    def test_ipsae_rejects_nonfinite_cross_role_pae(self) -> None:
        pae = np.array([[0.0, np.nan], [2.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "must be finite"):
            logical_ipsae_from_pae(
                pae,
                binder_mask=[True, False],
                target_mask=[False, True],
                pae_cutoff=15.0,
            )

    def test_ipsae_is_invariant_to_physical_chain_split_merge_and_rename(self) -> None:
        pae = np.array(
            [
                [0.0, 20.0, 1.0, 3.0],
                [20.0, 0.0, 2.0, 4.0],
                [5.0, 2.0, 0.0, 20.0],
                [3.0, 6.0, 20.0, 0.0],
            ]
        )
        split_binder, split_target = logical_role_masks(
            [1, 2, 3, 4],
            asym_id_to_chain_id={1: "b1", 2: "b2", 3: "t1", 4: "t2"},
            chain_role_map={"binder": ["b1", "b2"], "target": ["t1", "t2"]},
        )
        merged_binder, merged_target = logical_role_masks(
            [7, 7, 9, 9],
            asym_id_to_chain_id={7: "renamed-binder", 9: "renamed-target"},
            chain_role_map={
                "binder": ["renamed-binder"],
                "target": ["renamed-target"],
            },
        )

        split = logical_ipsae_from_pae(
            pae,
            binder_mask=split_binder,
            target_mask=split_target,
            pae_cutoff=15.0,
        )
        merged = logical_ipsae_from_pae(
            pae,
            binder_mask=merged_binder,
            target_mask=merged_target,
            pae_cutoff=15.0,
        )

        self.assertAlmostEqual(split["value"], merged["value"])
        self.assertEqual(split["best_direction"], merged["best_direction"])

    def test_protenix_native_matrix_and_frame_mask_drive_threshold_decision(self) -> None:
        full_data = {
            "token_pair_tm_expected": [
                [0.0, 0.0, 0.8, 0.8],
                [0.0, 0.0, 0.1, 0.1],
                [0.2, 0.2, 0.0, 0.0],
                [0.9, 0.9, 0.0, 0.0],
            ],
            "token_pair_tm_normalization_count": 4,
            "token_asym_id": [0, 0, 1, 1],
            "token_has_frame": [True, True, True, False],
        }
        iptm, details = _logical_iptm_metric_from_full_data(
            full_data,
            chain_role_map={"binder": ["A"], "target": ["B"]},
        )
        assert iptm is not None
        self.assertEqual(details, {})
        self.assertAlmostEqual(iptm["value"], 0.8)

        metrics = _validation_metrics_from_summary(
            {
                "iptm": 0.99,
                "chain_pair_iptm": [[0.0, 0.2], [0.2, 0.0]],
            },
            chain_role_map={"binder": ["A"], "target": ["B"]},
            min_validation_iptm=0.7,
            min_validation_ipsae=None,
            iptm=iptm,
        )
        self.assertTrue(metrics["validation_passed"])
        self.assertAlmostEqual(metrics["validation_iptm"], 0.8)
        self.assertAlmostEqual(metrics["validation_iptm_pair_min"], 0.2)
        self.assertEqual(metrics["validation_iptm_aggregation"], LOGICAL_IPTM_AGGREGATION)

    def test_protenix_all_missing_source_frames_matches_native_zero(self) -> None:
        iptm, details = _logical_iptm_metric_from_full_data(
            {
                "token_pair_tm_expected": [[0.0, 0.9], [0.8, 0.0]],
                "token_pair_tm_normalization_count": 2,
                "token_asym_id": [0, 1],
                "token_has_frame": [False, False],
            },
            chain_role_map={"binder": ["A"], "target": ["B"]},
        )

        self.assertEqual(details, {})
        assert iptm is not None
        self.assertEqual(iptm["value"], 0.0)

    def test_old_protenix_output_is_unavailable_not_pair_approximated(self) -> None:
        iptm, details = _logical_iptm_metric_from_full_data(
            {
                "token_pair_pae": [[0.0, 1.0], [1.0, 0.0]],
                "token_asym_id": [0, 1],
                "token_has_frame": [True, True],
            },
            chain_role_map={"binder": ["A"], "target": ["B"]},
        )

        self.assertIsNone(iptm)
        self.assertIn("missing token_pair_tm_expected", details["validation_iptm_error"])

    def test_generation_selection_reverses_when_old_score_is_target_target_dominated(self) -> None:
        import torch

        binder = torch.tensor([False, False, False, True])
        target = ~binder
        valid = torch.ones(1, 4, dtype=torch.bool)

        def candidate_logits(
            *,
            target_to_binder_bin: int,
            binder_to_target_bin: int,
            target_pair_bin: int,
        ) -> torch.Tensor:
            logits = torch.full((1, 4, 4, 8), -20.0)
            logits[..., 7] = 20.0
            for row in range(4):
                for col in range(4):
                    if binder[row] != binder[col]:
                        logits[0, row, col, :] = -20.0
                        selected_bin = (
                            target_to_binder_bin if target[row] else binder_to_target_bin
                        )
                        logits[0, row, col, selected_bin] = 20.0
                    elif target[row] and target[col] and row != col:
                        logits[0, row, col, :] = -20.0
                        logits[0, row, col, target_pair_bin] = 20.0
            return logits

        candidate_a = candidate_logits(
            target_to_binder_bin=0,
            binder_to_target_bin=7,
            target_pair_bin=7,
        )
        candidate_b = candidate_logits(
            target_to_binder_bin=7,
            binder_to_target_bin=7,
            target_pair_bin=0,
        )
        corrected = [
            float(logical_iptm_from_pae_logits(item, binder_mask=binder, target_mask=target, valid_token_mask=valid)[0])
            for item in (candidate_a, candidate_b)
        ]

        def old_physical_chain_score(logits: torch.Tensor) -> float:
            bin_width = 32.0 / logits.shape[-1]
            bins = torch.arange(0.5 * bin_width, 32.0, bin_width)
            d0 = 1.24 * (19 - 15) ** (1 / 3) - 1.8
            expected = (torch.softmax(logits.float(), -1) * (1 / (1 + (bins / d0) ** 2))).sum(-1)
            physical_interchain = ~torch.eye(4, dtype=torch.bool)
            rows = (expected[0] * physical_interchain).sum(-1) / (physical_interchain.sum(-1) + 1e-5)
            return float(rows.max())

        old_scores = [old_physical_chain_score(item) for item in (candidate_a, candidate_b)]
        self.assertGreater(old_scores[1], old_scores[0])
        self.assertGreater(corrected[0], corrected[1])

        selected: list[str] = []

        def run_step(step: int, _temperature: float, _confidence: bool) -> StepResult:
            return StepResult(
                sequences=[["AAA|X", "AAA|Y"][step]],
                iptm=[[corrected[0], corrected[1]][step]],
                losses={"total_loss": [0.0]},
            )

        run_design_loop(
            steps=2,
            batch_size=1,
            run_step=run_step,
            score_sequence=lambda _index, sequence, _trajectory: selected.append(sequence) or [],
        )
        self.assertEqual(selected, ["AAA|X"])


if __name__ == "__main__":
    unittest.main()
