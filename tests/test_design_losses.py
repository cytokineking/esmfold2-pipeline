from __future__ import annotations

import importlib.util
import types
import unittest
from unittest.mock import patch

import numpy as np

from esmfold2_pipeline.design import losses
from esmfold2_pipeline.esm_adapter.binder_design import (
    _framework_penalty_hotspot_indices,
)


class DesignLossTest(unittest.TestCase):
    def test_mosaic_framework_penalty_scope_selects_target_mask(self) -> None:
        self.assertEqual(
            _framework_penalty_hotspot_indices(
                hotspot_indices=(2, 5),
                scope="auto",
            ),
            (2, 5),
        )
        self.assertEqual(
            _framework_penalty_hotspot_indices(
                hotspot_indices=(),
                scope="auto",
            ),
            (),
        )
        self.assertEqual(
            _framework_penalty_hotspot_indices(
                hotspot_indices=(2, 5),
                scope="hotspot",
            ),
            (2, 5),
        )
        self.assertEqual(
            _framework_penalty_hotspot_indices(
                hotspot_indices=(2, 5),
                scope="target_all",
            ),
            (),
        )
        with self.assertRaisesRegex(ValueError, "requires target.hotspots"):
            _framework_penalty_hotspot_indices(
                hotspot_indices=(),
                scope="hotspot",
            )

    def test_target_geometry_drift_hinge_works_with_torch_like_object(self) -> None:
        fake_torch = types.SimpleNamespace(
            relu=lambda value: np.maximum(value, np.float32(0.0)),
        )

        loss = losses.compute_target_geometry_drift_hinge_loss(
            fake_torch,
            np.array([0.9], dtype=np.float32),
            tolerance_angstrom=0.5,
            stiffness_angstrom=0.1,
        )
        below_tolerance = losses.compute_target_geometry_drift_hinge_loss(
            fake_torch,
            np.array([0.25], dtype=np.float32),
            tolerance_angstrom=0.5,
            stiffness_angstrom=0.1,
        )

        self.assertAlmostEqual(float(loss[0]), 4.0, places=6)
        self.assertAlmostEqual(float(below_tolerance[0]), 0.0, places=6)

    def test_design_structure_loss_composes_drift_then_hotspot(self) -> None:
        with patch.object(
            losses,
            "compute_structure_losses",
            return_value={"total_loss": 1.0},
        ) as base_losses, patch.object(
            losses,
            "compute_target_geometry_drift_loss",
            return_value=(2.0, 3.0),
        ) as drift_loss, patch.object(
            losses,
            "compute_hotspot_contact_loss",
            return_value=4.0,
        ) as hotspot_loss:
            result = losses.compute_design_structure_losses(
                "distogram-logits",
                5,
                torch_module="torch-runtime",
                bin_distance="bin-midpoints",
                target_geometry_reference_distances=np.zeros((2, 2)),
                target_geometry_pair_mask=np.array(
                    [[False, True], [False, False]],
                ),
                target_geometry_weight=0.5,
                target_geometry_tolerance_angstrom=1.0,
                target_geometry_stiffness_angstrom=2.0,
                hotspot_indices=(0,),
                hotspot_contact_weight=0.25,
                hotspot_contact_cutoff_angstrom=20.0,
                hotspot_num_contacts=2,
                hotspot_contact_probability_target=0.6,
                hotspot_loss_mode="probability_hinge",
                binder_contact_indices=(1,),
            )

        self.assertEqual(result["total_loss"], 3.0)
        self.assertEqual(result["target_geometry_drift_loss"], 2.0)
        self.assertEqual(result["target_geometry_drift_rmse"], 3.0)
        self.assertEqual(result["hotspot_contact_loss"], 4.0)
        base_losses.assert_called_once_with(
            "distogram-logits",
            5,
            torch_module="torch-runtime",
            bin_distance="bin-midpoints",
            include_inter_contact=True,
        )
        drift_loss.assert_called_once()
        self.assertEqual(drift_loss.call_args.args, (
            "torch-runtime",
            "distogram-logits",
            5,
        ))
        np.testing.assert_array_equal(
            drift_loss.call_args.kwargs["reference_distances"],
            np.zeros((2, 2)),
        )
        np.testing.assert_array_equal(
            drift_loss.call_args.kwargs["pair_mask"],
            np.array([[False, True], [False, False]]),
        )
        self.assertEqual(drift_loss.call_args.kwargs["tolerance_angstrom"], 1.0)
        self.assertEqual(drift_loss.call_args.kwargs["stiffness_angstrom"], 2.0)
        self.assertEqual(drift_loss.call_args.kwargs["bin_distances"], "bin-midpoints")
        hotspot_loss.assert_called_once_with(
            "torch-runtime",
            "distogram-logits",
            5,
            hotspot_indices=(0,),
            contact_cutoff_angstrom=20.0,
            hotspot_num_contacts=2,
            contact_probability_target=0.6,
            hotspot_loss_mode="probability_hinge",
            binder_contact_indices=(1,),
            bin_distances="bin-midpoints",
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_base_structure_loss_total_uses_original_weights(self) -> None:
        import torch

        distogram_logits = torch.zeros((1, 5, 5, 128), dtype=torch.float32)

        result = losses.compute_structure_losses(
            distogram_logits,
            binder_length=2,
            torch_module=torch,
        )

        expected_total = (
            0.5 * result["intra_contact_loss"]
            + 0.5 * result["inter_contact_loss"]
            + 0.2 * result["glob_loss"]
        )
        self.assertTrue(torch.allclose(result["total_loss"], expected_total))
        self.assertEqual(set(result), {
            "glob_loss",
            "inter_contact_loss",
            "intra_contact_loss",
            "total_loss",
        })

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_structure_losses_can_disable_legacy_inter_contact(self) -> None:
        import torch

        distogram_logits = torch.zeros((1, 5, 5, 128), dtype=torch.float32)

        result = losses.compute_structure_losses(
            distogram_logits,
            binder_length=2,
            torch_module=torch,
            include_inter_contact=False,
        )

        expected_total = (
            0.5 * result["intra_contact_loss"]
            + 0.2 * result["glob_loss"]
        )
        self.assertTrue(torch.allclose(result["total_loss"], expected_total))
        self.assertNotIn("inter_contact_loss", result)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_mosaic_cdr_loss_scores_target_hotspots_or_any_target(self) -> None:
        import torch

        target_length = 3
        binder_length = 4
        distogram_logits = torch.zeros(
            (1, target_length + binder_length, target_length + binder_length, 2),
            dtype=torch.float32,
        )
        bin_distances = torch.tensor([1.0, 10.0], dtype=torch.float32)
        cdr_indices = (1, 3)
        cdr_row = target_length + 1
        distogram_logits[0, cdr_row, 0, 0] = 10.0

        any_target_scores = losses.mosaic_cdr_contact_probability_scores(
            torch,
            distogram_logits,
            binder_length,
            cdr_indices=cdr_indices,
            contact_cutoff_angstrom=5.0,
            num_target_contacts=1,
            hotspot_indices=(),
            bin_distances=bin_distances,
        )
        hotspot_scores = losses.mosaic_cdr_contact_probability_scores(
            torch,
            distogram_logits,
            binder_length,
            cdr_indices=cdr_indices,
            contact_cutoff_angstrom=5.0,
            num_target_contacts=1,
            hotspot_indices=(2,),
            bin_distances=bin_distances,
        )

        self.assertGreater(float(any_target_scores[0, 0]), 0.99)
        self.assertLess(float(hotspot_scores[0, 0]), 0.51)

        any_target_loss = losses.compute_mosaic_cdr_contact_loss(
            torch,
            distogram_logits,
            binder_length,
            cdr_indices=cdr_indices,
            contact_cutoff_angstrom=5.0,
            num_target_contacts=1,
            hotspot_indices=(),
            bin_distances=bin_distances,
        )
        missed_hotspot_loss = losses.compute_mosaic_cdr_contact_loss(
            torch,
            distogram_logits,
            binder_length,
            cdr_indices=cdr_indices,
            contact_cutoff_angstrom=5.0,
            num_target_contacts=1,
            hotspot_indices=(2,),
            bin_distances=bin_distances,
        )
        self.assertLess(float(any_target_loss[0]), float(missed_hotspot_loss[0]))

        distogram_logits[0, cdr_row, 2, 0] = 10.0
        hotspot_loss = losses.compute_mosaic_cdr_contact_loss(
            torch,
            distogram_logits,
            binder_length,
            cdr_indices=cdr_indices,
            contact_cutoff_angstrom=5.0,
            num_target_contacts=1,
            hotspot_indices=(2,),
            bin_distances=bin_distances,
        )

        self.assertLess(float(hotspot_loss[0]), float(missed_hotspot_loss[0]))
        self.assertLess(float(hotspot_loss[0]), 0.4)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_framework_contact_penalty_ignores_cdr_contacts(self) -> None:
        import torch

        target_length = 2
        binder_length = 4
        distogram_logits = torch.zeros(
            (1, target_length + binder_length, target_length + binder_length, 2),
            dtype=torch.float32,
        )
        bin_distances = torch.tensor([1.0, 10.0], dtype=torch.float32)
        cdr_indices = (1,)
        cdr_row = target_length + 1
        framework_row = target_length + 0
        distogram_logits[0, cdr_row, 0, 0] = 10.0

        cdr_only_penalty = losses.compute_framework_contact_penalty_loss(
            torch,
            distogram_logits,
            binder_length,
            cdr_indices=cdr_indices,
            contact_cutoff_angstrom=5.0,
            num_target_contacts=1,
            contact_probability_threshold=0.8,
            hotspot_indices=(),
            bin_distances=bin_distances,
        )

        self.assertAlmostEqual(float(cdr_only_penalty[0]), 0.0, places=6)

        distogram_logits[0, framework_row, 0, 0] = 10.0
        framework_penalty = losses.compute_framework_contact_penalty_loss(
            torch,
            distogram_logits,
            binder_length,
            cdr_indices=cdr_indices,
            contact_cutoff_angstrom=5.0,
            num_target_contacts=1,
            contact_probability_threshold=0.8,
            hotspot_indices=(),
            bin_distances=bin_distances,
        )

        self.assertGreater(float(framework_penalty[0]), 0.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_framework_contact_penalty_rejects_zero_threshold(self) -> None:
        import torch

        distogram_logits = torch.zeros((1, 5, 5, 2), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "greater than 0"):
            losses.compute_framework_contact_penalty_loss(
                torch,
                distogram_logits,
                binder_length=2,
                cdr_indices=(0,),
                contact_cutoff_angstrom=5.0,
                num_target_contacts=1,
                contact_probability_threshold=0.0,
                bin_distances=torch.tensor([1.0, 10.0], dtype=torch.float32),
            )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_structure_losses_accept_cpu_bin_distances_for_cuda_logits(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        distogram_logits = torch.zeros((1, 5, 5, 128), dtype=torch.float32, device="cuda")
        bin_distance = losses.get_mid_points(torch)

        result = losses.compute_structure_losses(
            distogram_logits,
            binder_length=2,
            bin_distance=bin_distance,
            torch_module=torch,
        )

        self.assertEqual(result["total_loss"].device.type, "cuda")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_binder_contact_mask_can_restrict_to_cdr_indices(self) -> None:
        import torch

        mask = losses.binder_contact_mask(
            torch,
            full_len=7,
            target_length=4,
            binder_length=3,
            binder_contact_indices=(1,),
            device=torch.device("cpu"),
        )

        self.assertEqual(mask.tolist(), [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
