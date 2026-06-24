from __future__ import annotations

import importlib.util
import types
import unittest
from unittest.mock import patch

import numpy as np

from esmfold2_pipeline.design import losses


class DesignLossTest(unittest.TestCase):
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
