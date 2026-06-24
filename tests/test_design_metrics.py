from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from esmfold2_pipeline.design import metrics


class _FakeTorch:
    @staticmethod
    def sort(values):
        return np.sort(np.asarray(values)), None


class DesignMetricsTest(unittest.TestCase):
    def test_entropy_to_confidence_matches_tutorial_mapping(self) -> None:
        self.assertEqual(metrics.entropy_to_confidence(0.0), 1.0)
        self.assertEqual(metrics.entropy_to_confidence(math.log(51)), 0.0)
        self.assertEqual(metrics.entropy_to_confidence(10.0), 0.0)

    def test_distogram_iptm_proxy_uses_lowest_pair_entropies_and_cdr_indices(self) -> None:
        distogram_logits = np.zeros((1, 5, 5, 128), dtype=np.float32)

        with patch.object(
            metrics,
            "binding_confidence_entropy",
            side_effect=[
                np.array([0.3, 0.1, 0.2, 0.7], dtype=np.float32),
                np.array([0.6, 0.2, 0.4], dtype=np.float32),
            ],
        ) as entropy:
            result = metrics.compute_distogram_iptm_proxy(
                distogram_logits,
                target_length=2,
                binder_sequence="ACD",
                is_antibody=True,
                cdr_indices=(1, 2),
                bin_distance="bin-midpoints",
                torch_module=_FakeTorch,
            )

        expected_complex = 1.0 - 0.2 / math.log(51)
        expected_cdr = 1.0 - 0.3 / math.log(51)
        self.assertAlmostEqual(result["distogram_iptm_proxy"], expected_complex)
        self.assertAlmostEqual(result["cdr_distogram_iptm_proxy"], expected_cdr)
        self.assertEqual(entropy.call_count, 2)
        self.assertEqual(entropy.call_args_list[0].args[0].shape, (3, 2, 128))
        self.assertEqual(entropy.call_args_list[1].args[0].shape, (2, 2, 128))
        self.assertEqual(entropy.call_args_list[0].args[1], "bin-midpoints")
        self.assertEqual(entropy.call_args_list[0].kwargs["cutoff"], 22.0)

    def test_distogram_iptm_proxy_returns_nan_cdr_score_for_non_antibody(self) -> None:
        distogram_logits = np.zeros((5, 5, 128), dtype=np.float32)

        with patch.object(
            metrics,
            "binding_confidence_entropy",
            return_value=np.array([0.2, 0.1, 0.3], dtype=np.float32),
        ):
            result = metrics.compute_distogram_iptm_proxy(
                distogram_logits,
                target_length=2,
                binder_sequence="ACD",
                is_antibody=False,
                bin_distance="bin-midpoints",
                torch_module=_FakeTorch,
            )

        self.assertTrue(math.isnan(result["cdr_distogram_iptm_proxy"]))


if __name__ == "__main__":
    unittest.main()
