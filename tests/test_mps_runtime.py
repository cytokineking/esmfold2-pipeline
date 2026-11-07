from __future__ import annotations

from contextlib import contextmanager
import os
import random
import tempfile
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from esmfold2_pipeline.esm_adapter.binder_design import _local_model_load_specs
from esmfold2_pipeline.esm_adapter.mps import mps_seed_context


class MPSRuntimeTests(unittest.TestCase):
    def test_device_selection_preserves_cuda_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            specs = _local_model_load_specs(inversion_model_name="a", critic_name="b")
        self.assertEqual([spec.device for spec in specs], ["cuda", "cuda"])
        with patch.dict("os.environ", {"ESMFOLD2_PIPELINE_DEVICE": "mps"}):
            specs = _local_model_load_specs(inversion_model_name="a", critic_name="b")
        self.assertEqual([spec.device for spec in specs], ["mps", "mps"])

    def test_mps_random_state_is_restored_on_error(self) -> None:
        states = []
        torch = types.SimpleNamespace(mps=types.SimpleNamespace(
            get_rng_state=lambda: "saved-state", set_rng_state=states.append,
        ))

        @contextmanager
        def upstream(seed):
            states.append(seed)
            yield

        with self.assertRaisesRegex(RuntimeError, "stop"):
            with mps_seed_context(torch, upstream, 42):
                raise RuntimeError("stop")
        self.assertEqual(states, [42, "saved-state"])

    def test_interrupted_design_resumes_with_identical_logits(self) -> None:
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("requires torch and numpy")
        from esmfold2_pipeline.design.loop import run_gradient_design_loop
        device = os.environ.get("ESMFOLD2_TEST_DEVICE", "cpu")
        if device == "mps" and not torch.backends.mps.is_available():
            self.skipTest("MPS unavailable")

        @contextmanager
        def seeded(seed):
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                yield

        def device_seeded(seed):
            if device == "mps":
                return mps_seed_context(torch, seeded, seed)
            return seeded(seed)

        def run(path, fail_once):
            def fold(_model, target, _target_one_hot, design, *, num_loops,
                     num_sampling_steps, calculate_confidence, seed):
                del num_sampling_steps
                if num_loops == 1 and seed == 44 and fail_once[0]:
                    fail_once[0] = False
                    raise RuntimeError("interrupted")
                factor = 1 + torch.rand((), device=device).item() + float(np.random.random())
                weights = torch.arange(20, dtype=design.dtype, device=design.device)
                result = {"seq_list": [target + "|AAAAA"],
                          "distogram_logits": (design * weights).sum(-1) * factor,
                          "inputs": {}, "output": {}}
                if calculate_confidence:
                    result["iptm"] = torch.tensor([0.8], device=device)
                return result

            return run_gradient_design_loop(
                target_sequence="A", binder_sequence="A###A", is_antibody=True,
                seed=42, steps=4, batch_size=1, inversion_models={"toy": object()},
                critic_models={"toy": object()}, esmc_model=object(), fold_complex=fold,
                compute_structure_losses=lambda x, _n: {"total_loss": x.square().mean(-1)},
                compute_plm_loss=lambda **kw: (kw["binder_design"] * torch.arange(20, device=device)).sum((-1, -2)),
                build_complex=lambda *_: object(),
                compute_distogram_iptm_proxy=lambda *_: {},
                seed_context=device_seeded, device=device, checkpoint_path=path,
                checkpoint_fingerprint="test-config" if path else None,
            )

        def reset():
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)

        reset()
        expected = run(None, [False])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.pt"
            reset()
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run(path, [True])
            self.assertTrue(path.exists())
            random.seed(1234)
            np.random.seed(1234)
            torch.manual_seed(1234)
            resumed = run(path, [False])
            self.assertEqual(resumed.best_sequences, expected.best_sequences)
            self.assertEqual(set(resumed.trajectory), set(range(4)))
            torch.testing.assert_close(
                resumed.critic_results[0]["logits"], expected.critic_results[0]["logits"],
                rtol=0, atol=0,
            )
