"""Small real-CUDA parity check for the portable ESMC masking path."""

from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch

from esmfold2_pipeline.design.loop import PROTEIN_1TO3, TOKENS
from esmfold2_pipeline.design.plm import compute_esmc_pseudoperplexity_nll


class _Tokenizer:
    def __init__(self):
        three_to_one = {value: key for key, value in PROTEIN_1TO3.items()}
        aas = [three_to_one[token] for token in TOKENS[2:22]]
        self.vocab = {f"special_{index}": index for index in range(4)}
        self.vocab.update({aa: index + 4 for index, aa in enumerate(aas)})
        self.cls_token_id = 0
        self.eos_token_id = 1


class DesignPlmCudaParityTest(unittest.TestCase):
    def test_loss_gradient_masks_and_rng_match(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is only installed on the GPU image")
        if not torch.cuda.is_available():
            self.skipTest("CUDA GPU required")

        torch.manual_seed(70)
        seen = []

        class ESMC:
            def __init__(self):
                self.embed = types.SimpleNamespace(
                    weight=torch.randn((24, 32), device="cuda")
                )

            def transformer(self, value, **_kwargs):
                seen.append(value.detach().clone())
                return value, None

        class Model:
            def __init__(self):
                self.config = types.SimpleNamespace(vocab_size=24, mask_token_id=2)
                self.esmc = ESMC()
                self.lm_head = torch.nn.Linear(32, 24, bias=False, device="cuda")

        model = Model()
        initial = torch.randn((1, 12, 20), device="cuda")
        score_mask = torch.tensor(
            [[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0]],
            device="cuda", dtype=torch.bool,
        )

        def run(mode):
            seen.clear()
            torch.cuda.manual_seed(1234)
            logits = initial.clone().requires_grad_(True)
            with patch.dict(os.environ, {"ESMFOLD2_PIPELINE_OPTIMIZATIONS": mode}):
                loss = compute_esmc_pseudoperplexity_nll(
                    esmc_model=model,
                    binder_design=logits.softmax(dim=-1),
                    score_mask=score_mask,
                    tokenizer_factory=_Tokenizer,
                )
                gradient = torch.autograd.grad(loss.sum(), logits)[0]
            return (loss.detach().clone(), gradient.detach().clone(),
                    [item.clone() for item in seen], torch.cuda.get_rng_state().clone())

        baseline = run("off")
        portable = run("portable")
        reused = run("portable")
        for candidate in (portable, reused):
            self.assertTrue(torch.equal(baseline[0], candidate[0]), "PLM loss differs")
            self.assertTrue(torch.equal(baseline[1], candidate[1]), "PLM gradient differs")
            self.assertEqual(len(baseline[2]), len(candidate[2]))
            for before, after in zip(baseline[2], candidate[2]):
                self.assertTrue(torch.equal(before, after), "masked inputs differ")
            self.assertTrue(torch.equal(baseline[3], candidate[3]), "CUDA RNG differs")
        self.assertEqual(model._esmfold2_portable_plm_stats["reused"], 1)


if __name__ == "__main__":
    unittest.main()
