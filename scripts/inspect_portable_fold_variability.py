#!/usr/bin/env python3
"""Inspect stock run-to-run variation beside a portable model call.

Run on a qualified GPU image after sourcing env.sh. ESMFold2 feature creation
is outside the model call and may vary, so prepare its tensors once here.
This is diagnostic evidence, not a numerical equivalence assertion: the pinned
stock ESMFold2 call is not bitwise repeatable on the qualified H100 image.
"""

from __future__ import annotations

import json
import os

import torch

from esmfold2_pipeline.design.loop import sequence_to_one_hot
from esmfold2_pipeline.esm_adapter.binder_design import (
    _fold_and_get_distogram_for_sequence_target,
    _load_local_hf_esmfold2_model,
    _resolve_esmc_snapshot,
)
from esmfold2_pipeline.esm_adapter.folding_runtime import load_esm_folding_runtime
from esmfold2_pipeline.esm_adapter.portable import call_model


def main() -> int:
    runtime = load_esm_folding_runtime(None)
    model = _load_local_hf_esmfold2_model(
        runtime, "ESMFold2-Experimental-Fast-Cutoff2025",
        lm_dropout=0.5, cache_esmc=True, device="cuda",
        esmc_snapshot=_resolve_esmc_snapshot(),
    )
    # Use the stock PyTorch backend to make the adapter's exactness check
    # independent of nondeterministic accelerator reductions.
    model.set_kernel_backend(None)
    model.configure_lm_dropout(0.0, force_lm_dropout_during_inference=False)
    target = runtime.TARGET_SEQUENCES["pd-l1"]
    target_one_hot = sequence_to_one_hot(target, torch_module=torch)
    torch.manual_seed(2718)
    initial = torch.randn((1, 18, 20), device="cuda")
    os.environ["ESMFOLD2_PIPELINE_OPTIMIZATIONS"] = "off"
    fixture = _fold_and_get_distogram_for_sequence_target(
        runtime, model, target, target_one_hot, initial.softmax(dim=-1),
        num_loops=1, num_sampling_steps=1,
        calculate_confidence=False, seed=91,
    )
    prepared = fixture["inputs"]

    def fold(mode):
        os.environ["ESMFOLD2_PIPELINE_OPTIMIZATIONS"] = mode
        logits = initial.clone().requires_grad_(True)
        design = torch.nn.functional.pad(logits.softmax(dim=-1), (2, 11))
        inputs = {name: value.clone() for name, value in prepared.items()}
        inputs["res_type_soft"] = torch.cat((target_one_hot, design), dim=1)
        with runtime.seed_context(91):
            output = call_model(
                model,
                lambda: model(
                    **inputs, num_diffusion_samples=1, num_sampling_steps=1,
                    num_loops=1, calculate_confidence=False, seed=91,
                    lm_mask_pct=0.0, msa_column_mask_rate=0.0,
                    msa_subsample_at_inference=False,
                ),
                calculate_confidence=False, seed=91, num_sampling_steps=1,
            )
        gradient = torch.autograd.grad(
            output["distogram_logits"].float().sum(), logits
        )[0]
        coords = output["sample_atom_coords"]
        torch.cuda.synchronize()
        return (output["distogram_logits"].detach(), gradient.detach(),
                coords.detach(), torch.cuda.get_rng_state().clone())

    baseline = fold("off")
    baseline_repeat = fold("off")
    candidate = fold("portable")
    names = ("distogram", "design_gradient", "sample_coordinates", "cuda_rng")
    evidence = {}
    for name, before, repeated, after in zip(names, baseline, baseline_repeat, candidate):
        evidence[name] = {}
        for label, other in (("baseline_repeat", repeated), ("portable", after)):
            evidence[name][label] = {
                "equal": bool(torch.equal(before, other)),
                "max_abs_difference": (
                    float((before.float() - other.float()).abs().max().item())
                    if before.shape == other.shape else None
                ),
            }
    evidence["model_stats"] = getattr(model, "_esmfold2_portable_stats", {})
    print(json.dumps(evidence, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
