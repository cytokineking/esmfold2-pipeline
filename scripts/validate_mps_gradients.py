#!/usr/bin/env python3
"""Compare fixed-input CPU and MPS gradients using the real pretrained models.

Run ``prepare``, then ``run cpu`` and ``run mps``, then ``compare`` with the
same --out directory. All models and tensors are unquantized FP32. The small
two-chain example makes the full 6B CPU reference tractable.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F

from esmfold2_pipeline.config import load_campaign_config
from esmfold2_pipeline.design import plm
from esmfold2_pipeline.design.loop import (
    LEARNING_RATE,
    build_gradient_mask,
    build_initial_soft_sequence_logits,
    normalized_gradient_tensor,
    sequence_to_one_hot,
)
from esmfold2_pipeline.esm_adapter.binder_design import (
    _load_pretrained_with_required_weights,
    _apply_target_distogram_conditioning,
    _build_distogram_template_pair_bias,
    _local_structure_loss_callback,
    _patched_model_folding_trunk_with_distogram_template,
    _resolve_esmc_snapshot,
)
from esmfold2_pipeline.esm_adapter.folding_runtime import load_esm_folding_runtime
from esmfold2_pipeline.esm_adapter.mps import (
    MPS_MODEL_REVISIONS,
    install_mps_compatibility,
    optimize_mps_trunk,
)
from esmfold2_pipeline.structure import parse_structure_target


TARGET = "ALIVYWEMEDKNIIQF"  # PD-L1 A:52-67, containing two benchmark hotspots.
PROMPT = "EVQLVESGGG########GWFRQAPG"
CDR = tuple(i for i, aa in enumerate(PROMPT) if aa == "#")
MODEL = "biohub/ESMFold2-Experimental-Fast-Cutoff2025"
CONFIG = Path(__file__).resolve().parents[1] / "example_configs/apple/pdl1-vhh.yaml"


def cropped_target(config_path: Path):
    config = load_campaign_config(config_path)
    full = parse_structure_target(config.target_structure)
    chain = full.chains[0]
    start, end = 34, 50
    assert chain.sequence[start:end] == TARGET
    cropped = replace(
        chain,
        sequence=chain.sequence[start:end],
        residues=tuple(replace(r, model_residue_index_0=i)
                       for i, r in enumerate(chain.residues[start:end])),
        hotspot_indices=tuple(i-start for i in chain.hotspot_indices if start <= i < end),
        distogram=tuple(tuple(row[start:end]) for row in chain.distogram[start:end]),
        distogram_mask=tuple(tuple(row[start:end]) for row in chain.distogram_mask[start:end]),
        representative_coord_mask=chain.representative_coord_mask[start:end],
    )
    return config, replace(full, chains=(cropped,))


def prepare(out: Path, config_path: Path) -> None:
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    runtime = load_esm_folding_runtime()
    torch.manual_seed(42)
    logits = build_initial_soft_sequence_logits(PROMPT, batch_size=1).detach()
    design = logits.softmax(-1)
    ids = F.pad(design, (2, 11)).argmax(-1)[0]
    binder = "".join(runtime.PROTEIN_3TO1[runtime.TOKENS[int(i)]] for i in ids)
    inputs = StructurePredictionInput(sequences=[
        ProteinInput(id="A", sequence=TARGET, msa=None),
        ProteinInput(id="B", sequence=binder, msa=None),
    ])
    features = runtime.prepare_esmfold2_tensors(inputs, seed=42)
    _, target = cropped_target(config_path)
    _apply_target_distogram_conditioning(features, structure_target=target, assembly_pairs=())
    generator = torch.Generator().manual_seed(20261106)
    scores = torch.rand((1, 4, len(CDR)), generator=generator)
    dropout_mask = torch.rand((1, len(TARGET) + len(PROMPT),
                               len(TARGET) + len(PROMPT), 256), generator=generator) >= 0.5
    torch.save({"features": features, "logits": logits, "scores": scores,
                "dropout_mask": dropout_mask, "binder": binder}, out / "fixed-inputs.pt")
    print(f"prepared target={TARGET} binder={binder} mutable={CDR}", flush=True)


def run(device: str, out: Path, config_path: Path) -> None:
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        install_mps_compatibility(torch)
    fixed = torch.load(out / "fixed-inputs.pt", map_location="cpu", weights_only=True)
    runtime = load_esm_folding_runtime()
    started = time.perf_counter()
    snapshot = _resolve_esmc_snapshot()
    mlm = _load_pretrained_with_required_weights(
        runtime.ESMCForMaskedLM, snapshot, torch_dtype=torch.float32,
    ).to(device).eval().requires_grad_(False)
    fold = runtime.ESMFold2ExperimentalModel.from_pretrained(
        MODEL, load_esmc=False, revision=MPS_MODEL_REVISIONS[MODEL],
    )
    fold._esmc = mlm.esmc
    fold.configure_lm_dropout(0.5, force_lm_dropout_during_inference=True)
    fold.set_kernel_backend(None)
    fold = fold.to(device).eval().requires_grad_(False)
    if device == "mps":
        optimize_mps_trunk(fold)
        torch.mps.synchronize()
    print(f"{device} model setup seconds={time.perf_counter()-started:.2f}", flush=True)

    # Confidence and sampled coordinates do not participate in the distogram
    # objective. Omit them only in this comparison, to avoid unrelated random
    # draws and CPU work; the benchmark runs the real late confidence/critic.
    features = {k: v.unsqueeze(0).to(device) for k, v in fixed["features"].items()}
    logits = fixed["logits"].to(device).requires_grad_(True)
    gradient_mask = build_gradient_mask(PROMPT, 1).to(device)
    design = logits.softmax(-1)
    soft = torch.cat((sequence_to_one_hot(TARGET, device=device), F.pad(design, (2, 11))), dim=1)
    config, target = cropped_target(config_path)
    template_bias = _build_distogram_template_pair_bias(
        runtime, fold, {"res_type_soft": soft}, structure_target=target,
        binder_length=len(PROMPT), enabled=True,
    )
    structure_loss = _local_structure_loss_callback(
        runtime, structure_target=target,
        target_geometry_drift=config.target_geometry_drift,
        target_geometry_drift_indices=tuple(range(len(TARGET))),
        hotspot_contact_weight=config.hotspot_contact_weight,
        hotspot_distogram_contact_cutoff_angstrom=config.hotspot_distogram_contact_cutoff_angstrom,
        hotspot_num_contacts=config.hotspot_num_contacts,
        hotspot_contact_probability_target=config.hotspot_contact_probability_target,
        hotspot_loss_mode=config.hotspot_loss_mode,
        binder_contact_indices=CDR,
        binder_target_contact_mode=config.binder_target_contact_mode,
        mosaic_cdr_contact_weight=config.mosaic_cdr_contact_weight,
        mosaic_cdr_contact_cutoff_angstrom=config.mosaic_cdr_contact_cutoff_angstrom,
        mosaic_cdr_num_target_contacts=config.mosaic_cdr_num_target_contacts,
        mosaic_framework_contact_penalty_weight=config.mosaic_framework_contact_penalty_weight,
        mosaic_framework_contact_penalty_cutoff_angstrom=config.mosaic_framework_contact_penalty_cutoff_angstrom,
        mosaic_framework_contact_probability_threshold=config.mosaic_framework_contact_probability_threshold,
        mosaic_framework_contact_penalty_scope=config.mosaic_framework_contact_penalty_scope,
    )
    fold.confidence_head = None
    fold.structure_head.sample = lambda **kw: {
        "sample_atom_coords": kw["ref_pos"].unsqueeze(1)
    }
    original_dropout = F.dropout
    dropout_calls = []
    def matched_dropout(value, p=0.5, training=True, inplace=False):
        if training and p == 0.5 and tuple(value.shape) == tuple(fixed["dropout_mask"].shape):
            dropout_calls.append(tuple(value.shape))
            return value * fixed["dropout_mask"].to(value.device, value.dtype) / (1-p)
        return original_dropout(value, p=p, training=training, inplace=inplace)
    start = time.perf_counter()
    with patch.object(F, "dropout", matched_dropout), _patched_model_folding_trunk_with_distogram_template(fold, template_bias):
        result = fold(**features, res_type_soft=soft, num_loops=1,
                      num_diffusion_samples=1, num_sampling_steps=1, lm_mask_pct=0.0, seed=42)
    if len(dropout_calls) != 1:
        raise RuntimeError(f"expected one matched LM dropout draw, saw {len(dropout_calls)}")
    dist = result["distogram_logits"]
    loss_components = structure_loss(dist, len(PROMPT))
    required = {"target_geometry_drift_loss", "mosaic_cdr_contact_loss",
                "mosaic_framework_contact_penalty_loss"}
    if not required.issubset(loss_components):
        raise RuntimeError("benchmark loss components are missing")
    dist_loss = loss_components["total_loss"]
    soft_grad, structure_grad = torch.autograd.grad(dist_loss.mean(), (soft, logits))
    if device == "mps":
        torch.mps.synchronize()
    print(f"{device} structure seconds={time.perf_counter()-start:.2f}", flush=True)
    del result, dist, dist_loss, soft, design, features, fold

    logits2 = fixed["logits"].to(device).requires_grad_(True)
    start = time.perf_counter()
    nll = plm.compute_esmc_pseudoperplexity_nll(
        esmc_model=mlm, binder_design=logits2.softmax(-1),
        score_mask=gradient_mask.sum(-1) > 0, batch_size=4, n_passes=4,
        tokenizer_factory=runtime.ESMCTokenizer, mask_random_scores=fixed["scores"],
    )
    plm_grad = torch.autograd.grad(nll.mean(), logits2)[0]
    if device == "mps":
        torch.mps.synchronize()
    print(f"{device} plm seconds={time.perf_counter()-start:.2f} nll={nll.item():.7g}", flush=True)
    grads = {"res_type_soft": soft_grad.detach().cpu(),
             "structure_logits": structure_grad.detach().cpu(),
             "plm_logits": plm_grad.detach().cpu(),
             "plm_loss": nll.detach().cpu(), "mask": gradient_mask.detach().cpu()}
    for name in ("structure_logits", "plm_logits"):
        normalized = normalized_gradient_tensor(grads[name], grads["mask"])
        grads[name + "_update"] = -LEARNING_RATE * normalized
    torch.save(grads, out / f"{device}-gradients.pt")
    print(f"{device} saved", flush=True)


def compare(out: Path) -> None:
    ref = torch.load(out / "cpu-gradients.pt", weights_only=True)
    got = torch.load(out / "mps-gradients.pt", weights_only=True)
    assert torch.equal(ref["mask"], got["mask"])
    mask = ref["mask"].bool()
    for data in (ref, got):
        # Antibody design combines separately normalized gradients with a 0.05
        # PLM weight. These fixed-input checks use temperature 1 and lr 0.1.
        data["combined_update"] = (
            data["structure_logits_update"] + 0.05 * data["plm_logits_update"]
        )
    report = {}
    for name in ("res_type_soft", "structure_logits", "plm_logits",
                 "structure_logits_update", "plm_logits_update", "combined_update"):
        a, b = ref[name].float(), got[name].float()
        if name != "res_type_soft":
            a, b = a[mask], b[mask]
        aa, bb = a.flatten(), b.flatten()
        na, nb = aa.norm().item(), bb.norm().item()
        cosine = F.cosine_similarity(aa, bb, dim=0).item() if na and nb else None
        if cosine is not None:
            cosine = min(1.0, max(-1.0, cosine))
        report[name] = {"finite_cpu": bool(torch.isfinite(aa).all()),
                        "finite_mps": bool(torch.isfinite(bb).all()),
                        "cpu_norm": na, "mps_norm": nb,
                        "cosine": cosine,
                        "relative_l2_error": (aa-bb).norm().item() / max(na, 1e-30),
                        "max_abs_difference": (aa-bb).abs().max().item()}
    report["plm_loss"] = {"cpu": ref["plm_loss"].item(), "mps": got["plm_loss"].item()}
    report["mutable_positions"] = int(ref["mask"].any(-1).sum())
    site_update_norms = got["combined_update"].norm(dim=-1)[0]
    report["mps_mutable_site_update_l2"] = {
        str(index): site_update_norms[index].item() for index in CDR
    }
    report["fixed_and_cysteine_updates_zero"] = all(
        bool((got[k][~mask] == 0).all())
        for k in ("structure_logits_update", "plm_logits_update", "combined_update")
    )
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    for name in ("res_type_soft", "structure_logits", "plm_logits"):
        row = report[name]
        if not (row["finite_cpu"] and row["finite_mps"] and row["cpu_norm"] > 0
                and row["mps_norm"] > 0 and row["cosine"] is not None
                and row["cosine"] > 0.99 and row["relative_l2_error"] < 0.1):
            raise RuntimeError(f"gradient comparison failed: {name}")
    if not report["fixed_and_cysteine_updates_zero"]:
        raise RuntimeError("gradient mask did not protect fixed positions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "compare"))
    parser.add_argument("device", nargs="?", choices=("cpu", "mps"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(args.out, args.config)
    elif args.action == "run":
        if not args.device:
            parser.error("run requires cpu or mps")
        run(args.device, args.out, args.config)
    else:
        compare(args.out)


if __name__ == "__main__":
    main()
