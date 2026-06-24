from __future__ import annotations

import math
from typing import Any

from esmfold2_pipeline.design.losses import get_mid_points


def binding_confidence_entropy(
    distogram_logits,
    bin_distance,
    cutoff: float,
    *,
    torch_module: Any | None = None,
):
    """Pair entropy within cutoff, matching the ESM tutorial proxy metric."""

    torch = _torch(torch_module)
    probs = torch.softmax(distogram_logits, dim=-1)
    cutoff_mask = bin_distance < cutoff
    p_cut = probs[..., cutoff_mask]
    p_cut = p_cut / (p_cut.sum(-1, keepdim=True) + 1e-8)
    return -(p_cut * torch.log(p_cut + 1e-10)).sum(-1)


def entropy_to_confidence(mean_entropy: float) -> float:
    """Map mean pair entropy to [0, 1]; lower entropy means higher confidence."""

    return float(max(0.0, min(1.0, 1.0 - mean_entropy / math.log(51))))


def compute_distogram_iptm_proxy(
    distogram_logits,
    target_length: int,
    binder_sequence: str,
    is_antibody: bool | None,
    *,
    cdr_indices: tuple[int, ...] | None = None,
    bin_distance=None,
    torch_module: Any | None = None,
) -> dict[str, float]:
    """Algorithm 15 distogram iPTM proxy from the ESM tutorial.

    The tutorial derives antibody CDR indices through ANARCI. The local design
    loop already owns CDR accounting, so antibody callers must pass CDR indices
    explicitly.
    """

    torch = _torch(torch_module)
    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]

    binder_length = len(binder_sequence)
    if distogram_logits.shape[0] != target_length + binder_length:
        raise AssertionError(
            f"{distogram_logits.shape[0]} != {target_length + binder_length}"
        )

    if bin_distance is None:
        bin_distance = get_mid_points(torch)
        if hasattr(bin_distance, "to"):
            bin_distance = bin_distance.to(distogram_logits.device)
    binder_start = target_length

    def mean_lowest_k(entropies, k: int) -> float:
        sorted_entropies, _indices = torch.sort(entropies.reshape(-1))
        count = (
            int(sorted_entropies.numel())
            if hasattr(sorted_entropies, "numel")
            else int(sorted_entropies.size)
        )
        k = min(k, count)
        return float(sorted_entropies[:k].mean())

    binder_to_target_entropy = binding_confidence_entropy(
        distogram_logits[binder_start:, :target_length, :],
        bin_distance,
        cutoff=22.0,
        torch_module=torch,
    )
    distogram_iptm_proxy = entropy_to_confidence(
        mean_lowest_k(binder_to_target_entropy, k=binder_length)
    )

    if not is_antibody:
        cdr_distogram_iptm_proxy = float("nan")
    else:
        if not cdr_indices:
            raise ValueError("antibody distogram iPTM proxy requires CDR indices")
        cdr_rows = [binder_start + index for index in cdr_indices]
        cdr_to_target_entropy = binding_confidence_entropy(
            distogram_logits[cdr_rows, :target_length, :],
            bin_distance,
            cutoff=22.0,
            torch_module=torch,
        )
        cdr_distogram_iptm_proxy = entropy_to_confidence(
            mean_lowest_k(cdr_to_target_entropy, k=len(cdr_indices))
        )

    return {
        "distogram_iptm_proxy": distogram_iptm_proxy,
        "cdr_distogram_iptm_proxy": cdr_distogram_iptm_proxy,
    }


def _torch(torch_module: Any | None):
    if torch_module is not None:
        return torch_module
    import torch  # type: ignore

    return torch
