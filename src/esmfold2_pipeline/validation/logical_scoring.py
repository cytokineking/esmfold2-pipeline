from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


LOGICAL_METRIC_SCOPE = "logical_binder_target"
LOGICAL_IPTM_AGGREGATION = "native_expected_tm_cross_role_source_row_max"
LOGICAL_IPSAE_AGGREGATION = "native_ipsae_cross_role_directional_max"


def tm_normalization(residue_count: int) -> float:
    """Return the native AlphaFold/ESMFold TM-score normalization constant."""

    return 1.24 * (max(int(residue_count), 19) - 15) ** (1.0 / 3.0) - 1.8


def esmfold2_pae_bin_centers(num_bins: int) -> np.ndarray:
    """Return the exact ESMFold2 PAE centers spanning [0, 32) Angstrom."""

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    bin_width = np.float32(32.0 / int(num_bins))
    return (np.arange(num_bins, dtype=np.float32) + np.float32(0.5)) * bin_width


def ipsae_normalization(residue_count: int, *, nucleic_acid: bool = False) -> float:
    """Return ipSAE's per-row d0, including its published lower bound."""

    count = max(int(residue_count), 27)
    minimum = 2.0 if nucleic_acid else 1.0
    return max(minimum, 1.24 * (count - 15) ** (1.0 / 3.0) - 1.8)


def logical_role_masks(
    token_asym_ids: Sequence[Any],
    *,
    asym_id_to_chain_id: Mapping[Any, str],
    chain_role_map: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary physical chain identifiers into binder/target token masks."""

    binder_chains = {str(value) for value in chain_role_map.get("binder", ())}
    target_chains = {str(value) for value in chain_role_map.get("target", ())}
    if not binder_chains or not target_chains:
        raise ValueError("logical binder and target roles must both be non-empty")
    overlap = binder_chains & target_chains
    if overlap:
        raise ValueError(f"physical chains cannot have both logical roles: {sorted(overlap)}")

    chain_ids = np.asarray(
        [asym_id_to_chain_id.get(value) for value in token_asym_ids],
        dtype=object,
    )
    if any(value is None for value in chain_ids):
        missing = {
            value
            for value in token_asym_ids
            if asym_id_to_chain_id.get(value) is None
        }
        raise ValueError(f"missing physical-chain mapping for asym ids: {sorted(missing)}")
    binder_mask = np.isin(chain_ids, tuple(binder_chains))
    target_mask = np.isin(chain_ids, tuple(target_chains))
    if not binder_mask.any() or not target_mask.any():
        raise ValueError("logical role mapping selected no binder or target tokens")
    if np.any(binder_mask & target_mask):
        raise ValueError("binder and target token masks must be disjoint")
    if not np.all(binder_mask | target_mask):
        raise ValueError("every non-padding token must have a logical binder or target role")
    return binder_mask, target_mask


def logical_cross_role_tm_score(
    token_pair_tm_expected: Any,
    *,
    binder_mask: Any,
    target_mask: Any,
    source_valid_mask: Any | None = None,
    pair_valid_mask: Any | None = None,
    denominator_epsilon: float = 1e-8,
) -> float | np.ndarray:
    """Apply native ipTM row reduction to logical binder-target interactions.

    Target-column normalization includes every valid residue in the opposite
    logical role.  The maximum is taken only over valid source rows.  Physical
    chain identity is intentionally absent from the calculation.
    """

    values = np.asarray(token_pair_tm_expected, dtype=float)
    if values.ndim not in (2, 3) or values.shape[-1] != values.shape[-2]:
        raise ValueError("token_pair_tm_expected must be a square matrix or batch")
    length = values.shape[-1]
    binder = np.asarray(binder_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if binder.shape != (length,) or target.shape != (length,):
        raise ValueError("logical role masks must match the token-pair matrix")
    if np.any(binder & target) or not binder.any() or not target.any():
        raise ValueError("logical binder and target masks must be non-empty and disjoint")
    if not np.all(binder | target):
        raise ValueError("every scored token must have a logical binder or target role")

    cross_role = (binder[:, None] & target[None, :]) | (
        target[:, None] & binder[None, :]
    )
    if pair_valid_mask is not None:
        pair_valid = np.asarray(pair_valid_mask, dtype=bool)
        if pair_valid.shape == (length,):
            pair_valid = pair_valid[:, None] & pair_valid[None, :]
        if pair_valid.shape != (length, length):
            raise ValueError("pair_valid_mask must be token-length or token-pair shaped")
        cross_role &= pair_valid

    source_valid = (
        np.ones(length, dtype=bool)
        if source_valid_mask is None
        else np.asarray(source_valid_mask, dtype=bool)
    )
    if source_valid.shape != (length,):
        raise ValueError("source_valid_mask must match the token-pair matrix")
    eligible_rows = source_valid & cross_role.any(axis=-1)
    if not eligible_rows.any():
        raise ValueError("no valid logical binder-target source rows")
    if not np.all(np.isfinite(values[..., cross_role])):
        raise ValueError("logical binder-target TM contributions must be finite")

    denominator = cross_role.sum(axis=-1)
    numerator = np.where(cross_role, values, 0.0).sum(axis=-1)
    row_scores = np.divide(
        numerator,
        denominator + float(denominator_epsilon),
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    score = np.max(row_scores[..., eligible_rows], axis=-1)
    return float(score) if values.ndim == 2 else score


def logical_iptm_from_pae_logits(
    pae_logits: Any,
    *,
    binder_mask: Any,
    target_mask: Any,
    valid_token_mask: Any,
):
    """Calculate grouped ESMFold2 ipTM directly from native PAE logits."""

    import torch

    logits = pae_logits
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    if logits.ndim != 4 or logits.shape[-2] != logits.shape[-3]:
        raise ValueError("pae_logits must have shape [batch, token, token, bin]")
    batch_size, length = logits.shape[:2]
    binder = torch.as_tensor(binder_mask, dtype=torch.bool, device=logits.device)
    target = torch.as_tensor(target_mask, dtype=torch.bool, device=logits.device)
    if binder.shape != (length,) or target.shape != (length,):
        raise ValueError("logical role masks must match PAE logits")
    if (
        bool((binder & target).any().item())
        or not bool(binder.any().item())
        or not bool(target.any().item())
    ):
        raise ValueError("logical binder and target masks must be non-empty and disjoint")
    valid = torch.as_tensor(valid_token_mask, dtype=torch.bool, device=logits.device)
    if valid.ndim == 1:
        valid = valid.unsqueeze(0).expand(batch_size, -1)
    if valid.shape != (batch_size, length):
        raise ValueError("valid_token_mask must have shape [token] or [batch, token]")
    unassigned = ~(binder | target)
    if bool((valid & unassigned[None, :]).any().item()):
        raise ValueError("every valid token must have a logical binder or target role")

    num_bins = logits.shape[-1]
    bin_width = 32.0 / num_bins
    bins = torch.arange(
        0.5 * bin_width,
        32.0,
        bin_width,
        dtype=torch.float32,
        device=logits.device,
    )
    lengths = valid.sum(dim=-1).clamp_min(19).to(dtype=torch.float32)
    d0 = 1.24 * (lengths - 15) ** (1.0 / 3.0) - 1.8
    weights = 1.0 / (1.0 + (bins[None, :] / d0[:, None]) ** 2)

    pair_valid = valid[:, :, None] & valid[:, None, :]
    min_value = torch.finfo(logits.dtype).min
    probabilities = logits.masked_fill(~pair_valid[..., None], min_value).softmax(dim=-1)
    expected_tm = (probabilities * weights[:, None, None, :]).sum(dim=-1)

    cross_role = (binder[:, None] & target[None, :]) | (
        target[:, None] & binder[None, :]
    )
    scoring_mask = pair_valid & cross_role[None, ...]
    if bool((~torch.isfinite(expected_tm) & scoring_mask).any().item()):
        raise ValueError("logical binder-target TM contributions must be finite")
    denominator = scoring_mask.sum(dim=-1)
    row_scores = expected_tm.masked_fill(~scoring_mask, 0).sum(dim=-1) / (
        denominator + 1e-5
    )
    eligible = valid & denominator.gt(0)
    row_scores = row_scores.masked_fill(~eligible, torch.finfo(row_scores.dtype).min)
    scores = row_scores.max(dim=-1).values
    if bool((~eligible.any(dim=-1)).any().item()):
        raise ValueError("no valid logical binder-target source rows")
    return scores


def logical_ipsae_from_pae(
    token_pair_pae: Any,
    *,
    binder_mask: Any,
    target_mask: Any,
    pae_cutoff: float,
    source_valid_mask: Any | None = None,
) -> dict[str, Any]:
    """Calculate ipSAE over logical role unions with native directional reduction."""

    pae = np.asarray(token_pair_pae, dtype=float)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError("token_pair_pae must be a square matrix")
    length = pae.shape[0]
    binder = np.asarray(binder_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    source_valid = (
        np.ones(length, dtype=bool)
        if source_valid_mask is None
        else np.asarray(source_valid_mask, dtype=bool)
    )
    if binder.shape != (length,) or target.shape != (length,):
        raise ValueError("logical role masks must match token_pair_pae")
    if source_valid.shape != (length,):
        raise ValueError("source_valid_mask must match token_pair_pae")
    if np.any(binder & target) or not binder.any() or not target.any():
        raise ValueError("logical binder and target masks must be non-empty and disjoint")
    if not np.all(binder | target):
        raise ValueError("every scored token must have a logical binder or target role")
    cross_role = (binder[:, None] & target[None, :]) | (
        target[:, None] & binder[None, :]
    )
    if not np.all(np.isfinite(pae[cross_role])):
        raise ValueError("logical binder-target PAE values must be finite")

    def directional(source: np.ndarray, destination: np.ndarray) -> dict[str, Any]:
        best_score = 0.0
        best_source: int | None = None
        best_count = 0
        best_d0 = ipsae_normalization(0)
        for source_index in np.flatnonzero(source & source_valid):
            selected = destination & np.isfinite(pae[source_index]) & (
                pae[source_index] < float(pae_cutoff)
            )
            count = int(selected.sum())
            if count == 0:
                score = 0.0
                d0 = ipsae_normalization(0)
            else:
                d0 = ipsae_normalization(count)
                score = float(
                    np.mean(
                        1.0 / (1.0 + (pae[source_index, selected] / d0) ** 2)
                    )
                )
            if best_source is None or score > best_score:
                best_score = score
                best_source = int(source_index)
                best_count = count
                best_d0 = d0
        return {
            "value": best_score,
            "source_token_index": best_source,
            "n0res": best_count,
            "d0res": best_d0,
        }

    binder_to_target = directional(binder, target)
    target_to_binder = directional(target, binder)
    best_direction = (
        "binder_to_target"
        if binder_to_target["value"] >= target_to_binder["value"]
        else "target_to_binder"
    )
    return {
        "value": max(binder_to_target["value"], target_to_binder["value"]),
        "min": min(binder_to_target["value"], target_to_binder["value"]),
        "max": max(binder_to_target["value"], target_to_binder["value"]),
        "mean": (binder_to_target["value"] + target_to_binder["value"]) / 2.0,
        "directional_values": {
            "binder_to_target": binder_to_target,
            "target_to_binder": target_to_binder,
        },
        "best_direction": best_direction,
    }
