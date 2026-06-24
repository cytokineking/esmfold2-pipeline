from __future__ import annotations

import re


_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def shard_id(batch_index: int) -> str:
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    return f"shard_{batch_index:06d}"


def candidate_id(batch_index: int, candidate_index: int) -> str:
    if candidate_index < 0:
        raise ValueError("candidate_index must be non-negative")
    return f"cand_{batch_index:06d}_{candidate_index:04d}"


def semantic_candidate_id(
    *,
    target_name: str | None,
    binder_scaffold: str,
    seed: int,
    candidate_index: int = 0,
) -> str:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if candidate_index < 0:
        raise ValueError("candidate_index must be non-negative")
    stem = (
        f"{slug_identifier(target_name, fallback='target')}"
        f"_{binder_code(binder_scaffold)}"
    )
    stem = f"{stem}_seed{seed}"
    if candidate_index:
        stem = f"{stem}_c{candidate_index}"
    return stem


def slug_identifier(value: str | None, *, fallback: str) -> str:
    if value is None:
        return fallback
    slug = _SLUG_RE.sub("_", value.strip()).strip("_").lower()
    return slug or fallback


def binder_code(binder_scaffold: str) -> str:
    value = slug_identifier(binder_scaffold, fallback="binder")
    if value in {"miniprotein", "minibinder", "mini_binder"}:
        return "mp"
    if value in {"scfv", "sc_fv"}:
        return "scfv"
    return value

