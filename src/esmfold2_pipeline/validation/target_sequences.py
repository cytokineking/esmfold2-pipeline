from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from esmfold2_pipeline.validation.msa import normalize_sequence


class EffectiveTargetSequenceError(ValueError):
    """Raised when prepared target sequence metadata is present but unusable."""


def effective_target_sequences(
    campaign_dir: str | Path,
    resolved_config: dict[str, Any] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return target sequences and labels used by validation.

    A prepared structural target is authoritative because its chain summary
    records the effective chain order and any configured crop. Configuration
    sequences remain the compatibility fallback for sequence-only targets and
    campaigns that have not prepared structure artifacts yet.
    """

    root = Path(campaign_dir)
    summary_path = root / "target" / "chain_summary.json"
    if summary_path.exists():
        return _prepared_target_sequences(summary_path)
    return configured_target_sequences(resolved_config or {})


def _prepared_target_sequences(
    summary_path: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        payload = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EffectiveTargetSequenceError(
            f"prepared target chain summary is unreadable: {summary_path}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("chains"), list):
        raise EffectiveTargetSequenceError(
            "prepared target chain summary must contain a chains list"
        )
    chains = payload["chains"]
    if not chains:
        raise EffectiveTargetSequenceError(
            "prepared target chain summary contains no chains"
        )

    sequences: list[str] = []
    labels: list[str] = []
    for index, chain in enumerate(chains):
        if not isinstance(chain, dict):
            raise EffectiveTargetSequenceError(
                f"prepared target chain summary entry {index} is not an object"
            )
        sequence = chain.get("sequence")
        label = (
            chain.get("canonical_chain_id")
            or chain.get("auth_asym_id")
            or chain.get("label_asym_id")
        )
        if not isinstance(sequence, str) or not sequence.strip():
            raise EffectiveTargetSequenceError(
                f"prepared target chain summary entry {index} has no sequence"
            )
        if not isinstance(label, str) or not label.strip():
            raise EffectiveTargetSequenceError(
                f"prepared target chain summary entry {index} has no chain ID"
            )
        normalized = _normalize_chain_sequence(
            sequence,
            context=f"prepared target chain {label!r}",
        )
        declared_length = chain.get("length")
        if declared_length is not None and (
            isinstance(declared_length, bool)
            or not isinstance(declared_length, int)
            or declared_length != len(normalized)
        ):
            raise EffectiveTargetSequenceError(
                "prepared target chain summary entry "
                f"{index} length does not match its sequence"
            )
        normalized_label = label.strip()
        if normalized_label in labels:
            raise EffectiveTargetSequenceError(
                f"prepared target chain summary repeats canonical chain ID {normalized_label!r}"
            )
        sequences.append(normalized)
        labels.append(normalized_label)
    return tuple(sequences), tuple(labels)


def configured_target_sequences(
    resolved_config: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    target = resolved_config.get("target")
    if not isinstance(target, dict):
        return (), ()

    direct = target.get("sequence")
    if isinstance(direct, str) and direct.strip():
        return (
            _normalize_chain_sequence(direct, context="configured target sequence"),
        ), ("B",)

    sequences = target.get("sequences")
    chains = target.get("chains")
    if not isinstance(sequences, dict) or not isinstance(chains, list):
        return (), ()
    out: list[str] = []
    labels: list[str] = []
    for chain in chains:
        sequence = sequences.get(chain)
        if not isinstance(sequence, str) or not sequence.strip():
            continue
        out.append(
            _normalize_chain_sequence(
                sequence,
                context=f"configured target chain {chain!r}",
            )
        )
        labels.append(str(chain))
    return tuple(out), tuple(labels)


def _normalize_chain_sequence(sequence: str, *, context: str) -> str:
    normalized = normalize_sequence(sequence)
    if not normalized:
        raise EffectiveTargetSequenceError(f"{context} is empty")
    if "|" in normalized:
        raise EffectiveTargetSequenceError(
            f"{context} must describe one chain, not a chain-delimited sequence"
        )
    return normalized
