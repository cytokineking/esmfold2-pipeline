from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shlex
from typing import Any, Sequence

from esmfold2_pipeline.db import connect_database

DEFAULT_VALIDATION_HOTSPOT_CUTOFF_ANGSTROM = 5.0


@dataclass(frozen=True)
class HotspotTarget:
    protenix_chain_id: str
    source_chain_id: str
    hotspot_indices: tuple[int, ...]


@dataclass(frozen=True)
class ValidationHotspotContext:
    binder_chain_ids: tuple[str, ...]
    target_hotspots: tuple[HotspotTarget, ...]
    contact_cutoff_angstrom: float


@dataclass(frozen=True)
class CifAtom:
    chain_id: str
    residue_index: int | None
    atom_name: str
    element: str
    x: float
    y: float
    z: float


def validation_hotspot_context(
    campaign_dir: str | Path,
    *,
    chain_role_map: dict[str, Sequence[str]],
    contact_cutoff_angstrom: float | None = None,
) -> ValidationHotspotContext | None:
    """Build hotspot context from durable target metadata, if hotspots exist."""

    root = Path(campaign_dir)
    summary_path = root / "target" / "chain_summary.json"
    if not summary_path.exists():
        return None

    summary = _load_json(summary_path)
    chains = summary.get("chains")
    if not isinstance(chains, list):
        return None

    target_chain_ids = tuple(str(chain) for chain in chain_role_map.get("target", ()))
    hotspot_targets: list[HotspotTarget] = []
    for index, chain in enumerate(chains):
        if index >= len(target_chain_ids) or not isinstance(chain, dict):
            continue
        hotspot_indices = tuple(
            int(value)
            for value in chain.get("hotspot_indices", ())
            if _is_int_like(value)
        )
        if not hotspot_indices:
            continue
        hotspot_targets.append(
            HotspotTarget(
                protenix_chain_id=target_chain_ids[index],
                source_chain_id=str(
                    chain.get("canonical_chain_id")
                    or chain.get("auth_asym_id")
                    or chain.get("label_asym_id")
                    or target_chain_ids[index]
                ),
                hotspot_indices=hotspot_indices,
            )
        )

    if not hotspot_targets:
        return None

    cutoff = (
        float(contact_cutoff_angstrom)
        if contact_cutoff_angstrom is not None
        else _hotspot_cutoff_from_campaign(root)
    )
    if cutoff <= 0:
        raise ValueError("validation hotspot contact cutoff must be positive")

    return ValidationHotspotContext(
        binder_chain_ids=tuple(str(chain) for chain in chain_role_map.get("binder", ())),
        target_hotspots=tuple(hotspot_targets),
        contact_cutoff_angstrom=cutoff,
    )


def score_validation_hotspots(
    cif_path: str | Path,
    *,
    context: ValidationHotspotContext,
) -> dict[str, Any]:
    atoms = parse_cif_heavy_atoms(cif_path)
    binder_atoms = [atom for atom in atoms if atom.chain_id in context.binder_chain_ids]
    if not binder_atoms:
        return _hotspot_failure_metrics(
            context,
            "no binder heavy atoms found in Protenix CIF",
        )

    by_chain_residue: dict[tuple[str, int], list[CifAtom]] = {}
    for atom in atoms:
        if atom.residue_index is None:
            continue
        by_chain_residue.setdefault((atom.chain_id, atom.residue_index), []).append(atom)

    by_chain: dict[str, Any] = {}
    min_distance: float | None = None
    missing: list[str] = []
    hotspot_count = 0
    for target in context.target_hotspots:
        chain_distances: list[float] = []
        for hotspot_index in target.hotspot_indices:
            hotspot_count += 1
            target_atoms = by_chain_residue.get((target.protenix_chain_id, hotspot_index))
            if not target_atoms:
                missing.append(f"{target.protenix_chain_id}:{hotspot_index + 1}")
                continue
            distance = _minimum_distance(binder_atoms, target_atoms)
            chain_distances.append(distance)
            min_distance = distance if min_distance is None else min(min_distance, distance)

        chain_min = min(chain_distances) if chain_distances else None
        by_chain[target.protenix_chain_id] = {
            "source_chain_id": target.source_chain_id,
            "hotspot_indices": list(target.hotspot_indices),
            "hotspot_residue_numbers": [index + 1 for index in target.hotspot_indices],
            "hotspot_min_heavy_atom_distance_min": chain_min,
            "hotspot_pass": (
                chain_min is not None
                and chain_min <= context.contact_cutoff_angstrom
            ),
        }

    if min_distance is None:
        return _hotspot_failure_metrics(
            context,
            "no target hotspot heavy atoms found in Protenix CIF",
            by_chain=by_chain,
            missing=missing,
            hotspot_count=hotspot_count,
        )

    passed = min_distance <= context.contact_cutoff_angstrom
    metrics: dict[str, Any] = {
        "validation_hotspot_configured": True,
        "validation_hotspot_scope": "binder_target",
        "validation_hotspot_pass": passed,
        "validation_hotspot_satisfaction": 1.0 if passed else 0.0,
        "validation_hotspot_distance_angstrom": min_distance,
        "validation_hotspot_min_heavy_atom_distance_min": min_distance,
        "validation_hotspot_contact_cutoff_angstrom": context.contact_cutoff_angstrom,
        "validation_hotspot_count": hotspot_count,
        "validation_hotspot_by_chain": by_chain,
    }
    if missing:
        metrics["validation_hotspot_missing_residues"] = missing
    if not passed:
        metrics["validation_hotspot_fail_reason"] = (
            f"validation hotspot distance {min_distance:.4f} exceeds cutoff "
            f"{context.contact_cutoff_angstrom:.4f}"
        )
    return metrics


def parse_cif_heavy_atoms(cif_path: str | Path) -> list[CifAtom]:
    """Parse heavy atoms from a simple mmCIF atom_site loop."""

    lines = Path(cif_path).read_text(errors="ignore").splitlines()
    atoms: list[CifAtom] = []
    for fields, rows in _iter_atom_site_loops(lines):
        index = {field: idx for idx, field in enumerate(fields)}
        for row in rows:
            atom = _atom_from_row(row, index)
            if atom is not None and not _is_hydrogen(atom):
                atoms.append(atom)
    return atoms


def _iter_atom_site_loops(lines: Sequence[str]):
    cursor = 0
    while cursor < len(lines):
        if lines[cursor].strip() != "loop_":
            cursor += 1
            continue
        cursor += 1
        fields: list[str] = []
        while cursor < len(lines):
            text = lines[cursor].strip()
            if text.startswith("_atom_site."):
                fields.append(text.split()[0])
                cursor += 1
                continue
            break
        if not fields:
            continue

        values: list[str] = []
        rows: list[list[str]] = []
        while cursor < len(lines):
            text = lines[cursor].strip()
            if not text or text.startswith("#"):
                cursor += 1
                if values:
                    values = []
                break
            if text == "loop_" or text.startswith("_") or text.startswith("data_"):
                break
            values.extend(shlex.split(text, posix=True))
            while len(values) >= len(fields):
                rows.append(values[: len(fields)])
                values = values[len(fields) :]
            cursor += 1
        if rows:
            yield fields, rows


def _atom_from_row(row: list[str], index: dict[str, int]) -> CifAtom | None:
    chain_id = _first_field(
        row,
        index,
        "_atom_site.label_asym_id",
        "_atom_site.auth_asym_id",
    )
    atom_name = _first_field(
        row,
        index,
        "_atom_site.label_atom_id",
        "_atom_site.auth_atom_id",
    )
    element = _first_field(row, index, "_atom_site.type_symbol") or atom_name[:1]
    x = _float_field(row, index, "_atom_site.Cartn_x")
    y = _float_field(row, index, "_atom_site.Cartn_y")
    z = _float_field(row, index, "_atom_site.Cartn_z")
    if chain_id in (None, "", ".", "?") or atom_name in (None, "", ".", "?"):
        return None
    if x is None or y is None or z is None:
        return None
    residue_seq = _first_field(
        row,
        index,
        "_atom_site.label_seq_id",
        "_atom_site.auth_seq_id",
    )
    residue_index = _residue_index(residue_seq)
    return CifAtom(
        chain_id=str(chain_id),
        residue_index=residue_index,
        atom_name=str(atom_name),
        element=str(element or ""),
        x=x,
        y=y,
        z=z,
    )


def _hotspot_failure_metrics(
    context: ValidationHotspotContext,
    reason: str,
    *,
    by_chain: dict[str, Any] | None = None,
    missing: list[str] | None = None,
    hotspot_count: int | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "validation_hotspot_configured": True,
        "validation_hotspot_scope": "binder_target",
        "validation_hotspot_pass": False,
        "validation_hotspot_satisfaction": 0.0,
        "validation_hotspot_contact_cutoff_angstrom": context.contact_cutoff_angstrom,
        "validation_hotspot_fail_reason": reason,
        "validation_hotspot_count": (
            hotspot_count
            if hotspot_count is not None
            else sum(len(target.hotspot_indices) for target in context.target_hotspots)
        ),
    }
    if by_chain is not None:
        metrics["validation_hotspot_by_chain"] = by_chain
    if missing:
        metrics["validation_hotspot_missing_residues"] = missing
    return metrics


def _hotspot_cutoff_from_campaign(root: Path) -> float:
    db_path = root / "campaign.sqlite"
    if not db_path.exists():
        return DEFAULT_VALIDATION_HOTSPOT_CUTOFF_ANGSTROM
    conn = connect_database(db_path)
    try:
        row = conn.execute(
            "SELECT resolved_config_json FROM campaign WHERE id = 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return DEFAULT_VALIDATION_HOTSPOT_CUTOFF_ANGSTROM
    try:
        resolved = json.loads(row["resolved_config_json"] or "{}")
    except json.JSONDecodeError:
        return DEFAULT_VALIDATION_HOTSPOT_CUTOFF_ANGSTROM
    loss = resolved.get("loss") if isinstance(resolved, dict) else None
    if not isinstance(loss, dict):
        return DEFAULT_VALIDATION_HOTSPOT_CUTOFF_ANGSTROM
    value = loss.get("hotspot_critic_contact_cutoff_angstrom")
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULT_VALIDATION_HOTSPOT_CUTOFF_ANGSTROM


def _minimum_distance(first: Sequence[CifAtom], second: Sequence[CifAtom]) -> float:
    best = math.inf
    for left in first:
        for right in second:
            distance = math.sqrt(
                (left.x - right.x) ** 2
                + (left.y - right.y) ** 2
                + (left.z - right.z) ** 2
            )
            if distance < best:
                best = distance
    return best


def _first_field(row: list[str], index: dict[str, int], *fields: str) -> str | None:
    for field in fields:
        position = index.get(field)
        if position is not None and 0 <= position < len(row):
            value = row[position]
            if value not in {"", ".", "?"}:
                return value
    return None


def _float_field(row: list[str], index: dict[str, int], field: str) -> float | None:
    value = _first_field(row, index, field)
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _residue_index(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(float(value))
    except ValueError:
        return None
    return parsed - 1


def _is_hydrogen(atom: CifAtom) -> bool:
    element = atom.element.strip().upper()
    atom_name = atom.atom_name.strip().upper()
    return element in {"H", "D"} or atom_name.startswith("H")


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value
