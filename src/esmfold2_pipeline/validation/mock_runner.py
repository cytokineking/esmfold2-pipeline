from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
from typing import Any

from esmfold2_pipeline.artifact_layout import (
    VALIDATION_PASSING_DIR,
    VALIDATION_REJECTED_DIR,
    VALIDATION_STAGING_DIR,
    validator_slug,
    validator_structure_status_dir,
)
from esmfold2_pipeline.artifacts import write_text_atomic
from esmfold2_pipeline.db import CampaignStore, ValidationClaim, initialize_database

MOCK_VALIDATION_MODEL = "mock-protenix-v2"


@dataclass(frozen=True)
class MockValidationRunResult:
    completed_tasks: int
    recorded_structures: int
    skipped_no_pending: bool


@dataclass(frozen=True)
class _ValidationTaskDetails:
    validation_id: str
    candidate_id: str
    model_name: str
    validation_config_hash: str
    needs_config_suffix: bool
    selection_rank: int | None
    seed: int
    designed_sequence: str
    binder_scaffold: str | None
    framework: str | None


@dataclass(frozen=True)
class _MockStructure:
    structure_id: str
    sample_rank: int
    status: str
    metrics: dict[str, Any]
    cif_text: str


def run_mock_validation(
    campaign_dir: str | Path,
    *,
    worker_id: str = "mock-validation-worker-0",
    gpu_id: str | None = None,
    max_tasks: int | None = None,
) -> MockValidationRunResult:
    if max_tasks is not None and max_tasks < 0:
        raise ValueError("max_tasks must be non-negative")

    root = Path(campaign_dir)
    conn = initialize_database(root / "campaign.sqlite")
    store = CampaignStore(conn)
    completed = 0
    recorded_structures = 0
    try:
        while max_tasks is None or completed < max_tasks:
            claims = store.claim_next_pending_validation_tasks(
                worker_id=worker_id,
                batch_size=1,
                hostname=socket.gethostname(),
                pid=os.getpid(),
                gpu_id=gpu_id,
            )
            if not claims:
                break
            claim = claims[0]
            try:
                details = _fetch_task_details(conn, claim)
                structures = _mock_structures(details)
                best = max(
                    structures,
                    key=lambda item: (
                        item.status == "passing",
                        float(item.metrics.get("validation_iptm") or -1.0),
                        float(item.metrics.get("validation_ipSAE") or -1.0),
                        float(item.metrics.get("ranking_score") or -1.0),
                    ),
                )
                best_path: str | None = None
                for structure in structures:
                    relpath = _publish_structure(
                        root=root,
                        store=store,
                        details=details,
                        structure=structure,
                    )
                    recorded_structures += 1
                    if structure.structure_id == best.structure_id:
                        best_path = relpath
                if best_path is None:
                    raise RuntimeError("mock validation did not publish a best structure")
                store.complete_validation_task(
                    validation_id=claim.validation_id,
                    attempt_id=claim.attempt_id,
                    output_structure_path=best_path,
                    metrics={
                        **best.metrics,
                        "best_structure_id": best.structure_id,
                        "structure_count": len(structures),
                    },
                    runtime_seconds=0.0,
                )
                completed += 1
            except Exception as exc:
                store.fail_validation_task(
                    validation_id=claim.validation_id,
                    attempt_id=claim.attempt_id,
                    error_message=str(exc),
                )
                raise
    finally:
        conn.close()

    return MockValidationRunResult(
        completed_tasks=completed,
        recorded_structures=recorded_structures,
        skipped_no_pending=completed == 0,
    )


def _fetch_task_details(conn, claim: ValidationClaim) -> _ValidationTaskDetails:
    row = conn.execute(
        """
        SELECT
            vt.validation_id,
            vt.candidate_id,
            vt.model_name,
            vt.validation_config_hash,
            vt.selection_rank,
            c.designed_sequence,
            c.design_metrics_json,
            s.seed,
            (
                SELECT COUNT(DISTINCT vt2.validation_config_hash)
                FROM validation_tasks AS vt2
                WHERE vt2.candidate_id = vt.candidate_id
                  AND vt2.model_name = vt.model_name
            ) AS validation_config_variant_count
        FROM validation_tasks AS vt
        JOIN candidates AS c
          ON c.candidate_id = vt.candidate_id
        JOIN shards AS s
          ON s.shard_id = c.shard_id
        WHERE vt.validation_id = ?
        """,
        (claim.validation_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"unknown validation task: {claim.validation_id}")
    design_metrics = json.loads(row["design_metrics_json"] or "{}")
    return _ValidationTaskDetails(
        validation_id=row["validation_id"],
        candidate_id=row["candidate_id"],
        model_name=row["model_name"],
        validation_config_hash=row["validation_config_hash"],
        needs_config_suffix=int(row["validation_config_variant_count"] or 0) > 1,
        selection_rank=(
            int(row["selection_rank"])
            if row["selection_rank"] is not None
            else None
        ),
        seed=int(row["seed"]),
        designed_sequence=row["designed_sequence"],
        binder_scaffold=design_metrics.get("binder_scaffold"),
        framework=design_metrics.get("framework") or design_metrics.get("framework_name"),
    )


def _mock_structures(details: _ValidationTaskDetails) -> list[_MockStructure]:
    base_iptm = 0.8 + (details.seed * 0.001)
    base_ipsae = 0.6 + (details.seed * 0.001)
    common = {
        "validation_model": details.model_name,
        "validation_metric_scope": "binder_target",
        "validation_chain_role_map": {"binder": ["A"], "target": ["B"]},
        "validation_global_iptm": 0.95,
        "binder_scaffold": details.binder_scaffold,
        "framework": details.framework,
    }
    passing_metrics = {
        **common,
        "validation_iptm": round(base_iptm, 4),
        "validation_ipSAE": round(base_ipsae, 4),
        "ptm": 0.55,
        "ranking_score": 0.71,
        "hotspot_satisfaction": 1.0,
        "pass_reason": "mock structure passes scoped validation thresholds",
    }
    rejected_metrics = {
        **common,
        "validation_iptm": 0.21,
        "validation_ipSAE": 0.12,
        "ptm": 0.25,
        "ranking_score": 0.19,
        "hotspot_satisfaction": 0.0,
        "fail_reason": "mock rejected lower-confidence sample",
    }
    return [
        _MockStructure(
            structure_id=f"seed{details.seed}_sample0",
            sample_rank=0,
            status="passing",
            metrics=passing_metrics,
            cif_text=_mock_cif(details, sample_rank=0, status="passing"),
        ),
        _MockStructure(
            structure_id=f"seed{details.seed}_sample1",
            sample_rank=1,
            status="rejected",
            metrics=rejected_metrics,
            cif_text=_mock_cif(details, sample_rank=1, status="rejected"),
        ),
    ]


def _publish_structure(
    *,
    root: Path,
    store: CampaignStore,
    details: _ValidationTaskDetails,
    structure: _MockStructure,
) -> str:
    stem = _structure_stem(details, structure)
    pending_relpath = validator_structure_status_dir(
        details.model_name,
        VALIDATION_STAGING_DIR,
    ) / f"{stem}.cif"
    final_relpath = validator_structure_status_dir(
        details.model_name,
        _status_dir(structure.status),
    ) / f"{stem}.cif"

    write_text_atomic(root / pending_relpath, structure.cif_text)
    store.record_validation_structure(
        validation_id=details.validation_id,
        structure_id=structure.structure_id,
        candidate_id=details.candidate_id,
        model_name=details.model_name,
        seed=details.seed,
        sample_rank=structure.sample_rank,
        status="pending",
        structure_path=pending_relpath.as_posix(),
        metrics=structure.metrics,
    )

    write_text_atomic(root / final_relpath, structure.cif_text)
    store.record_validation_structure(
        validation_id=details.validation_id,
        structure_id=structure.structure_id,
        candidate_id=details.candidate_id,
        model_name=details.model_name,
        seed=details.seed,
        sample_rank=structure.sample_rank,
        status=structure.status,
        structure_path=final_relpath.as_posix(),
        metrics=structure.metrics,
    )
    try:
        (root / pending_relpath).unlink()
    except FileNotFoundError:
        pass
    return final_relpath.as_posix()


def _structure_stem(
    details: _ValidationTaskDetails,
    structure: _MockStructure,
) -> str:
    candidate = _safe_path_part(details.candidate_id)
    config_part = (
        f"__cfg-{_safe_path_part(details.validation_config_hash)[:12]}"
        if details.needs_config_suffix
        else ""
    )
    return (
        f"{candidate}__{validator_slug(details.model_name)}"
        f"{config_part}__sample{structure.sample_rank:02d}"
    )


def _status_dir(status: str) -> str:
    if status == "passing":
        return VALIDATION_PASSING_DIR
    if status == "rejected":
        return VALIDATION_REJECTED_DIR
    raise ValueError(f"unsupported validation structure status: {status}")


def _safe_path_part(value: str) -> str:
    chars = [
        char.lower() if char.isalnum() else "_"
        for char in str(value).strip()
    ]
    text = "".join(chars).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "value"


def _mock_cif(
    details: _ValidationTaskDetails,
    *,
    sample_rank: int,
    status: str,
) -> str:
    data_id = _safe_path_part(f"{details.candidate_id}_{sample_rank}")
    return (
        f"data_{data_id}\n"
        "#\n"
        f"_entry.id {data_id}\n"
        f"_audit.creation_method 'esmfold2-pipeline mock validation {status}'\n"
        "#\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "ATOM 1 C CA GLY A 1 0.000 0.000 0.000\n"
        "ATOM 2 C CA GLY B 1 3.000 0.000 0.000\n"
        "#\n"
    )
