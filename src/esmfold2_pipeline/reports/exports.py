from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
import sqlite3
from typing import Any, Literal

import yaml

from esmfold2_pipeline.artifact_layout import (
    ANALYSIS_COMBINED_RANKING_CSV,
    ANALYSIS_DIR,
    ANALYSIS_PLOTS_DIR,
    ANALYSIS_RANKING_DIAGNOSTICS_CSV,
    ANALYSIS_RANKING_SUMMARY_JSON,
    ANALYSIS_TOP_RANKED_DIR,
    ESMFOLD2_DIR,
    ESMFOLD2_METRICS_CSV,
    ESMFOLD2_SELECTED_DESIGNS_CSV,
    ESMFOLD2_SELECTED_MANIFEST_CSV,
    ESMFOLD2_SELECTED_STRUCTURES_DIR,
    ESMFOLD2_SUMMARY_JSON,
    VALIDATION_RESULTS_CSV,
    VALIDATION_STRUCTURE_SAMPLES_CSV,
    VALIDATION_SUMMARY_JSON,
    VALIDATION_PASSING_DIR,
    validator_dir,
    validator_slug,
)
from esmfold2_pipeline.artifacts import (
    write_bytes_atomic,
    write_json_atomic,
    write_text_atomic,
)
from esmfold2_pipeline.config import (
    ANALYSIS_RANKING_MODES,
    DEFAULT_ANALYSIS_MAX_BINDER_RMSD_ANGSTROM,
    DEFAULT_ANALYSIS_RANKING_MODE,
    DEFAULT_ANALYSIS_RMSD_WEIGHT,
    DEFAULT_ANALYSIS_TOP_K,
    AnalysisRankingConfig,
)
from esmfold2_pipeline.db import connect_database
from esmfold2_pipeline.planning import binder_code
from esmfold2_pipeline.reports.status import inspect_campaign

AGGREGATE_CSV = ESMFOLD2_METRICS_CSV
RANKED_CSV = ESMFOLD2_SELECTED_DESIGNS_CSV
SUMMARY_JSON = ESMFOLD2_SUMMARY_JSON
SELECTED_MANIFEST_CSV = ESMFOLD2_SELECTED_MANIFEST_CSV
VALIDATION_MANIFEST_CSV = VALIDATION_RESULTS_CSV
VALIDATION_STRUCTURES_MANIFEST_CSV = VALIDATION_STRUCTURE_SAMPLES_CSV
RMSD_COLORBAR_MAX_ANGSTROM = 20.0

HOTSPOT_FIELDS = [
    "hotspot_pass",
    "hotspot_distance_angstrom",
]

TARGET_GEOMETRY_DRIFT_FIELDS = [
    "target_geometry_drift_distance_rmse",
    "target_geometry_drift_aligned_rmsd",
]

MOSAIC_CDR_FIELDS = [
    "binder_target_contact_mode",
    "mosaic_cdr_contact_loss_enabled",
    "mosaic_cdr_contact_weight",
    "mosaic_cdr_contact_cutoff_angstrom",
    "mosaic_cdr_num_target_contacts",
    "mosaic_cdr_contact_scope",
    "mosaic_cdr_contact_probability_mean",
    "mosaic_cdr_contact_probability_min",
    "mosaic_cdr_contact_probability_max",
    "mosaic_cdr_contact_loss",
    "mosaic_framework_contact_penalty_enabled",
    "mosaic_framework_contact_penalty_weight",
    "mosaic_framework_contact_penalty_cutoff_angstrom",
    "mosaic_framework_contact_penalty_scope",
    "mosaic_framework_contact_penalty_target_scope",
    "mosaic_framework_contact_probability_threshold",
    "mosaic_framework_contact_probability_mean",
    "mosaic_framework_contact_probability_max",
    "mosaic_framework_contact_penalty_loss",
]

SCFV_CDR_FIELDS = [
    "cdrh1",
    "cdrh2",
    "cdrh3",
    "cdrl1",
    "cdrl2",
    "cdrl3",
]
VHH_CDR_FIELDS = [
    "hcdr1",
    "hcdr2",
    "hcdr3",
]
CDR_FIELDS = [*VHH_CDR_FIELDS, *SCFV_CDR_FIELDS]

AGGREGATE_FIELDS = [
    "candidate_id",
    "seed",
    "designed_sequence",
    "binder_length",
    "binder_chain_id",
    "iptm",
    "distogram_iptm_proxy",
    "ptm",
    "plddt_complex",
    "plddt_binder",
    "plddt_target",
    "final_loss",
    "iptm_scope",
    "complex_iptm",
    "critic_name",
    "structure_path",
]

RANKED_FIELDS = ["rank", *AGGREGATE_FIELDS]

MANIFEST_FIELDS = [
    "rank",
    "candidate_id",
    "seed",
    "designed_sequence",
    "binder_length",
    "binder_chain_id",
    "iptm",
    "distogram_iptm_proxy",
    "ptm",
    "plddt_complex",
    "plddt_binder",
    "plddt_target",
    "final_loss",
    "iptm_scope",
    "complex_iptm",
    "critic_name",
    "structure_file",
    "source_structure_path",
]

POSE_AGREEMENT_FIELDS = [
    "target_aligned_rmsd",
    "binder_ca_rmsd_after_target_alignment",
    "binder_backbone_rmsd_after_target_alignment",
    "binder_centroid_distance_after_target_alignment",
]
POSE_AGREEMENT_ERROR_FIELD = "pose_agreement_error"

VALIDATION_MANIFEST_FIELDS = [
    "validation_rank",
    "candidate_id",
    "selection_rank",
    "seed",
    "designed_sequence",
    "binder_scaffold",
    "framework",
    "framework_source",
    "binder_length",
    "binder_chain_id",
    "esm_critic_name",
    "esm_iptm",
    "esm_distogram_iptm_proxy",
    "esm_hotspot_pass",
    "esm_hotspot_distance_angstrom",
    "validator_model",
    "validator_passed",
    "validator_iptm",
    "validator_ipsae",
    "min_validator_ipsae",
    "validator_ipsae_pass",
    "validator_ptm",
    "validator_ranking_score",
    "validator_global_iptm",
    "validator_hotspot_pass",
    "validator_hotspot_distance_angstrom",
    *POSE_AGREEMENT_FIELDS,
    POSE_AGREEMENT_ERROR_FIELD,
    "structure_count",
    "best_structure_id",
    "pass_reason",
    "fail_reason",
    "error_message",
    "completed_at",
    "validator_runtime_seconds",
    "source_structure_path",
    "validated_structure_path",
    "validation_id",
    "validator_config_hash",
    "validator_metric_scope",
]

VALIDATION_STRUCTURES_MANIFEST_FIELDS = [
    "candidate_id",
    "structure_id",
    "sample_rank",
    "structure_status",
    "validator_model",
    "validator_iptm",
    "validator_ipsae",
    "min_validator_ipsae",
    "validator_ipsae_pass",
    "validator_ptm",
    "validator_ranking_score",
    "validator_global_iptm",
    "validator_hotspot_pass",
    "validator_hotspot_distance_angstrom",
    "validator_passed",
    *POSE_AGREEMENT_FIELDS,
    POSE_AGREEMENT_ERROR_FIELD,
    "selection_rank",
    "seed",
    "designed_sequence",
    "binder_scaffold",
    "framework",
    "binder_length",
    "pass_reason",
    "fail_reason",
    "created_at",
    "source_structure_path",
    "structure_path",
    "validation_id",
    "validator_config_hash",
    "validator_metric_scope",
]

RANKING_DIAGNOSTIC_FIELDS = [
    "final_rank",
    "validator_rank",
    "ranking_eligible",
    "ranking_exclusion_reason",
    "ranking_mode",
    "ranking_metric_basis",
    "confidence_score",
    "evaluator_score",
    "agreement_score",
    "final_score",
    "pareto_front",
    "rmsd_pass",
    *[field for field in VALIDATION_MANIFEST_FIELDS if field != "validation_rank"],
    "copied_esmfold2_structure",
    "copied_validator_structure",
]

COMPACT_RANKING_BASE_FIELDS = [
    "rank",
    "design_name",
    "sequence",
    "binder_length",
    "consensus_score",
    "esmfold2_rank",
    "esmfold2_iptm",
    "validator_rank",
    "validator_iptm",
    "validator_ipsae",
    "binder_rmsd_angstrom",
    "esmfold2_hotspot_distance_angstrom",
    "validator_hotspot_distance_angstrom",
    "esmfold2_structure",
    "validator_structure",
]


@dataclass(frozen=True)
class AggregateResult:
    metrics_csv: Path
    summary_json: Path
    row_count: int


@dataclass(frozen=True)
class SelectResult:
    ranked_csv: Path
    summary_json: Path
    selected_count: int
    candidate_count: int


@dataclass(frozen=True)
class ExportResult:
    selected_dir: Path
    manifest_csv: Path
    summary_json: Path
    selected_count: int
    copied_files: int


@dataclass(frozen=True)
class ValidationModelReport:
    model_name: str
    validated_dir: Path
    manifest_csv: Path
    structures_manifest_csv: Path
    summary_json: Path
    task_rows: int
    structure_rows: int


@dataclass(frozen=True)
class ValidationReportResult:
    model_reports: tuple[ValidationModelReport, ...]

    def __post_init__(self) -> None:
        if not self.model_reports:
            raise ValueError("model_reports cannot be empty")

    @property
    def validated_dir(self) -> Path:
        return self._single_model_report().validated_dir

    @property
    def manifest_csv(self) -> Path:
        return self._single_model_report().manifest_csv

    @property
    def structures_manifest_csv(self) -> Path:
        return self._single_model_report().structures_manifest_csv

    @property
    def summary_json(self) -> Path:
        return self._single_model_report().summary_json

    @property
    def task_rows(self) -> int:
        return sum(report.task_rows for report in self.model_reports)

    @property
    def structure_rows(self) -> int:
        return sum(report.structure_rows for report in self.model_reports)

    def _single_model_report(self) -> ValidationModelReport:
        if len(self.model_reports) != 1:
            raise ValueError("multiple validator reports available; use model_reports")
        return self.model_reports[0]


@dataclass(frozen=True)
class AnalysisResult:
    combined_ranking_csv: Path
    diagnostics_csv: Path
    summary_json: Path
    plots_dir: Path
    top_ranked_dir: Path
    ranked_count: int
    diagnostic_count: int
    copied_designs: int


def aggregate_campaign(campaign_dir: str | Path) -> AggregateResult:
    """Write complete candidate/critic rows into the ESMFold2 report folder."""

    root = Path(campaign_dir)
    esmfold2_dir = root / ESMFOLD2_DIR
    rows = _completed_metric_rows(root)

    metrics_csv = esmfold2_dir / AGGREGATE_CSV
    write_text_atomic(
        metrics_csv,
        _csv_text(
            _report_fields(root, AGGREGATE_FIELDS, rows),
            rows,
            metadata=_campaign_csv_metadata(root, rows),
        ),
    )
    summary_json = _write_campaign_summary(root, metric_rows=rows)

    return AggregateResult(
        metrics_csv=metrics_csv,
        summary_json=summary_json,
        row_count=len(rows),
    )


def select_campaign(
    campaign_dir: str | Path,
    *,
    max_designs: int = 100,
    min_iptm: float | None = None,
    require_hotspot_contact: Literal["auto", "always", "never"] = "auto",
) -> SelectResult:
    """Deduplicate by exact sequence and write a deterministic ranked CSV."""

    if max_designs <= 0:
        raise ValueError("max_designs must be positive")
    if require_hotspot_contact not in {"auto", "always", "never"}:
        raise ValueError("require_hotspot_contact must be one of: auto, always, never")

    root = Path(campaign_dir)
    rows = _completed_metric_rows(root)
    if min_iptm is not None:
        rows = [
            row
            for row in rows
            if row["iptm"] is not None and float(row["iptm"]) >= min_iptm
        ]

    if _should_require_hotspot_contact(root, rows, require_hotspot_contact):
        rows = [row for row in rows if _row_has_hotspot_contact(row)]

    deduped = _deduplicate_by_sequence(rows)
    ranked = sorted(deduped, key=_sort_key)[:max_designs]
    output_rows = [
        {"rank": index, **row}
        for index, row in enumerate(ranked, start=1)
    ]

    ranked_csv = root / ESMFOLD2_DIR / RANKED_CSV
    write_text_atomic(
        ranked_csv,
        _csv_text(
            _report_fields(root, RANKED_FIELDS, output_rows),
            output_rows,
            metadata=_campaign_csv_metadata(root, rows),
        ),
    )
    summary_json = _write_campaign_summary(
        root,
        metric_rows=rows,
        ranked_rows=output_rows,
        selection={
            "candidate_pool": len(deduped),
            "selected_count": len(output_rows),
            "max_designs": max_designs,
            "min_iptm": min_iptm,
            "require_hotspot_contact": require_hotspot_contact,
        },
    )

    return SelectResult(
        ranked_csv=ranked_csv,
        summary_json=summary_json,
        selected_count=len(output_rows),
        candidate_count=len(deduped),
    )


def export_campaign(
    campaign_dir: str | Path,
    *,
    max_designs: int | None = None,
) -> ExportResult:
    """Copy selected ESMFold2 structures into selected_structures with a manifest."""

    if max_designs is not None and max_designs <= 0:
        raise ValueError("max_designs must be positive")

    root = Path(campaign_dir)
    ranked_csv = root / ESMFOLD2_DIR / RANKED_CSV
    if not ranked_csv.exists():
        raise FileNotFoundError(
            f"missing ranked CSV: {ranked_csv}; run select before export"
        )

    rows = _read_ranked_csv(ranked_csv)
    if max_designs is not None:
        rows = rows[:max_designs]
    cdr_fields = _cdr_fields_for_rows(rows)

    selected_dir = root / ESMFOLD2_SELECTED_STRUCTURES_DIR
    _clear_generated_selection(selected_dir)

    manifest_rows: list[dict[str, Any]] = []
    copied_files = 0
    for row in rows:
        rank = int(row["rank"])

        structure_src = _artifact_path(root, str(row["structure_path"]))

        structure_name = structure_src.name

        structure_dst = selected_dir / structure_name
        write_bytes_atomic(structure_dst, structure_src.read_bytes())
        copied_files += 1

        manifest_rows.append(
            {
                "rank": rank,
                "candidate_id": row["candidate_id"],
                "framework": row.get("framework"),
                "seed": row["seed"],
                "designed_sequence": row["designed_sequence"],
                **{field: row.get(field) for field in cdr_fields},
                "binder_length": row["binder_length"],
                "binder_chain_id": row["binder_chain_id"],
                "iptm": row["iptm"],
                "cdr_distogram_iptm_proxy": row.get("cdr_distogram_iptm_proxy"),
                "distogram_iptm_proxy": row["distogram_iptm_proxy"],
                "ptm": row["ptm"],
                "plddt_complex": row["plddt_complex"],
                "plddt_binder": row["plddt_binder"],
                "plddt_target": row["plddt_target"],
                "hotspot_pass": row.get("hotspot_pass"),
                "hotspot_distance_angstrom": row.get("hotspot_distance_angstrom"),
                "target_geometry_drift_distance_rmse": row.get(
                    "target_geometry_drift_distance_rmse"
                ),
                "target_geometry_drift_aligned_rmsd": row.get(
                    "target_geometry_drift_aligned_rmsd"
                ),
                "final_loss": row["final_loss"],
                "iptm_scope": row["iptm_scope"],
                "complex_iptm": row["complex_iptm"],
                "critic_name": row["critic_name"],
                "structure_file": structure_name,
                "source_structure_path": row["structure_path"],
            }
        )

    manifest_csv = selected_dir / SELECTED_MANIFEST_CSV
    write_text_atomic(
        manifest_csv,
        _csv_text(
            _report_fields(root, MANIFEST_FIELDS, manifest_rows),
            manifest_rows,
            metadata=_campaign_csv_metadata(root, rows),
        ),
    )
    summary_json = _write_campaign_summary(
        root,
        metric_rows=_completed_metric_rows(root),
        ranked_rows=rows,
        export={
            "selected_dir": selected_dir.relative_to(root).as_posix(),
            "manifest_csv": manifest_csv.relative_to(root).as_posix(),
            "selected_count": len(rows),
            "copied_files": copied_files,
        },
    )

    return ExportResult(
        selected_dir=selected_dir,
        manifest_csv=manifest_csv,
        summary_json=summary_json,
        selected_count=len(rows),
        copied_files=copied_files,
    )


def report_validation(campaign_dir: str | Path) -> ValidationReportResult:
    """Write validation manifests and summary from SQLite and promoted CIFs."""

    root = Path(campaign_dir)
    task_rows = _validation_task_rows(root)
    structure_rows = _validation_structure_rows(root)
    reports = tuple(
        _write_validation_model_report(
            root,
            model_name=model_name,
            task_rows=[
                row for row in task_rows if _row_validator_model(row) == model_name
            ],
            structure_rows=[
                row for row in structure_rows if _row_validator_model(row) == model_name
            ],
        )
        for model_name in _validation_report_model_names(task_rows, structure_rows)
    )

    return ValidationReportResult(model_reports=reports)


def _write_validation_model_report(
    root: Path,
    *,
    model_name: str,
    task_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
) -> ValidationModelReport:
    validated_dir = root / validator_dir(model_name)

    ranked_tasks = [
        {"validation_rank": index, **row}
        for index, row in enumerate(
            sorted(task_rows, key=_validation_task_sort_key),
            start=1,
        )
    ]
    ranked_structures = sorted(structure_rows, key=_validation_structure_sort_key)

    manifest_csv = validated_dir / VALIDATION_MANIFEST_CSV
    write_text_atomic(
        manifest_csv,
        _csv_text(
            _validation_report_fields(VALIDATION_MANIFEST_FIELDS, ranked_tasks),
            ranked_tasks,
        ),
    )

    structures_manifest_csv = validated_dir / VALIDATION_STRUCTURES_MANIFEST_CSV
    write_text_atomic(
        structures_manifest_csv,
        _csv_text(
            _validation_report_fields(
                VALIDATION_STRUCTURES_MANIFEST_FIELDS,
                ranked_structures,
            ),
            ranked_structures,
        ),
    )

    summary_json = write_json_atomic(
        validated_dir / VALIDATION_SUMMARY_JSON,
        _validation_summary(
            root,
            task_rows=ranked_tasks,
            structure_rows=ranked_structures,
            manifest_csv=manifest_csv,
            structures_manifest_csv=structures_manifest_csv,
        ),
    )

    return ValidationModelReport(
        model_name=model_name,
        validated_dir=validated_dir,
        manifest_csv=manifest_csv,
        structures_manifest_csv=structures_manifest_csv,
        summary_json=summary_json,
        task_rows=len(ranked_tasks),
        structure_rows=len(ranked_structures),
    )


def analyze_campaign(
    campaign_dir: str | Path,
    *,
    top_k: int | None = None,
    max_binder_rmsd_angstrom: float | None = None,
    rmsd_weight: float | None = None,
) -> AnalysisResult:
    """Rank validated designs and copy top-k paired structures for inspection."""

    root = Path(campaign_dir)
    if top_k is None:
        top_k = _analysis_top_k(root)
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    analysis_dir = root / ANALYSIS_DIR
    plots_dir = analysis_dir / ANALYSIS_PLOTS_DIR
    top_ranked_dir = analysis_dir / ANALYSIS_TOP_RANKED_DIR
    task_rows = _validation_task_rows(root)
    ranking_config = _analysis_ranking_config(
        root,
        max_binder_rmsd_angstrom=max_binder_rmsd_angstrom,
        rmsd_weight=rmsd_weight,
    )
    scored_rows = _score_analysis_rows(task_rows, config=ranking_config)
    ranked_rows, diagnostic_rows = _finalize_analysis_rows(scored_rows)

    copied_designs = _write_top_ranked_structures(
        root,
        top_ranked_dir=top_ranked_dir,
        ranked_rows=ranked_rows,
        top_k=top_k,
    )
    combined_ranking_csv = analysis_dir / ANALYSIS_COMBINED_RANKING_CSV
    compact_rows = [_compact_ranking_row(row) for row in ranked_rows]
    write_text_atomic(
        combined_ranking_csv,
        _csv_text(
            _compact_ranking_fields(compact_rows),
            compact_rows,
        ),
    )
    diagnostics_csv = analysis_dir / ANALYSIS_RANKING_DIAGNOSTICS_CSV
    write_text_atomic(
        diagnostics_csv,
        _csv_text(
            _validation_report_fields(
                RANKING_DIAGNOSTIC_FIELDS,
                diagnostic_rows,
            ),
            diagnostic_rows,
        ),
    )
    plot_paths, plot_warnings = _write_analysis_plots(
        plots_dir=plots_dir,
        ranked_rows=diagnostic_rows,
    )
    summary_json = write_json_atomic(
        analysis_dir / ANALYSIS_RANKING_SUMMARY_JSON,
        _analysis_summary(
            root,
            ranked_rows=ranked_rows,
            diagnostic_rows=diagnostic_rows,
            combined_ranking_csv=combined_ranking_csv,
            diagnostics_csv=diagnostics_csv,
            top_ranked_dir=top_ranked_dir,
            top_k=top_k,
            copied_designs=copied_designs,
            plot_paths=plot_paths,
            plot_warnings=plot_warnings,
            ranking_config=ranking_config,
        ),
    )

    return AnalysisResult(
        combined_ranking_csv=combined_ranking_csv,
        diagnostics_csv=diagnostics_csv,
        summary_json=summary_json,
        plots_dir=plots_dir,
        top_ranked_dir=top_ranked_dir,
        ranked_count=len(ranked_rows),
        diagnostic_count=len(diagnostic_rows),
        copied_designs=copied_designs,
    )


def _validation_report_model_names(
    task_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
) -> tuple[str, ...]:
    models = sorted(
        {
            model
            for row in [*task_rows, *structure_rows]
            if (model := _row_validator_model(row))
        }
    )
    if not models:
        return ("protenix-v2",)
    return tuple(models)


def _row_validator_model(row: dict[str, Any]) -> str:
    return str(row.get("validator_model") or row.get("validation_model") or "")


def _analysis_top_k(root: Path) -> int:
    metadata = _campaign_metadata(root)
    analysis = metadata.get("analysis")
    if isinstance(analysis, dict):
        value = analysis.get("top_k")
        parsed = _optional_int_value(value)
        if parsed is not None and parsed > 0:
            return parsed
    return DEFAULT_ANALYSIS_TOP_K


def _analysis_ranking_config(
    root: Path,
    *,
    max_binder_rmsd_angstrom: float | None,
    rmsd_weight: float | None,
) -> AnalysisRankingConfig:
    metadata = _campaign_metadata(root)
    analysis = metadata.get("analysis")
    analysis = analysis if isinstance(analysis, dict) else {}
    raw = analysis.get("ranking")
    raw = raw if isinstance(raw, dict) else {}

    mode = str(raw.get("mode", DEFAULT_ANALYSIS_RANKING_MODE))
    if mode not in ANALYSIS_RANKING_MODES:
        choices = ", ".join(sorted(ANALYSIS_RANKING_MODES))
        raise ValueError(f"analysis.ranking.mode must be one of: {choices}")

    configured_max_rmsd = raw.get(
        "max_binder_rmsd_angstrom",
        DEFAULT_ANALYSIS_MAX_BINDER_RMSD_ANGSTROM,
    )
    if configured_max_rmsd is None:
        resolved_max_rmsd = None
    else:
        resolved_max_rmsd = _positive_finite_float(
            configured_max_rmsd,
            "analysis.ranking.max_binder_rmsd_angstrom",
        )
    if max_binder_rmsd_angstrom is not None:
        resolved_max_rmsd = _positive_finite_float(
            max_binder_rmsd_angstrom,
            "max_binder_rmsd_angstrom",
        )

    configured_rmsd_weight = raw.get(
        "rmsd_weight",
        DEFAULT_ANALYSIS_RMSD_WEIGHT,
    )
    resolved_rmsd_weight = _ranking_weight(
        configured_rmsd_weight,
        "analysis.ranking.rmsd_weight",
    )
    if rmsd_weight is not None:
        resolved_rmsd_weight = _ranking_weight(rmsd_weight, "rmsd_weight")

    return AnalysisRankingConfig(
        mode=mode,
        max_binder_rmsd_angstrom=resolved_max_rmsd,
        rmsd_weight=resolved_rmsd_weight,
    )


def _positive_finite_float(value: Any, name: str) -> float:
    parsed = _optional_float_value(value)
    if parsed is None or not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return parsed


def _ranking_weight(value: Any, name: str) -> float:
    parsed = _optional_float_value(value)
    if parsed is None or not math.isfinite(parsed) or not 0 <= parsed < 1:
        raise ValueError(f"{name} must be at least 0 and less than 1")
    return parsed


def _score_analysis_rows(
    rows: list[dict[str, Any]],
    *,
    config: AnalysisRankingConfig,
) -> list[dict[str, Any]]:
    scored = [_score_analysis_row(row, config=config) for row in rows]
    _assign_pareto_fronts(scored)
    return scored


def _compact_ranking_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "rank": row.get("final_rank"),
        "design_name": row.get("candidate_id"),
        "sequence": row.get("designed_sequence"),
        "binder_type": row.get("binder_type"),
        "framework": row.get("framework"),
        "binder_length": row.get("binder_length"),
        "consensus_score": row.get("final_score"),
        "esmfold2_rank": row.get("selection_rank"),
        "esmfold2_iptm": row.get("esm_iptm"),
        "validator_rank": row.get("validator_rank"),
        "validator_iptm": row.get("validator_iptm"),
        "validator_ipsae": row.get("validator_ipsae"),
        "binder_rmsd_angstrom": row.get(
            "binder_ca_rmsd_after_target_alignment"
        ),
        "esmfold2_hotspot_distance_angstrom": row.get(
            "esm_hotspot_distance_angstrom"
        ),
        "validator_hotspot_distance_angstrom": row.get(
            "validator_hotspot_distance_angstrom"
        ),
        "esmfold2_structure": row.get("source_structure_path"),
        "validator_structure": row.get("validated_structure_path"),
    }
    for field in CDR_FIELDS:
        compact[field] = row.get(field)
    return compact


def _finalize_analysis_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _assign_validator_ranks(rows)
    diagnostic_rows = sorted(rows, key=_combined_ranking_sort_key)
    ranked_rows: list[dict[str, Any]] = []
    for row in diagnostic_rows:
        if _optional_bool_value(row.get("ranking_eligible")):
            row["final_rank"] = len(ranked_rows) + 1
            ranked_rows.append(row)
        else:
            row["final_rank"] = None
    return ranked_rows, diagnostic_rows


def _assign_validator_ranks(rows: list[dict[str, Any]]) -> None:
    rankable = [
        row
        for row in rows
        if str(row.get("_validator_status") or "") == "completed"
        and _optional_float_value(row.get("evaluator_score")) is not None
    ]
    rankable.sort(
        key=lambda row: (
            _descending(row.get("evaluator_score")),
            _descending(row.get("validator_iptm")),
            _descending(row.get("validator_ipsae")),
            str(row.get("candidate_id") or ""),
        )
    )
    for row in rows:
        row["validator_rank"] = None
    for rank, row in enumerate(rankable, start=1):
        row["validator_rank"] = rank


def _compact_ranking_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields = list(COMPACT_RANKING_BASE_FIELDS)
    insertion_index = fields.index("sequence") + 1
    if any(row.get("framework") not in (None, "") for row in rows):
        fields.insert(insertion_index, "framework")
        insertion_index += 1
    for field in _cdr_fields_for_rows(rows):
        if any(row.get(field) not in (None, "") for row in rows):
            fields.insert(insertion_index, field)
            insertion_index += 1

    optional_fields = (
        "validator_ipsae",
        "esmfold2_hotspot_distance_angstrom",
        "validator_hotspot_distance_angstrom",
    )
    for field in optional_fields:
        if not any(row.get(field) not in (None, "") for row in rows):
            fields.remove(field)
    return fields


def _score_analysis_row(
    row: dict[str, Any],
    *,
    config: AnalysisRankingConfig,
) -> dict[str, Any]:
    result = dict(row)
    reasons: list[str] = []
    resolved_mode = "consensus" if config.mode == "auto" else config.mode

    if str(row.get("_validator_status") or "") != "completed":
        reasons.append("validator status is not completed")
    if not _validation_task_passed(row):
        reasons.append("validator did not pass")

    esm_iptm = _ranking_unit_interval(
        row.get("esm_iptm"),
        "missing or invalid ESMFold2 ipTM",
        reasons,
    )
    validator_iptm = _ranking_unit_interval(
        row.get("validator_iptm"),
        "missing or invalid validator ipTM",
        reasons,
    )

    validator_ipsae_raw = row.get("validator_ipsae")
    validator_ipsae: float | None = None
    if validator_ipsae_raw not in (None, ""):
        validator_ipsae = _ranking_unit_interval(
            validator_ipsae_raw,
            "invalid validator ipSAE",
            reasons,
        )

    evaluator_score: float | None = None
    confidence_score: float | None = None
    metric_basis = "esmfold2_iptm+validator_iptm"
    if validator_iptm is not None:
        if validator_ipsae is None and validator_ipsae_raw in (None, ""):
            evaluator_score = validator_iptm
        elif validator_ipsae is not None:
            evaluator_score = math.sqrt(validator_iptm * validator_ipsae)
            metric_basis += "+validator_ipsae"
    if esm_iptm is not None and evaluator_score is not None:
        confidence_score = math.sqrt(esm_iptm * evaluator_score)

    rmsd = _optional_float_value(
        row.get("binder_ca_rmsd_after_target_alignment")
    )
    if rmsd is not None and (not math.isfinite(rmsd) or rmsd < 0):
        rmsd = None
    rmsd_required = (
        config.max_binder_rmsd_angstrom is not None or config.rmsd_weight > 0
    )
    rmsd_pass: bool | None = None
    if rmsd is None:
        if rmsd_required:
            reasons.append("missing binder CA RMSD after target alignment")
    elif config.max_binder_rmsd_angstrom is None:
        rmsd_pass = True
    else:
        rmsd_pass = rmsd <= config.max_binder_rmsd_angstrom
        if not rmsd_pass:
            reasons.append(
                f"binder CA RMSD {rmsd:.3f} exceeds "
                f"{config.max_binder_rmsd_angstrom:.3f} angstrom"
            )

    agreement_score: float | None = None
    if rmsd is not None:
        agreement_scale = (
            config.max_binder_rmsd_angstrom
            or DEFAULT_ANALYSIS_MAX_BINDER_RMSD_ANGSTROM
        )
        agreement_score = math.exp(-0.5 * (rmsd / agreement_scale) ** 2)

    final_score: float | None = None
    if confidence_score is not None:
        if config.rmsd_weight == 0:
            final_score = confidence_score
        elif agreement_score is not None:
            final_score = (
                confidence_score ** (1 - config.rmsd_weight)
                * agreement_score ** config.rmsd_weight
            )

    result.update(
        {
            "ranking_eligible": not reasons,
            "ranking_exclusion_reason": "; ".join(reasons) or None,
            "ranking_mode": resolved_mode,
            "ranking_metric_basis": metric_basis,
            "confidence_score": confidence_score,
            "evaluator_score": evaluator_score,
            "agreement_score": agreement_score,
            "final_score": final_score,
            "pareto_front": None,
            "rmsd_pass": rmsd_pass,
        }
    )
    return result


def _ranking_unit_interval(
    value: Any,
    reason: str,
    reasons: list[str],
) -> float | None:
    parsed = _optional_float_value(value)
    if parsed is None or not math.isfinite(parsed) or not 0 <= parsed <= 1:
        reasons.append(reason)
        return None
    return parsed


def _assign_pareto_fronts(rows: list[dict[str, Any]]) -> None:
    points: list[tuple[float, float, str, dict[str, Any]]] = []
    for row in rows:
        if not _optional_bool_value(row.get("ranking_eligible")):
            continue
        esm_iptm = _optional_float_value(row.get("esm_iptm"))
        evaluator_score = _optional_float_value(row.get("evaluator_score"))
        if esm_iptm is None or evaluator_score is None:
            continue
        points.append(
            (
                esm_iptm,
                evaluator_score,
                str(row.get("candidate_id") or ""),
                row,
            )
        )

    points.sort(key=lambda item: (-item[0], -item[1], item[2]))
    front_maxima_negated: list[float] = []
    index = 0
    while index < len(points):
        esm_iptm, evaluator_score = points[index][0], points[index][1]
        end = index + 1
        while (
            end < len(points)
            and points[end][0] == esm_iptm
            and points[end][1] == evaluator_score
        ):
            end += 1

        front_index = bisect_right(front_maxima_negated, -evaluator_score)
        pareto_front = front_index + 1
        for point in points[index:end]:
            point[3]["pareto_front"] = pareto_front

        if front_index == len(front_maxima_negated):
            front_maxima_negated.append(-evaluator_score)
        else:
            front_maxima_negated[front_index] = -evaluator_score
        index = end


def _combined_ranking_sort_key(
    row: dict[str, Any],
) -> tuple[int, int, int, float, float, float, float, int, str]:
    status_order = _validation_status_order(row.get("_validator_status"))
    eligible_order = 0 if _optional_bool_value(row.get("ranking_eligible")) else 1
    pass_order = 0 if _optional_bool_value(row.get("validator_passed")) else 1
    return (
        status_order,
        eligible_order,
        pass_order,
        _descending(row.get("final_score")),
        _ascending(row.get("binder_ca_rmsd_after_target_alignment")),
        _descending(row.get("evaluator_score")),
        _descending(row.get("esm_iptm")),
        _optional_int_value(row.get("selection_rank")) or 10**9,
        str(row.get("candidate_id") or ""),
    )


def _write_top_ranked_structures(
    root: Path,
    *,
    top_ranked_dir: Path,
    ranked_rows: list[dict[str, Any]],
    top_k: int,
) -> int:
    _clear_generated_tree(top_ranked_dir)
    copied_designs = 0
    for row in ranked_rows:
        if copied_designs >= top_k:
            break
        if not _optional_bool_value(row.get("ranking_eligible")):
            continue
        try:
            source_path = _artifact_path(root, str(row.get("source_structure_path") or ""))
            validated_path = _artifact_path(root, str(row.get("validated_structure_path") or ""))
        except Exception:
            continue
        rank = _optional_int_value(row.get("final_rank")) or copied_designs + 1
        candidate_id = str(row.get("candidate_id") or "candidate")
        validator_name = validator_slug(str(row.get("validator_model") or "validator"))
        # Group by model so each top-ranked design is a paired pair of files that
        # sort by rank and stay collision-safe even if the folders are merged:
        #   top_ranked/esmfold2/rank0001_<candidate>_esmfold2.pdb
        #   top_ranked/<validator>/rank0001_<candidate>_<validator>.cif
        # Per-design metadata is intentionally omitted; combined_ranking.csv is
        # the single source of truth and rank joins each file back to its row.
        stem = f"rank{rank:04d}_{_safe_artifact_name(candidate_id)}"
        esm_dst = top_ranked_dir / ESMFOLD2_DIR / f"{stem}_{ESMFOLD2_DIR}.pdb"
        validator_dst = (
            top_ranked_dir
            / validator_name
            / f"{stem}_{validator_name}{validated_path.suffix or '.cif'}"
        )
        write_bytes_atomic(esm_dst, source_path.read_bytes())
        write_bytes_atomic(validator_dst, validated_path.read_bytes())
        row["copied_esmfold2_structure"] = esm_dst.relative_to(root).as_posix()
        row["copied_validator_structure"] = validator_dst.relative_to(root).as_posix()
        copied_designs += 1
    return copied_designs


def _write_analysis_plots(
    *,
    plots_dir: Path,
    ranked_rows: list[dict[str, Any]],
) -> tuple[list[Path], list[str]]:
    if not ranked_rows:
        return [], ["skipped analysis plots: no ranked rows"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [], [
            "skipped analysis plots: could not import matplotlib "
            f"({type(exc).__name__}: {exc})"
        ]

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_validator_slug = _analysis_validator_slug(ranked_rows)
    plot_paths: list[Path] = []
    warnings: list[str] = []
    paths, plot_warnings = _scatter_plot(
        plt,
        plots_dir=plots_dir,
        rows=ranked_rows,
        x_field="esm_iptm",
        y_field="validator_iptm",
        color_field="binder_ca_rmsd_after_target_alignment",
        filename=(
            f"esmfold2_iptm_vs_{plot_validator_slug}_iptm_colored_by_rmsd.png"
        ),
        xlabel="ESMFold2 ipTM",
        ylabel="Validator ipTM",
        color_label="Binder RMSD after target alignment (Å)",
    )
    plot_paths.extend(paths)
    warnings.extend(plot_warnings)
    paths, plot_warnings = _scatter_plot(
        plt,
        plots_dir=plots_dir,
        rows=ranked_rows,
        x_field="esm_iptm",
        y_field="validator_ipsae",
        color_field="binder_ca_rmsd_after_target_alignment",
        filename=(
            f"esmfold2_iptm_vs_{plot_validator_slug}_ipsae_colored_by_rmsd.png"
        ),
        xlabel="ESMFold2 ipTM",
        ylabel="Validator ipSAE",
        color_label="Binder RMSD after target alignment (Å)",
    )
    plot_paths.extend(paths)
    warnings.extend(plot_warnings)
    paths, plot_warnings = _scatter_plot(
        plt,
        plots_dir=plots_dir,
        rows=ranked_rows,
        x_field="binder_ca_rmsd_after_target_alignment",
        y_field="validator_iptm",
        color_field=None,
        filename=f"{plot_validator_slug}_iptm_vs_binder_rmsd.png",
        xlabel="Binder RMSD after target alignment (Å)",
        ylabel="Validator ipTM",
        color_label=None,
    )
    plot_paths.extend(paths)
    warnings.extend(plot_warnings)
    paths, plot_warnings = _scatter_plot(
        plt,
        plots_dir=plots_dir,
        rows=ranked_rows,
        x_field="binder_ca_rmsd_after_target_alignment",
        y_field="validator_ipsae",
        color_field=None,
        filename=f"{plot_validator_slug}_ipsae_vs_binder_rmsd.png",
        xlabel="Binder RMSD after target alignment (Å)",
        ylabel="Validator ipSAE",
        color_label=None,
    )
    plot_paths.extend(paths)
    warnings.extend(plot_warnings)
    plt.close("all")
    return plot_paths, warnings


def _analysis_validator_slug(rows: list[dict[str, Any]]) -> str:
    slugs = {
        validator_slug(str(row.get("validator_model") or "validator"))
        for row in rows
        if row.get("validator_model")
    }
    if len(slugs) == 1:
        return next(iter(slugs))
    return "validator"


def _scatter_stats_text(np: Any, xs: Any, ys: Any) -> str:
    """n / Pearson / Spearman annotation, omitting coefficients when undefined."""

    lines = [f"n={len(xs)}"]
    if len(xs) >= 2 and np.std(xs) > 0 and np.std(ys) > 0:
        pearson = float(np.corrcoef(xs, ys)[0, 1])
        rank_x = xs.argsort().argsort().astype(float)
        rank_y = ys.argsort().argsort().astype(float)
        spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])
        lines.append(f"Pearson r={pearson:.2f}")
        lines.append(f"Spearman rho={spearman:.2f}")
    return "\n".join(lines)


def _scatter_plot(
    plt: Any,
    *,
    plots_dir: Path,
    rows: list[dict[str, Any]],
    x_field: str,
    y_field: str,
    color_field: str | None,
    filename: str,
    xlabel: str,
    ylabel: str,
    color_label: str | None,
) -> tuple[list[Path], list[str]]:
    import numpy as np

    points = []
    for row in rows:
        x = _optional_float_value(row.get(x_field))
        y = _optional_float_value(row.get(y_field))
        if x is None or y is None:
            continue
        color = _optional_float_value(row.get(color_field)) if color_field else None
        rejected = _optional_bool_value(row.get("validator_passed")) is not True
        points.append((x, y, color, rejected))
    if not points:
        return [], [f"skipped plot {filename}: no rows with {x_field} and {y_field}"]

    xs = np.array([point[0] for point in points], dtype=float)
    ys = np.array([point[1] for point in points], dtype=float)
    rejected = np.array([point[3] for point in points], dtype=bool)
    colors = [point[2] for point in points]
    has_color = bool(color_field) and all(value is not None for value in colors)

    figure, axis = plt.subplots(figsize=(8.4, 6.4))
    axis.grid(True, color="#e4e4e4", linewidth=0.8, zorder=0)
    if has_color:
        # RMSDs above this cap are all complete pose misses for plot triage.
        # viridis_r maps low RMSD -> yellow, high RMSD -> dark.
        color_values = np.array(colors, dtype=float)
        vmax = _colorbar_max_for_field(color_field, color_values)
        scatter = axis.scatter(
            xs, ys, c=np.clip(color_values, 0.0, vmax), cmap="viridis_r",
            vmin=0.0, vmax=vmax, s=78, edgecolor="none", alpha=0.95, zorder=3,
        )
        colorbar = figure.colorbar(scatter, ax=axis, pad=0.02)
        colorbar.set_label(color_label or color_field, fontsize=12)
        if color_field == "binder_ca_rmsd_after_target_alignment":
            colorbar.set_ticks([0.0, 5.0, 10.0, 15.0, RMSD_COLORBAR_MAX_ANGSTROM])
            colorbar.set_ticklabels(["0", "5", "10", "15", ""])
        if _colorbar_uses_overflow_label(color_field, color_values, vmax):
            colorbar.ax.text(
                0.5, 1.015, f">={vmax:g}", transform=colorbar.ax.transAxes,
                ha="center", va="bottom", fontsize=9,
            )
    else:
        axis.scatter(xs, ys, s=78, edgecolor="none", alpha=0.95, zorder=3)

    if rejected.any():
        axis.scatter(
            xs[rejected], ys[rejected], facecolors="none", edgecolors="#e0202a",
            linewidth=1.7, s=165, zorder=4, label="rejected",
        )
        axis.legend(
            loc="lower right", frameon=True, framealpha=0.92,
            edgecolor="#cccccc", fontsize=12,
        )

    axis.set_xlabel(xlabel, fontsize=13)
    axis.set_ylabel(ylabel, fontsize=13)
    axis.set_title(f"{ylabel} vs {xlabel}", fontsize=16)
    axis.text(
        0.03, 0.97, _scatter_stats_text(np, xs, ys), transform=axis.transAxes,
        va="top", ha="left", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc"),
    )
    figure.tight_layout()
    path = plots_dir / filename
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)
    return [path], []


def _colorbar_max_for_field(color_field: str, color_values: Any) -> float:
    if color_field == "binder_ca_rmsd_after_target_alignment":
        return RMSD_COLORBAR_MAX_ANGSTROM
    vmax = float(color_values.max()) or 1.0
    return vmax if vmax > 0 else 1.0


def _colorbar_uses_overflow_label(
    color_field: str,
    color_values: Any,
    vmax: float,
) -> bool:
    if color_field == "binder_ca_rmsd_after_target_alignment":
        return True
    return float(color_values.max()) > vmax + 1e-9


def _analysis_summary(
    root: Path,
    *,
    ranked_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    combined_ranking_csv: Path,
    diagnostics_csv: Path,
    top_ranked_dir: Path,
    top_k: int,
    copied_designs: int,
    plot_paths: list[Path],
    plot_warnings: list[str],
    ranking_config: AnalysisRankingConfig,
) -> dict[str, Any]:
    validator_counts = Counter(
        str(row.get("validator_model") or "unknown") for row in diagnostic_rows
    )
    passing_count = sum(
        1
        for row in diagnostic_rows
        if _optional_bool_value(row.get("validator_passed"))
    )
    return {
        "campaign": _campaign_metadata(root),
        "generated_at": _utc_now_text(),
        "outputs": {
            "combined_ranking_csv": combined_ranking_csv.relative_to(root).as_posix(),
            "ranking_diagnostics_csv": diagnostics_csv.relative_to(root).as_posix(),
            "top_ranked_dir": top_ranked_dir.relative_to(root).as_posix(),
            "plots": [path.relative_to(root).as_posix() for path in plot_paths],
        },
        "counts": {
            "ranked_designs": len(ranked_rows),
            "diagnostic_rows": len(diagnostic_rows),
            "validator_pass_count": passing_count,
            "validator_reject_or_failed_count": len(diagnostic_rows) - passing_count,
            "ranking_eligible_count": len(ranked_rows),
            "ranking_ineligible_count": len(diagnostic_rows) - len(ranked_rows),
            "top_k": top_k,
            "copied_designs": copied_designs,
        },
        "pose_agreement": _pose_agreement_summary(diagnostic_rows),
        "warnings": plot_warnings,
        "ranking": {
            "mode": (
                "consensus" if ranking_config.mode == "auto" else ranking_config.mode
            ),
            "configured_mode": ranking_config.mode,
            "max_binder_rmsd_angstrom": (
                ranking_config.max_binder_rmsd_angstrom
            ),
            "agreement_scale_angstrom": (
                ranking_config.max_binder_rmsd_angstrom
                or DEFAULT_ANALYSIS_MAX_BINDER_RMSD_ANGSTROM
            ),
            "rmsd_weight": ranking_config.rmsd_weight,
            "confidence_formula": (
                "esmfold2_iptm^0.50 * validator_iptm^0.25 * "
                "validator_ipsae^0.25"
            ),
            "agreement_formula": (
                "exp(-0.5 * (binder_ca_rmsd_after_target_alignment / "
                "agreement_scale_angstrom)^2)"
            ),
            "final_formula": (
                "confidence_score^(1-rmsd_weight) * "
                "agreement_score^rmsd_weight"
            ),
            "sort_order": [
                "completed validator status first",
                "ranking_eligible desc",
                "validator_passed desc",
                "final_score desc",
                "binder_ca_rmsd_after_target_alignment asc",
                "evaluator_score desc",
                "esm_iptm desc",
            ],
            "validator_models": dict(sorted(validator_counts.items())),
        },
        "top_designs": _top_analysis_summaries(ranked_rows),
    }


def _top_analysis_summaries(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in rows[:limit]:
        item = {
            "final_rank": _optional_int_value(row.get("final_rank")),
            "candidate_id": row.get("candidate_id"),
            "selection_rank": _optional_int_value(row.get("selection_rank")),
            "validator_model": row.get("validator_model"),
            "validator_passed": _optional_bool_value(row.get("validator_passed")),
            "ranking_eligible": _optional_bool_value(row.get("ranking_eligible")),
            "ranking_exclusion_reason": row.get("ranking_exclusion_reason"),
            "final_score": _optional_float_value(row.get("final_score")),
            "confidence_score": _optional_float_value(row.get("confidence_score")),
            "evaluator_score": _optional_float_value(row.get("evaluator_score")),
            "agreement_score": _optional_float_value(row.get("agreement_score")),
            "pareto_front": _optional_int_value(row.get("pareto_front")),
            "esm_iptm": _optional_float_value(row.get("esm_iptm")),
            "validator_iptm": _optional_float_value(row.get("validator_iptm")),
            "validator_ipsae": _optional_float_value(row.get("validator_ipsae")),
            "binder_ca_rmsd_after_target_alignment": _optional_float_value(
                row.get("binder_ca_rmsd_after_target_alignment")
            ),
            "validated_structure_path": row.get("validated_structure_path"),
        }
        summaries.append({key: value for key, value in item.items() if value is not None})
    return summaries


def _pose_agreement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_with_binder_ca_rmsd = sum(
        1
        for row in rows
        if _optional_float_value(row.get("binder_ca_rmsd_after_target_alignment"))
        is not None
    )
    errors = Counter(
        str(row.get(POSE_AGREEMENT_ERROR_FIELD))
        for row in rows
        if row.get(POSE_AGREEMENT_ERROR_FIELD)
    )
    return {
        "rows": len(rows),
        "rows_with_binder_ca_rmsd": rows_with_binder_ca_rmsd,
        "rows_missing_binder_ca_rmsd": len(rows) - rows_with_binder_ca_rmsd,
        "errors": dict(sorted(errors.items())),
    }


def _scoped_validation_report_issues(
    issues: list[Any],
    *,
    task_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
) -> list[Any]:
    task_ids = {
        str(row.get("validation_id"))
        for row in task_rows
        if row.get("validation_id")
    }
    structure_ids = {
        f"{row.get('validation_id')}:{row.get('structure_id')}"
        for row in structure_rows
        if row.get("validation_id") and row.get("structure_id")
    }
    scoped: list[Any] = []
    for issue in issues:
        if issue.table == "validation_tasks":
            if str(issue.row_id) in task_ids:
                scoped.append(issue)
            continue
        if issue.table == "validation_structures":
            if str(issue.row_id) in structure_ids:
                scoped.append(issue)
            continue
        scoped.append(issue)
    return scoped


def _validation_task_rows(root: Path) -> list[dict[str, Any]]:
    db_path = root / "campaign.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"missing campaign database: {db_path}")

    conn = connect_database(db_path)
    try:
        rows = conn.execute(
            """
            WITH source_critic AS (
                SELECT cm.*
                FROM critic_metrics AS cm
                JOIN (
                    SELECT candidate_id, MIN(critic_name) AS critic_name
                    FROM critic_metrics
                    WHERE status = 'completed'
                    GROUP BY candidate_id
                ) AS pick
                  ON pick.candidate_id = cm.candidate_id
                 AND pick.critic_name = cm.critic_name
            )
            SELECT
                vt.validation_id,
                vt.candidate_id,
                vt.model_name,
                vt.validation_config_hash,
                vt.selection_rank,
                vt.status AS validation_status,
                vt.output_structure_path,
                vt.metrics_json AS validation_metrics_json,
                vt.iptm AS validation_iptm,
                vt.ipsae AS validation_ipsae,
                vt.ptm AS validation_ptm,
                vt.ranking_score AS validation_ranking_score,
                vt.hotspot_satisfaction AS validation_hotspot_satisfaction,
                vt.runtime_seconds AS validation_runtime_seconds,
                vt.error_message AS validation_error_message,
                vt.completed_at AS validation_completed_at,
                c.designed_sequence,
                c.binder_chain_id,
                c.design_metrics_json,
                s.seed,
                sc.critic_name AS esm_critic_name,
                sc.structure_path AS esm_structure_path,
                sc.iptm AS esm_iptm,
                sc.distogram_iptm_proxy AS esm_distogram_iptm_proxy,
                sc.hotspot_satisfaction AS esm_hotspot_satisfaction,
                sc.metrics_json AS esm_metrics_json
            FROM validation_tasks AS vt
            JOIN candidates AS c
              ON c.candidate_id = vt.candidate_id
            JOIN shards AS s
              ON s.shard_id = c.shard_id
            LEFT JOIN source_critic AS sc
              ON sc.candidate_id = vt.candidate_id
            ORDER BY
                vt.selection_rank IS NULL,
                vt.selection_rank,
                vt.validation_id
            """
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise ValueError(f"invalid validation database: {db_path}") from exc
    finally:
        conn.close()

    return [_validation_task_row(row, root=root) for row in rows]


def _validation_task_row(row: sqlite3.Row, *, root: Path) -> dict[str, Any]:
    design_metrics = _json_object(row["design_metrics_json"], row["candidate_id"])
    validation_metrics = _json_object(
        row["validation_metrics_json"],
        row["validation_id"],
    )
    critic_metrics = _json_object(
        row["esm_metrics_json"] or "{}",
        f"{row['candidate_id']}:{row['esm_critic_name'] or 'critic'}",
    )
    binder_type = _binder_type(design_metrics)

    esm_hotspot = _first_nonempty(
        critic_metrics.get("cdr_hotspot_pass"),
        critic_metrics.get("cdr_hotspot_satisfaction"),
        critic_metrics.get("hotspot_satisfaction"),
        row["esm_hotspot_satisfaction"],
    )
    validation_hotspot = _first_nonempty(
        validation_metrics.get("validation_hotspot_pass"),
        validation_metrics.get("hotspot_pass"),
        validation_metrics.get("validation_hotspot_satisfaction"),
        validation_metrics.get("hotspot_satisfaction"),
        row["validation_hotspot_satisfaction"],
    )

    result = {
        "validation_id": row["validation_id"],
        "selection_rank": row["selection_rank"],
        "candidate_id": row["candidate_id"],
        "source_structure_path": row["esm_structure_path"],
        "validated_structure_path": row["output_structure_path"],
        "seed": row["seed"],
        "designed_sequence": row["designed_sequence"],
        **_cdr_sequences_from_design_metrics(
            design_metrics,
            binder_type=binder_type,
        ),
        "binder_scaffold": _first_nonempty(
            design_metrics.get("binder_scaffold"),
            validation_metrics.get("binder_scaffold"),
        ),
        "binder_type": binder_type,
        "framework": _first_nonempty(
            design_metrics.get("framework"),
            design_metrics.get("framework_name"),
            validation_metrics.get("framework"),
        ),
        "framework_source": _first_nonempty(
            design_metrics.get("framework_source"),
            validation_metrics.get("framework_source"),
        ),
        "binder_length": len(str(row["designed_sequence"] or "")),
        "binder_chain_id": row["binder_chain_id"],
        "esm_critic_name": row["esm_critic_name"],
        "esm_iptm": row["esm_iptm"],
        "esm_distogram_iptm_proxy": row["esm_distogram_iptm_proxy"],
        "esm_hotspot_pass": _hotspot_pass(esm_hotspot),
        "esm_hotspot_distance_angstrom": _validation_hotspot_distance(critic_metrics),
        "validator_model": row["model_name"],
        "validator_config_hash": row["validation_config_hash"],
        "validator_iptm": _first_nonempty(
            row["validation_iptm"],
            validation_metrics.get("validation_iptm"),
        ),
        "validator_ipsae": _first_nonempty(
            row["validation_ipsae"],
            validation_metrics.get("validation_ipSAE"),
            validation_metrics.get("validation_ipsae"),
        ),
        "min_validator_ipsae": validation_metrics.get("min_validation_ipSAE"),
        "validator_ipsae_pass": _optional_bool_value(
            validation_metrics.get("validation_ipSAE_pass")
        ),
        "validator_ptm": _first_nonempty(
            row["validation_ptm"],
            validation_metrics.get("ptm"),
        ),
        "validator_ranking_score": _first_nonempty(
            row["validation_ranking_score"],
            validation_metrics.get("ranking_score"),
        ),
        "validator_global_iptm": validation_metrics.get("validation_global_iptm"),
        "validator_metric_scope": validation_metrics.get("validation_metric_scope"),
        "validator_hotspot_pass": _hotspot_pass(validation_hotspot),
        "validator_hotspot_distance_angstrom": _validation_hotspot_distance(
            validation_metrics
        ),
        "structure_count": validation_metrics.get("structure_count"),
        "best_structure_id": validation_metrics.get("best_structure_id"),
        "validator_passed": validation_metrics.get("validation_passed"),
        "pass_reason": validation_metrics.get("pass_reason"),
        "fail_reason": validation_metrics.get("fail_reason"),
        "error_message": row["validation_error_message"],
        "completed_at": row["validation_completed_at"],
        "validator_runtime_seconds": row["validation_runtime_seconds"],
        "_validator_status": row["validation_status"],
    }
    result.update(
        _pose_agreement_metrics(
            root=root,
            source_structure_path=row["esm_structure_path"],
            validated_structure_path=row["output_structure_path"],
            binder_chain_id=row["binder_chain_id"],
            binder_sequence=row["designed_sequence"],
        )
    )
    return result


def _validation_structure_rows(root: Path) -> list[dict[str, Any]]:
    db_path = root / "campaign.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"missing campaign database: {db_path}")

    conn = connect_database(db_path)
    try:
        rows = conn.execute(
            """
            WITH source_critic AS (
                SELECT cm.*
                FROM critic_metrics AS cm
                JOIN (
                    SELECT candidate_id, MIN(critic_name) AS critic_name
                    FROM critic_metrics
                    WHERE status = 'completed'
                    GROUP BY candidate_id
                ) AS pick
                  ON pick.candidate_id = cm.candidate_id
                 AND pick.critic_name = cm.critic_name
            )
            SELECT
                vs.validation_id,
                vs.structure_id,
                vs.candidate_id,
                vs.model_name,
                vs.seed,
                vs.sample_rank,
                vs.status AS structure_status,
                vs.structure_path,
                vs.metrics_json AS structure_metrics_json,
                vs.scoped_iptm,
                vs.scoped_ipsae,
                vs.ptm,
                vs.ranking_score,
                vs.hotspot_satisfaction,
                vs.created_at,
                vt.validation_config_hash,
                vt.selection_rank,
                c.designed_sequence,
                c.binder_chain_id,
                c.design_metrics_json,
                sc.structure_path AS esm_structure_path
            FROM validation_structures AS vs
            JOIN validation_tasks AS vt
              ON vt.validation_id = vs.validation_id
            JOIN candidates AS c
              ON c.candidate_id = vs.candidate_id
            LEFT JOIN source_critic AS sc
              ON sc.candidate_id = vs.candidate_id
            ORDER BY
                vt.selection_rank IS NULL,
                vt.selection_rank,
                vs.seed,
                vs.sample_rank,
                vs.structure_id
            """
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise ValueError(f"invalid validation database: {db_path}") from exc
    finally:
        conn.close()

    return [_validation_structure_row(row, root=root) for row in rows]


def _validation_structure_row(row: sqlite3.Row, *, root: Path) -> dict[str, Any]:
    design_metrics = _json_object(row["design_metrics_json"], row["candidate_id"])
    metrics = _json_object(
        row["structure_metrics_json"],
        f"{row['validation_id']}:{row['structure_id']}",
    )
    binder_type = _binder_type(design_metrics)
    validation_hotspot = _first_nonempty(
        metrics.get("validation_hotspot_pass"),
        metrics.get("hotspot_pass"),
        metrics.get("validation_hotspot_satisfaction"),
        metrics.get("hotspot_satisfaction"),
        row["hotspot_satisfaction"],
    )
    result = {
        "validation_id": row["validation_id"],
        "structure_id": row["structure_id"],
        "candidate_id": row["candidate_id"],
        "selection_rank": row["selection_rank"],
        "validator_model": row["model_name"],
        "validator_config_hash": row["validation_config_hash"],
        "seed": row["seed"],
        "sample_rank": row["sample_rank"],
        "structure_status": row["structure_status"],
        "structure_path": row["structure_path"],
        "source_structure_path": row["esm_structure_path"],
        "designed_sequence": row["designed_sequence"],
        **_cdr_sequences_from_design_metrics(
            design_metrics,
            binder_type=binder_type,
        ),
        "binder_scaffold": _first_nonempty(
            design_metrics.get("binder_scaffold"),
            metrics.get("binder_scaffold"),
        ),
        "binder_type": binder_type,
        "framework": _first_nonempty(
            design_metrics.get("framework"),
            design_metrics.get("framework_name"),
            metrics.get("framework"),
        ),
        "framework_source": _first_nonempty(
            design_metrics.get("framework_source"),
            metrics.get("framework_source"),
        ),
        "binder_length": len(str(row["designed_sequence"] or "")),
        "validator_iptm": _first_nonempty(
            row["scoped_iptm"],
            metrics.get("validation_iptm"),
        ),
        "validator_ipsae": _first_nonempty(
            row["scoped_ipsae"],
            metrics.get("validation_ipSAE"),
            metrics.get("validation_ipsae"),
        ),
        "min_validator_ipsae": metrics.get("min_validation_ipSAE"),
        "validator_ipsae_pass": _optional_bool_value(
            metrics.get("validation_ipSAE_pass")
        ),
        "validator_ptm": _first_nonempty(row["ptm"], metrics.get("ptm")),
        "validator_ranking_score": _first_nonempty(
            row["ranking_score"],
            metrics.get("ranking_score"),
        ),
        "validator_global_iptm": metrics.get("validation_global_iptm"),
        "validator_metric_scope": metrics.get("validation_metric_scope"),
        "validator_hotspot_pass": _hotspot_pass(validation_hotspot),
        "validator_hotspot_distance_angstrom": _validation_hotspot_distance(metrics),
        "validator_passed": metrics.get("validation_passed"),
        "pass_reason": metrics.get("pass_reason"),
        "fail_reason": metrics.get("fail_reason"),
        "created_at": row["created_at"],
    }
    result.update(
        _pose_agreement_metrics(
            root=root,
            source_structure_path=row["esm_structure_path"],
            validated_structure_path=row["structure_path"],
            binder_chain_id=row["binder_chain_id"],
            binder_sequence=row["designed_sequence"],
        )
    )
    return result


def _validation_task_sort_key(row: dict[str, Any]) -> tuple[int, float, float, float, int, str]:
    return (
        _validation_status_order(row.get("_validator_status")),
        _descending(row.get("validator_iptm")),
        _descending(row.get("validator_ipsae")),
        _descending(row.get("validator_ranking_score")),
        _optional_int_value(row.get("selection_rank")) or 10**9,
        str(row.get("validation_id") or ""),
    )


def _validation_structure_sort_key(
    row: dict[str, Any],
) -> tuple[int, float, float, float, int, int, str]:
    return (
        0 if row.get("structure_status") == "passing" else 1,
        _descending(row.get("validator_iptm")),
        _descending(row.get("validator_ipsae")),
        _descending(row.get("validator_ranking_score")),
        _optional_int_value(row.get("seed")) or 0,
        _optional_int_value(row.get("sample_rank")) or 0,
        str(row.get("structure_id") or ""),
    )


def _validation_status_order(value: Any) -> int:
    status = str(value or "")
    return {
        "completed": 0,
        "running": 1,
        "pending": 2,
        "skipped": 3,
        "failed": 4,
    }.get(status, 5)


def _validation_summary(
    root: Path,
    *,
    task_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
    manifest_csv: Path,
    structures_manifest_csv: Path,
) -> dict[str, Any]:
    status = inspect_campaign(root)
    models = Counter(str(row["validator_model"]) for row in task_rows)
    config_hashes = Counter(str(row["validator_config_hash"]) for row in task_rows)
    scoped_issues = _scoped_validation_report_issues(
        status.issues,
        task_rows=task_rows,
        structure_rows=structure_rows,
    )
    task_counts = Counter(
        str(row.get("_validator_status") or "unknown") for row in task_rows
    )
    structure_counts = Counter(
        str(row.get("structure_status") or "unknown") for row in structure_rows
    )
    missing_issues = [
        issue
        for issue in scoped_issues
        if issue.kind.startswith("missing_validation")
    ]

    return {
        "campaign": _campaign_metadata(root),
        "generated_at": _utc_now_text(),
        "outputs": {
            "validator_dir": manifest_csv.parent.relative_to(root).as_posix(),
            "validation_results_csv": manifest_csv.relative_to(root).as_posix(),
            "structure_samples_csv": structures_manifest_csv.relative_to(root).as_posix(),
        },
        "counts": {
            "planned_validation_count": len(task_rows),
            "validation_tasks": dict(sorted(task_counts.items())),
            "validation_structures": dict(sorted(structure_counts.items())),
            "completed_count": task_counts.get("completed", 0),
            "failed_count": task_counts.get("failed", 0),
            "skipped_count": task_counts.get("skipped", 0),
            "pending_count": task_counts.get("pending", 0),
            "running_count": task_counts.get("running", 0),
            "passing_structure_count": structure_counts.get("passing", 0),
            "rejected_structure_count": structure_counts.get("rejected", 0),
            "missing_validation_artifacts": len(missing_issues),
            "issues": len(scoped_issues),
            "missing_artifacts": sum(
                1 for issue in scoped_issues if issue.kind.startswith("missing_")
            ),
            "untracked_artifacts": sum(
                1 for issue in scoped_issues if issue.kind == "untracked_artifact"
            ),
        },
        "validation": {
            "models": dict(sorted(models.items())),
            "config_hashes": dict(sorted(config_hashes.items())),
            "completed_task_pass_count": sum(
                1
                for row in task_rows
                if row.get("_validator_status") == "completed"
                and _validation_task_passed(row)
            ),
            "scoped_metric_note": (
                "validator_iptm and validator_ipsae are binder/VHH/scFv-to-target "
                "scoped metrics, not global complex metrics"
            ),
        },
        "pose_agreement": _pose_agreement_summary(structure_rows),
        "top_validation_rows": _top_validation_summaries(structure_rows),
        "issues": [_issue_summary(issue) for issue in scoped_issues],
    }


def _top_validation_summaries(
    rows: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in rows[:limit]:
        item = {
            "candidate_id": row.get("candidate_id"),
            "validation_id": row.get("validation_id"),
            "structure_id": row.get("structure_id"),
            "structure_status": row.get("structure_status"),
            "validator_model": row.get("validator_model"),
            "structure_path": row.get("structure_path"),
            "seed": _optional_int_value(row.get("seed")),
            "sample_rank": _optional_int_value(row.get("sample_rank")),
            "validator_iptm": _optional_float_value(row.get("validator_iptm")),
            "validator_ipsae": _optional_float_value(row.get("validator_ipsae")),
            "min_validator_ipsae": _optional_float_value(
                row.get("min_validator_ipsae")
            ),
            "validator_ipsae_pass": _optional_bool_value(
                row.get("validator_ipsae_pass")
            ),
            "validator_ptm": _optional_float_value(row.get("validator_ptm")),
            "validator_ranking_score": _optional_float_value(
                row.get("validator_ranking_score")
            ),
            "validator_global_iptm": _optional_float_value(
                row.get("validator_global_iptm")
            ),
            "validator_metric_scope": row.get("validator_metric_scope"),
            "binder_ca_rmsd_after_target_alignment": _optional_float_value(
                row.get("binder_ca_rmsd_after_target_alignment")
            ),
            "pass_reason": row.get("pass_reason"),
            "fail_reason": row.get("fail_reason"),
        }
        summaries.append({key: value for key, value in item.items() if value is not None})
    return summaries


def _validation_report_fields(
    base_fields: list[str],
    rows: list[dict[str, Any]],
) -> list[str]:
    fields = list(base_fields)
    present_cdr_fields = [
        field
        for field in _cdr_fields_for_rows(rows)
        if any(row.get(field) not in (None, "") for row in rows)
    ]
    if present_cdr_fields:
        insertion_index = fields.index("designed_sequence") + 1
        for offset, field in enumerate(present_cdr_fields):
            if field not in fields:
                fields.insert(insertion_index + offset, field)
    return fields


def _validation_hotspot_distance(metrics: dict[str, Any]) -> Any:
    return _first_nonempty(
        metrics.get("validation_hotspot_distance_angstrom"),
        metrics.get("hotspot_distance_angstrom"),
        metrics.get("cdr_hotspot_distance_angstrom"),
        metrics.get("cdr_hotspot_min_heavy_atom_distance_min"),
        metrics.get("hotspot_min_heavy_atom_distance_min"),
    )


def _validation_task_passed(row: dict[str, Any]) -> bool:
    value = _optional_bool_value(row.get("validator_passed"))
    if value is not None:
        return value
    path = str(row.get("validated_structure_path") or "")
    return f"/{VALIDATION_PASSING_DIR}/" in f"/{path}"


def _pose_agreement_metrics(
    *,
    root: Path,
    source_structure_path: Any,
    validated_structure_path: Any,
    binder_chain_id: Any,
    binder_sequence: Any,
) -> dict[str, float | str | None]:
    empty = {field: None for field in POSE_AGREEMENT_FIELDS}
    empty[POSE_AGREEMENT_ERROR_FIELD] = None
    if not source_structure_path or not validated_structure_path:
        return {
            **empty,
            POSE_AGREEMENT_ERROR_FIELD: "missing source or validated structure path",
        }
    try:
        source_path = _artifact_path(root, str(source_structure_path))
        validated_path = _artifact_path(root, str(validated_structure_path))
        source_structure = _parse_structure(source_path)
        validated_structure = _parse_structure(validated_path)
        source_chains = _chain_summaries(source_structure)
        validated_chains = _chain_summaries(validated_structure)
        source_binder = _pick_binder_chain(
            source_chains,
            binder_chain_id=str(binder_chain_id or ""),
            binder_sequence=str(binder_sequence or ""),
        )
        validated_binder = _pick_binder_chain(
            validated_chains,
            binder_chain_id="",
            binder_sequence=str(binder_sequence or ""),
        )
        if source_binder is None or validated_binder is None:
            return {
                **empty,
                POSE_AGREEMENT_ERROR_FIELD: "could not identify source or validator binder chain",
            }
        target_pairs = _matched_target_chain_pairs(
            source_chains,
            validated_chains,
            source_binder_id=source_binder["chain_id"],
            validated_binder_id=validated_binder["chain_id"],
        )
        if not target_pairs:
            return {
                **empty,
                POSE_AGREEMENT_ERROR_FIELD: "could not match target chains",
            }

        fixed_target_atoms: list[Any] = []
        moving_target_atoms: list[Any] = []
        for source_chain, validated_chain in target_pairs:
            fixed, moving = _paired_atoms(source_chain["ca_atoms"], validated_chain["ca_atoms"])
            fixed_target_atoms.extend(fixed)
            moving_target_atoms.extend(moving)
        if len(fixed_target_atoms) < 3 or len(fixed_target_atoms) != len(moving_target_atoms):
            return {
                **empty,
                POSE_AGREEMENT_ERROR_FIELD: "insufficient matched target CA atoms",
            }

        from Bio.PDB import Superimposer

        superimposer = Superimposer()
        superimposer.set_atoms(fixed_target_atoms, moving_target_atoms)
        superimposer.apply(validated_structure.get_atoms())

        binder_ca_fixed, binder_ca_moving = _paired_atoms(
            source_binder["ca_atoms"],
            validated_binder["ca_atoms"],
        )
        binder_backbone_fixed, binder_backbone_moving = _paired_atoms(
            source_binder["backbone_atoms"],
            validated_binder["backbone_atoms"],
        )
        return {
            "target_aligned_rmsd": float(superimposer.rms),
            "binder_ca_rmsd_after_target_alignment": _atom_rmsd(
                binder_ca_fixed,
                binder_ca_moving,
            ),
            "binder_backbone_rmsd_after_target_alignment": _atom_rmsd(
                binder_backbone_fixed,
                binder_backbone_moving,
            ),
            "binder_centroid_distance_after_target_alignment": _centroid_distance(
                binder_ca_fixed,
                binder_ca_moving,
            ),
            POSE_AGREEMENT_ERROR_FIELD: None,
        }
    except Exception as exc:
        return {
            **empty,
            POSE_AGREEMENT_ERROR_FIELD: f"{type(exc).__name__}: {exc}",
        }


def _parse_structure(path: Path) -> Any:
    from Bio.PDB import MMCIFParser, PDBParser

    if path.suffix.lower() in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True).get_structure(path.stem, path)
    return PDBParser(QUIET=True).get_structure(path.stem, path)


def _chain_summaries(structure: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    model = next(structure.get_models(), None)
    if model is None:
        return summaries
    for chain in model:
        residues = [residue for residue in chain if _is_protein_residue(residue)]
        ca_atoms = [residue["CA"] for residue in residues if "CA" in residue]
        backbone_atoms: list[Any] = []
        for residue in residues:
            for atom_name in ("N", "CA", "C", "O"):
                if atom_name in residue:
                    backbone_atoms.append(residue[atom_name])
        summaries.append(
            {
                "chain_id": str(chain.id),
                "sequence": "".join(_residue_one_letter(residue) for residue in residues),
                "ca_atoms": ca_atoms,
                "backbone_atoms": backbone_atoms,
            }
        )
    return summaries


def _is_protein_residue(residue: Any) -> bool:
    return residue.id[0] == " " and _residue_one_letter(residue) != "X"


def _residue_one_letter(residue: Any) -> str:
    try:
        from Bio.Data.PDBData import protein_letters_3to1_extended
    except Exception:
        return "X"
    name = str(residue.get_resname()).strip().upper()
    return protein_letters_3to1_extended.get(name, "X")


def _pick_binder_chain(
    chains: list[dict[str, Any]],
    *,
    binder_chain_id: str,
    binder_sequence: str,
) -> dict[str, Any] | None:
    if binder_chain_id:
        for chain in chains:
            if chain["chain_id"] == binder_chain_id:
                return chain
    normalized = _normalize_sequence_for_match(binder_sequence)
    if normalized:
        exact_matches = [
            chain
            for chain in chains
            if _normalize_sequence_for_match(chain.get("sequence")) == normalized
        ]
        if exact_matches:
            return exact_matches[0]
        same_length = [
            chain
            for chain in chains
            if len(_normalize_sequence_for_match(chain.get("sequence"))) == len(normalized)
        ]
        if same_length:
            return same_length[0]
    return chains[0] if chains else None


def _matched_target_chain_pairs(
    source_chains: list[dict[str, Any]],
    validated_chains: list[dict[str, Any]],
    *,
    source_binder_id: str,
    validated_binder_id: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    source_targets = [
        chain for chain in source_chains if chain["chain_id"] != source_binder_id
    ]
    validated_targets = [
        chain for chain in validated_chains if chain["chain_id"] != validated_binder_id
    ]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unused = list(validated_targets)
    for source in source_targets:
        if not unused:
            break
        source_seq = _normalize_sequence_for_match(source.get("sequence"))
        best_index = 0
        best_score = math.inf
        for index, candidate in enumerate(unused):
            candidate_seq = _normalize_sequence_for_match(candidate.get("sequence"))
            if source_seq and candidate_seq and source_seq == candidate_seq:
                best_index = index
                best_score = -1
                break
            length_penalty = abs(len(source_seq) - len(candidate_seq))
            if length_penalty < best_score:
                best_index = index
                best_score = length_penalty
        pairs.append((source, unused.pop(best_index)))
    return pairs


def _normalize_sequence_for_match(value: Any) -> str:
    return "".join(char for char in str(value or "").upper() if char.isalpha())


def _paired_atoms(first: list[Any], second: list[Any]) -> tuple[list[Any], list[Any]]:
    count = min(len(first), len(second))
    if count <= 0:
        return [], []
    return first[:count], second[:count]


def _atom_rmsd(first: list[Any], second: list[Any]) -> float | None:
    if not first or len(first) != len(second):
        return None
    total = 0.0
    for left, right in zip(first, second):
        delta = left.coord - right.coord
        total += float(delta.dot(delta))
    return math.sqrt(total / len(first))


def _centroid_distance(first: list[Any], second: list[Any]) -> float | None:
    if not first or len(first) != len(second):
        return None
    first_centroid = [0.0, 0.0, 0.0]
    second_centroid = [0.0, 0.0, 0.0]
    for left, right in zip(first, second):
        for index in range(3):
            first_centroid[index] += float(left.coord[index])
            second_centroid[index] += float(right.coord[index])
    count = float(len(first))
    first_centroid = [value / count for value in first_centroid]
    second_centroid = [value / count for value in second_centroid]
    return math.sqrt(
        sum(
            (first_centroid[index] - second_centroid[index]) ** 2
            for index in range(3)
        )
    )


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _issue_summary(issue: Any) -> dict[str, Any]:
    return {
        "kind": issue.kind,
        "path": issue.path,
        "table": issue.table,
        "row_id": issue.row_id,
        "message": issue.message,
    }


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def _completed_metric_rows(root: Path) -> list[dict[str, Any]]:
    db_path = root / "campaign.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"missing campaign database: {db_path}")

    conn = connect_database(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                c.candidate_id,
                s.seed,
                c.designed_sequence,
                c.binder_chain_id,
                c.sequence_path,
                cm.critic_name,
                cm.structure_path,
                cm.iptm,
                cm.ptm,
                cm.plddt,
                cm.distogram_iptm_proxy,
                cm.hotspot_satisfaction,
                c.design_metrics_json,
                cm.metrics_json
            FROM candidates AS c
            JOIN shards AS s
              ON s.shard_id = c.shard_id
            JOIN critic_metrics AS cm
              ON cm.candidate_id = c.candidate_id
            WHERE c.status = 'completed'
              AND cm.status = 'completed'
            ORDER BY
                COALESCE(cm.iptm, -1.0) DESC,
                COALESCE(cm.distogram_iptm_proxy, -1.0) DESC,
                c.candidate_id,
                cm.critic_name
            """
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise ValueError(f"invalid campaign database: {db_path}") from exc
    finally:
        conn.close()

    return [_metric_row(row) for row in rows]


def _metric_row(row: sqlite3.Row) -> dict[str, Any]:
    design_metrics = _json_object(row["design_metrics_json"], row["candidate_id"])
    critic_metrics = _json_object(
        row["metrics_json"],
        f"{row['candidate_id']}:{row['critic_name']}",
    )
    final_loss = _first_present(
        design_metrics,
        critic_metrics,
        keys=("final_loss", "loss"),
    )

    hotspot_contact_cutoff = _first_present(
        critic_metrics,
        {},
        keys=(
            "cdr_hotspot_contact_cutoff_angstrom",
            "hotspot_critic_contact_cutoff_angstrom",
            "hotspot_contact_cutoff_angstrom",
        ),
    )
    hotspot_satisfaction = _first_present(
        critic_metrics,
        {"hotspot_satisfaction": row["hotspot_satisfaction"]},
        keys=("cdr_hotspot_pass", "cdr_hotspot_satisfaction", "hotspot_satisfaction"),
    )
    hotspot_distance = _first_present(
        critic_metrics,
        {},
        keys=(
            "cdr_hotspot_distance_angstrom",
            "cdr_hotspot_min_heavy_atom_distance_min",
            "hotspot_min_heavy_atom_distance_min",
        ),
    )
    binder_type = _binder_type(design_metrics)
    binder_scaffold = _binder_scaffold(design_metrics, binder_type=binder_type)
    cdr_sequences = _cdr_sequences_from_design_metrics(
        design_metrics,
        binder_type=binder_type,
    )

    return {
        "candidate_id": row["candidate_id"],
        "target_name": design_metrics.get("target_name"),
        "binder_scaffold": binder_scaffold,
        "binder_type": binder_type,
        "framework": _first_present(
            design_metrics,
            {},
            keys=("framework", "framework_name"),
        ),
        "framework_source": design_metrics.get("framework_source"),
        "seed": row["seed"],
        "designed_sequence": row["designed_sequence"],
        **cdr_sequences,
        "binder_length": len(str(row["designed_sequence"])),
        "binder_chain_id": row["binder_chain_id"] or design_metrics.get("binder_chain_id"),
        "sequence_path": row["sequence_path"],
        "critic_name": row["critic_name"],
        "structure_path": row["structure_path"],
        "iptm": row["iptm"],
        "iptm_scope": critic_metrics.get("iptm_scope"),
        "complex_iptm": critic_metrics.get("complex_iptm"),
        "ptm": _first_present(critic_metrics, {"ptm": row["ptm"]}, keys=("ptm",)),
        "plddt_complex": _first_present(
            critic_metrics,
            {"plddt_complex": row["plddt"]},
            keys=("plddt_complex", "plddt"),
        ),
        "plddt_binder": critic_metrics.get("plddt_binder"),
        "plddt_target": critic_metrics.get("plddt_target"),
        "cdr_distogram_iptm_proxy": critic_metrics.get("cdr_distogram_iptm_proxy"),
        "distogram_iptm_proxy": row["distogram_iptm_proxy"],
        "hotspot_pass": _hotspot_pass(hotspot_satisfaction),
        "hotspot_distance_angstrom": hotspot_distance,
        "hotspot_contact_cutoff_angstrom": hotspot_contact_cutoff,
        "hotspot_critic_contact_cutoff_angstrom": hotspot_contact_cutoff,
        "hotspot_min_heavy_atom_distance_min": hotspot_distance,
        "target_geometry_drift_distance_rmse": critic_metrics.get(
            "target_geometry_drift_distance_rmse"
        ),
        "target_geometry_drift_aligned_rmsd": critic_metrics.get(
            "target_geometry_drift_aligned_rmsd"
        ),
        **{
            field: design_metrics.get(field)
            for field in MOSAIC_CDR_FIELDS
            if field in design_metrics
        },
        "final_loss": final_loss,
    }


def _binder_type(design_metrics: dict[str, Any]) -> str:
    value = design_metrics.get("binder_type")
    if isinstance(value, str) and value:
        return value
    scaffold = design_metrics.get("binder_scaffold")
    if isinstance(scaffold, str) and scaffold:
        return binder_code(scaffold)
    binder_name = design_metrics.get("binder_name")
    if isinstance(binder_name, str) and binder_name:
        return binder_code(binder_name)
    return "binder"


def _binder_scaffold(design_metrics: dict[str, Any], *, binder_type: str) -> str:
    value = design_metrics.get("binder_scaffold")
    if isinstance(value, str) and value:
        return value.strip().lower()
    normalized_type = str(binder_type).strip().lower()
    if normalized_type in {"scfv", "vhh"}:
        return normalized_type
    if normalized_type in {"mp", "miniprotein", "minibinder"}:
        return "miniprotein"
    return normalized_type or "binder"


def _cdr_sequences_from_design_metrics(
    design_metrics: dict[str, Any],
    *,
    binder_type: str,
) -> dict[str, str]:
    raw = design_metrics.get("cdr_sequences")
    if not isinstance(raw, dict):
        return {}
    if binder_type == "vhh":
        mapping = {
            "cdr1": "hcdr1",
            "cdr2": "hcdr2",
            "cdr3": "hcdr3",
            "hcdr1": "hcdr1",
            "hcdr2": "hcdr2",
            "hcdr3": "hcdr3",
        }
    else:
        mapping = {
            "hcdr1": "cdrh1",
            "hcdr2": "cdrh2",
            "hcdr3": "cdrh3",
            "lcdr1": "cdrl1",
            "lcdr2": "cdrl2",
            "lcdr3": "cdrl3",
        }
    return {
        field: str(value)
        for key, field in mapping.items()
        if (value := raw.get(key)) not in (None, "")
    }


def _deduplicate_by_sequence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_sequence: dict[str, dict[str, Any]] = {}
    for row in rows:
        sequence = str(row["designed_sequence"])
        existing = best_by_sequence.get(sequence)
        if existing is None or _sort_key(row) < _sort_key(existing):
            best_by_sequence[sequence] = row
    return list(best_by_sequence.values())


def _should_require_hotspot_contact(
    root: Path,
    rows: list[dict[str, Any]],
    require_hotspot_contact: Literal["auto", "always", "never"],
) -> bool:
    if require_hotspot_contact == "always":
        return True
    if require_hotspot_contact == "never":
        return False
    return _campaign_has_configured_hotspots(root) or any(
        _row_has_hotspot_metrics(row) for row in rows
    )


def _campaign_has_configured_hotspots(root: Path) -> bool:
    for name in ("resolved_config.yaml", "config.yaml"):
        path = root / name
        if not path.exists():
            continue
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(raw, dict):
            continue
        target = raw.get("target")
        if not isinstance(target, dict):
            continue
        hotspots = target.get("hotspots")
        if _has_nonempty_hotspot_value(hotspots):
            return True
    return False


def _has_nonempty_hotspot_value(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, dict):
        return any(_has_nonempty_hotspot_value(item) for item in value.values())
    return value is not None


def _row_has_hotspot_metrics(row: dict[str, Any]) -> bool:
    return (
        row.get("hotspot_contact_cutoff_angstrom") not in (None, "")
        or row.get("hotspot_distance_angstrom") not in (None, "")
        or row.get("hotspot_pass") not in (None, "")
    )


def _row_has_hotspot_contact(row: dict[str, Any]) -> bool:
    cutoff = row.get("hotspot_contact_cutoff_angstrom")
    min_distance = row.get("hotspot_min_heavy_atom_distance_min")
    if cutoff in (None, "") or min_distance in (None, ""):
        return False
    try:
        return float(min_distance) <= float(cutoff)
    except (TypeError, ValueError):
        return False


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float, str, str]:
    return (
        _descending(row["iptm"]),
        _descending(
            row.get("cdr_distogram_iptm_proxy")
            if row.get("binder_type") in {"scfv", "vhh"}
            else row.get("distogram_iptm_proxy")
        ),
        _descending(row.get("distogram_iptm_proxy")),
        str(row["candidate_id"]),
        str(row["critic_name"]),
    )


def _descending(value: Any) -> float:
    if value is None or value == "":
        return math.inf
    return -float(value)


def _ascending(value: Any) -> float:
    if value is None or value == "":
        return math.inf
    return float(value)


def _write_campaign_summary(
    root: Path,
    *,
    metric_rows: list[dict[str, Any]],
    ranked_rows: list[dict[str, Any]] | None = None,
    selection: dict[str, Any] | None = None,
    export: dict[str, Any] | None = None,
) -> Path:
    esmfold2_dir = root / ESMFOLD2_DIR
    status = inspect_campaign(root)
    metadata = _campaign_metadata(root)
    hotspot_rows = [row for row in metric_rows if _row_has_hotspot_metrics(row)]
    hotspot_pass_count = sum(1 for row in hotspot_rows if _row_has_hotspot_contact(row))
    iptm_scopes = Counter(str(row.get("iptm_scope") or "unknown") for row in metric_rows)
    framework_counts = Counter(
        str(row.get("framework"))
        for row in metric_rows
        if row.get("framework") not in (None, "")
    )
    ranked_rows = ranked_rows or []

    summary: dict[str, Any] = {
        "campaign": metadata,
        "counts": {
            "shards": status.shard_status_counts,
            "candidates": status.candidate_status_counts,
            "critics": status.critic_status_counts,
            "attempts": status.attempt_status_counts,
            "issues": len(status.issues),
            "missing_artifacts": status.missing_artifact_count,
            "untracked_artifacts": status.untracked_artifact_count,
        },
        "metrics": {
            "completed_metric_rows": len(metric_rows),
            "iptm_scope_counts": dict(sorted(iptm_scopes.items())),
            "framework_counts": dict(sorted(framework_counts.items())),
            "hotspot": {
                "configured": _campaign_has_configured_hotspots(root),
                "metric_rows": len(hotspot_rows),
                "pass_count": hotspot_pass_count,
                "pass_rate": (
                    hotspot_pass_count / len(hotspot_rows)
                    if hotspot_rows
                    else None
                ),
            },
        },
        "top_candidates": _top_candidate_summaries(ranked_rows or metric_rows),
    }
    if selection is not None:
        summary["selection"] = selection
    if export is not None:
        summary["export"] = export

    return write_json_atomic(esmfold2_dir / SUMMARY_JSON, summary)


def _campaign_metadata(root: Path) -> dict[str, Any]:
    db_path = root / "campaign.sqlite"
    if not db_path.exists():
        return {"campaign_dir": root.as_posix()}
    conn = connect_database(db_path)
    try:
        row = conn.execute(
            """
            SELECT config_hash, resolved_config_json, software_versions_json, created_at
            FROM campaign
            WHERE id = 1
            """
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"campaign_dir": root.as_posix()}

    resolved = _json_object(row["resolved_config_json"], "campaign.resolved_config_json")
    software_versions = _json_object(
        row["software_versions_json"],
        "campaign.software_versions_json",
    )
    target = resolved.get("target") if isinstance(resolved.get("target"), dict) else {}
    binder = resolved.get("binder") if isinstance(resolved.get("binder"), dict) else {}
    campaign = (
        resolved.get("campaign") if isinstance(resolved.get("campaign"), dict) else {}
    )
    loss = resolved.get("loss") if isinstance(resolved.get("loss"), dict) else {}
    analysis = (
        resolved.get("analysis")
        if isinstance(resolved.get("analysis"), dict)
        else {}
    )
    return {
        "campaign_dir": root.as_posix(),
        "config_hash": row["config_hash"],
        "created_at": row["created_at"],
        "target": target,
        "binder": binder,
        "campaign": campaign,
        "loss": loss,
        "analysis": analysis,
        "software_versions": software_versions,
    }


def _top_candidate_summaries(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in rows[:limit]:
        item = {
            "candidate_id": row.get("candidate_id"),
            "framework": row.get("framework"),
            "seed": _optional_int_value(row.get("seed")),
            "rank": _optional_int_value(row.get("rank")),
            "structure_path": row.get("structure_path"),
            "iptm": _optional_float_value(row.get("iptm")),
            "iptm_scope": row.get("iptm_scope"),
            "complex_iptm": _optional_float_value(row.get("complex_iptm")),
            "distogram_iptm_proxy": _optional_float_value(
                row.get("distogram_iptm_proxy")
            ),
            "cdr_distogram_iptm_proxy": _optional_float_value(
                row.get("cdr_distogram_iptm_proxy")
            ),
            "ptm": _optional_float_value(row.get("ptm")),
            "plddt_complex": _optional_float_value(row.get("plddt_complex")),
            "plddt_binder": _optional_float_value(row.get("plddt_binder")),
            "hotspot_pass": _optional_bool_value(row.get("hotspot_pass")),
            "hotspot_distance_angstrom": _optional_float_value(
                row.get("hotspot_distance_angstrom")
            ),
        }
        summaries.append({key: value for key, value in item.items() if value is not None})
    return summaries


def _report_fields(
    root: Path,
    base_fields: list[str],
    rows: list[dict[str, Any]],
) -> list[str]:
    fields = list(base_fields)
    if any(row.get("framework") not in (None, "") for row in rows):
        if "framework" not in fields:
            fields.insert(fields.index("seed"), "framework")
    present_cdr_fields = [
        field
        for field in _cdr_fields_for_rows(rows)
        if any(row.get(field) not in (None, "") for row in rows)
    ]
    if present_cdr_fields:
        insertion_index = fields.index("designed_sequence") + 1
        for offset, field in enumerate(present_cdr_fields):
            if field not in fields:
                fields.insert(insertion_index + offset, field)
    if any(row.get("cdr_distogram_iptm_proxy") not in (None, "") for row in rows):
        if "cdr_distogram_iptm_proxy" not in fields:
            fields.insert(fields.index("distogram_iptm_proxy"), "cdr_distogram_iptm_proxy")
    if any(
        row.get(field) not in (None, "")
        for field in TARGET_GEOMETRY_DRIFT_FIELDS
        for row in rows
    ):
        insertion_index = (
            fields.index("plddt_target") + 1
            if "plddt_target" in fields
            else len(fields)
        )
        for offset, field in enumerate(TARGET_GEOMETRY_DRIFT_FIELDS):
            if field not in fields:
                fields.insert(insertion_index + offset, field)
    if any(
        row.get(field) not in (None, "")
        for field in MOSAIC_CDR_FIELDS
        for row in rows
    ):
        insertion_index = (
            fields.index("final_loss")
            if "final_loss" in fields
            else len(fields)
        )
        for offset, field in enumerate(MOSAIC_CDR_FIELDS):
            if field not in fields:
                fields.insert(insertion_index + offset, field)
    has_hotspot_metrics = any(_row_has_hotspot_metrics(row) for row in rows)
    if not (_campaign_has_configured_hotspots(root) or has_hotspot_metrics):
        return fields
    insertion_index = (
        fields.index("plddt_target") + 1
        if "plddt_target" in fields
        else len(fields)
    )
    for offset, field in enumerate(HOTSPOT_FIELDS):
        if field not in fields:
            fields.insert(insertion_index + offset, field)
    return fields


def _cdr_fields_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    binder_types = {
        str(row.get("binder_type"))
        for row in rows
        if row.get("binder_type") not in (None, "")
    }
    if binder_types == {"vhh"}:
        return list(VHH_CDR_FIELDS)
    if binder_types == {"scfv"}:
        return list(SCFV_CDR_FIELDS)
    return [
        field
        for field in CDR_FIELDS
        if any(row.get(field) not in (None, "") for row in rows)
    ]


def _campaign_csv_metadata(
    root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = _campaign_metadata(root)
    target = metadata.get("target") if isinstance(metadata.get("target"), dict) else {}
    binder = metadata.get("binder") if isinstance(metadata.get("binder"), dict) else {}
    campaign = (
        metadata.get("campaign") if isinstance(metadata.get("campaign"), dict) else {}
    )
    loss = metadata.get("loss") if isinstance(metadata.get("loss"), dict) else {}
    drift = (
        loss.get("target_geometry_drift")
        if isinstance(loss.get("target_geometry_drift"), dict)
        else {}
    )
    first_row = rows[0] if rows else {}

    scaffold = binder.get("scaffold") or first_row.get("binder_scaffold")
    framework_values = _framework_values_from_binder_metadata(binder)
    if not framework_values:
        framework_values = sorted(
            {
                str(row.get("framework"))
                for row in rows
                if row.get("framework") not in (None, "")
            }
        )
    critics = campaign.get("critics")
    critic_text = None
    if isinstance(critics, list):
        critic_text = ";".join(str(critic) for critic in critics)
    if not critic_text:
        critic_names = sorted(
            {
                str(row.get("critic_name"))
                for row in rows
                if row.get("critic_name") not in (None, "")
            }
        )
        critic_text = ";".join(critic_names) if critic_names else None

    result = {
        "target_name": target.get("name") or first_row.get("target_name"),
        "binder_scaffold": scaffold,
        "frameworks": ";".join(framework_values) if framework_values else None,
        "num_designs": campaign.get("num_designs"),
        "steps": campaign.get("steps"),
        "critic": critic_text,
    }
    if loss.get("binder_target_contact_mode") == "mosaic_cdr":
        result["binder_target_contact_mode"] = loss.get(
            "binder_target_contact_mode"
        )
        result["mosaic_cdr_contact_weight"] = loss.get(
            "mosaic_cdr_contact_weight"
        )
        result["mosaic_framework_contact_penalty_weight"] = loss.get(
            "mosaic_framework_contact_penalty_weight"
        )
        result["mosaic_framework_contact_penalty_scope"] = loss.get(
            "mosaic_framework_contact_penalty_scope"
        )
    if "enabled" in drift:
        result["target_geometry_drift_enabled"] = drift.get("enabled")
        if drift.get("enabled"):
            result["target_geometry_drift_weight"] = drift.get("weight")
            result["target_geometry_drift_tolerance_angstrom"] = drift.get(
                "tolerance_angstrom"
            )
            result["target_geometry_drift_stiffness_angstrom"] = drift.get(
                "stiffness_angstrom"
            )
            result["target_geometry_drift_regions"] = _drift_regions_metadata_value(
                drift.get("regions")
            )
    return {key: value for key, value in result.items() if value not in (None, "")}


def _drift_regions_metadata_value(value: Any) -> str:
    if value in (None, "", {}, []):
        return "all"
    if isinstance(value, dict):
        parts = []
        for chain_id in sorted(value):
            selectors = value[chain_id]
            if isinstance(selectors, list):
                selector_text = "+".join(str(selector) for selector in selectors)
            else:
                selector_text = str(selectors)
            parts.append(f"{chain_id}:{selector_text}")
        return ";".join(parts)
    return str(value)


def _framework_values_from_binder_metadata(binder: dict[str, Any]) -> list[str]:
    def name_from_value(value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            return str(value["name"])
        return None

    if "framework" in binder:
        name = name_from_value(binder.get("framework"))
        return [name] if name else []
    frameworks = binder.get("frameworks")
    if isinstance(frameworks, list):
        return [
            name
            for item in frameworks
            if (name := name_from_value(item))
        ]
    return []


def _optional_float_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int_value(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool_value(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _csv_text(
    fields: list[str],
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    handle = io.StringIO(newline="")
    if metadata:
        csv.writer(handle).writerow(
            ["# campaign", *[f"{key}={_csv_value(value)}" for key, value in metadata.items()]]
        )
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return handle.getvalue()


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return _format_float(float(value))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("'") and _looks_like_float(stripped[1:]):
            return _format_float(float(stripped[1:]))
    return value


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _looks_like_float(value: str) -> bool:
    if not value:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def _hotspot_pass(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return bool(value)


def _read_ranked_csv(path: Path) -> list[dict[str, str]]:
    text = "".join(
        line
        for line in path.read_text().splitlines(keepends=True)
        if not line.startswith("#")
    )
    reader = csv.DictReader(io.StringIO(text))
    missing = [
        field for field in RANKED_FIELDS if field not in (reader.fieldnames or [])
    ]
    if missing:
        raise ValueError(f"ranked CSV is missing columns: {', '.join(missing)}")
    return list(reader)


def _artifact_path(root: Path, relpath: str) -> Path:
    if not relpath:
        raise ValueError("artifact path cannot be empty")
    path = Path(relpath)
    if path.is_absolute():
        raise ValueError(f"artifact path must be relative: {relpath}")

    root_resolved = root.resolve()
    full_path = (root_resolved / path).resolve()
    try:
        full_path.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"artifact path escapes campaign directory: {relpath}") from exc

    if not full_path.exists():
        raise FileNotFoundError(f"missing artifact: {full_path}")
    if not full_path.is_file():
        raise ValueError(f"artifact path is not a file: {full_path}")
    return full_path


def _clear_generated_selection(selected_dir: Path) -> None:
    selected_dir.mkdir(parents=True, exist_ok=True)
    for path in selected_dir.iterdir():
        if path.is_file():
            path.unlink()


def _clear_generated_tree(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            _clear_generated_tree(child)
            child.rmdir()
        else:
            child.unlink()


def _safe_artifact_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return text or "artifact"


def _json_object(text: str, row_id: str) -> dict[str, Any]:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metrics JSON for {row_id}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"metrics JSON must be an object for {row_id}")
    return value


def _first_present(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> Any:
    for source in (first, second):
        for key in keys:
            if key in source:
                return source[key]
    return None
