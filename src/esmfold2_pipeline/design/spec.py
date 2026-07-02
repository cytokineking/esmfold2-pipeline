from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esmfold2_pipeline.config import (
    DEFAULT_BINDER_TARGET_CONTACT_MODE,
    DEFAULT_MOSAIC_CDR_CONTACT_CUTOFF_ANGSTROM,
    DEFAULT_MOSAIC_CDR_CONTACT_WEIGHT,
    DEFAULT_MOSAIC_CDR_NUM_TARGET_CONTACTS,
    DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_CUTOFF_ANGSTROM,
    DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_SCOPE,
    DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_WEIGHT,
    DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PROBABILITY_THRESHOLD,
    TargetGeometryDriftConfig,
)
from esmfold2_pipeline.structure import PreparedTarget


@dataclass(frozen=True)
class DesignSpec:
    campaign_dir: Path
    candidate_id: str
    shard_id: str
    seed: int
    esm_repo: str | Path | None
    gpu_id: str | None
    steps: int
    target_name: str | None
    binder_name: str
    critic_name: str
    binder_scaffold: str | None = None
    binder_framework_name: str | None = None
    binder_framework_source: str | None = None
    binder_framework_template: str | None = None
    binder_framework_cdr_lengths: dict[str, tuple[int, int]] | None = None
    binder_framework_sequence: str | None = None
    binder_framework_cdr_indices: tuple[int, ...] | None = None
    target_sequence: str | None = None
    binder_length_range: tuple[int, int] | None = None
    is_antibody: bool | None = None
    inversion_model_name: str | None = None
    structure_target: PreparedTarget | None = None
    target_structure_indexing: str = "auto"
    conditioning_mode: str = "none"
    conditioning_assembly: bool = False
    conditioning_chain_pairs: tuple[tuple[str, str], ...] | None = None
    hotspot_contact_weight: float = 0.0
    hotspot_contact_cutoff_angstrom: float | None = None
    hotspot_distogram_contact_cutoff_angstrom: float | None = None
    hotspot_critic_contact_cutoff_angstrom: float | None = None
    hotspot_num_contacts: int = 1
    hotspot_contact_probability_target: float = 0.6
    hotspot_loss_mode: str | None = None
    binder_target_contact_mode: str = DEFAULT_BINDER_TARGET_CONTACT_MODE
    mosaic_cdr_contact_weight: float = DEFAULT_MOSAIC_CDR_CONTACT_WEIGHT
    mosaic_cdr_contact_cutoff_angstrom: float = (
        DEFAULT_MOSAIC_CDR_CONTACT_CUTOFF_ANGSTROM
    )
    mosaic_cdr_num_target_contacts: int = DEFAULT_MOSAIC_CDR_NUM_TARGET_CONTACTS
    mosaic_framework_contact_penalty_weight: float = (
        DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_WEIGHT
    )
    mosaic_framework_contact_penalty_cutoff_angstrom: float = (
        DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_CUTOFF_ANGSTROM
    )
    mosaic_framework_contact_probability_threshold: float = (
        DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PROBABILITY_THRESHOLD
    )
    mosaic_framework_contact_penalty_scope: str = (
        DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_SCOPE
    )
    target_geometry_drift: TargetGeometryDriftConfig | None = None
    artifact_stem: str | None = None
    disable_hf_xet: bool = True


@dataclass(frozen=True)
class BinderPromptPlan:
    binder_name: str | None
    binder_sequence: str | None
    is_antibody: bool | None
    cdr_indices: tuple[int, ...]
    cdr_lengths: dict[str, int]
    cdr_report_names: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeModels:
    inversion_models: dict[str, Any]
    critic_models: dict[str, Any]
    esmc_model: Any
    helpers: dict[str, Any]


@dataclass(frozen=True)
class DesignRunResult:
    best_sequences: list[str]
    trajectory: dict[int, dict[str, Any]]
    critic_results: list[dict[str, Any]]
    last_design_fold: dict[str, Any] | None = None
    last_confidence_fold: dict[str, Any] | None = None
