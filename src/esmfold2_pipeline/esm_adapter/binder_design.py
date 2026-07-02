from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np

from esmfold2_pipeline.artifact_layout import structure_relpath
from esmfold2_pipeline.artifacts import write_text_atomic
from esmfold2_pipeline.config import (
    BINDER_TARGET_CONTACT_MODES,
    DEFAULT_BINDER_TARGET_CONTACT_MODE,
    DEFAULT_ESMFOLD2_INVERSION_MODEL,
    DEFAULT_HOTSPOT_CONTACT_WEIGHT,
    DEFAULT_HOTSPOT_LOSS_MODE,
    DEFAULT_HOTSPOT_CRITIC_CONTACT_CUTOFF_ANGSTROM,
    DEFAULT_HOTSPOT_DISTOGRAM_CONTACT_CUTOFF_ANGSTROM,
    DEFAULT_MOSAIC_CDR_CONTACT_CUTOFF_ANGSTROM,
    DEFAULT_MOSAIC_CDR_CONTACT_WEIGHT,
    DEFAULT_MOSAIC_CDR_NUM_TARGET_CONTACTS,
    DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_CUTOFF_ANGSTROM,
    DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_SCOPE,
    DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_WEIGHT,
    DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PROBABILITY_THRESHOLD,
    HOTSPOT_LOSS_MODES,
    MOSAIC_FRAMEWORK_CONTACT_PENALTY_SCOPES,
    TargetGeometryDriftConfig,
)
from esmfold2_pipeline.design import (
    BinderPromptPlan,
    DesignRunResult,
    DesignSpec,
    RuntimeModels,
)
from esmfold2_pipeline.design import losses as design_losses
from esmfold2_pipeline.design import metrics as design_metrics
from esmfold2_pipeline.design import plm as design_plm
from esmfold2_pipeline.design.loop import run_gradient_design_loop
from esmfold2_pipeline.design.prompts import (
    antibody_cdr_report_names,
    cdr_prompt_from_indices,
    contiguous_index_runs,
    contiguous_mutable_runs,
    mutable_run_lengths,
    mutable_run_sequences,
    prepare_binder_prompt_plan,
    sample_antibody_template,
    sample_scfv_template,
    scfv_cdr_prompt_from_indices,
)
from esmfold2_pipeline.esm_adapter.folding_runtime import load_esm_folding_runtime
from esmfold2_pipeline.esm_adapter.imports import load_binder_design_module
from esmfold2_pipeline.planning import binder_code
from esmfold2_pipeline.structure import (
    PreparedTarget,
    resolve_target_geometry_drift_indices,
)
from esmfold2_pipeline.validation import (
    compute_fold_hotspot_contact_metrics,
    compute_fold_target_geometry_diagnostics,
    compute_fold_target_geometry_metrics,
    compute_fold_target_geometry_region_metrics,
    experimental_representative_coords,
)


_LOGGER = logging.getLogger(__name__)
_TEMPLATE_DISTOGRAM_INJECTION_DISABLE_ENV = (
    "ESMFOLD2_PIPELINE_DISABLE_TEMPLATE_DISTOGRAM_INJECTION"
)
_TEMPLATE_DISTOGRAM_LOG_KEYS_ATTR = "_esmfold2_pipeline_template_distogram_log_keys"
_DESIGN_BACKEND_ENV = "ESMFOLD2_PIPELINE_DESIGN_BACKEND"
_LOCAL_RUNTIME_CACHE_DISABLE_ENV = "ESMFOLD2_PIPELINE_DISABLE_LOCAL_RUNTIME_CACHE"
_SUPPORTED_DESIGN_BACKENDS = {"tutorial", "local"}
_LOCAL_ESMC_CACHE: Any | None = None
_LOCAL_INVERSION_LM_DROPOUT = 0.5
_LOCAL_CRITIC_LM_DROPOUT = 0.25
_LOCAL_MODEL_DEVICE = "cuda"
_LOCAL_MODEL_CACHE_ESMC = True


@dataclass(frozen=True)
class _LocalModelLoadSpec:
    model_name: str
    lm_dropout: float
    cache_esmc: bool
    device: str


@dataclass(frozen=True)
class _LocalRuntimeCacheKey:
    esm_repo: str | None
    gpu_id: str | None
    cuda_visible_devices: str | None
    inversion_model: _LocalModelLoadSpec
    critic_model: _LocalModelLoadSpec


@dataclass(frozen=True)
class _LocalDesignRuntime:
    binder_design: Any
    runtime_models: RuntimeModels


_LOCAL_DESIGN_RUNTIME_CACHE: dict[_LocalRuntimeCacheKey, _LocalDesignRuntime] = {}


@dataclass(frozen=True)
class DesignCandidateArtifact:
    candidate_id: str
    designed_sequence: str
    sequence_path: str | None
    critic_name: str
    structure_path: str
    design_metrics: dict[str, Any]
    critic_metrics: dict[str, Any]


@dataclass(frozen=True)
class ModelPreflightResult:
    inversion_model_name: str
    critic_name: str
    loaded_inversion_models: list[str]
    loaded_critic_models: list[str]
    esmc_loaded: bool


@dataclass(frozen=True)
class _LocalDesignExecution:
    design_run: DesignRunResult
    binder_design: Any


@dataclass(frozen=True)
class _ChainSpan:
    chain_id: str
    auth_asym_id: str
    label_asym_id: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


_BinderPromptPlan = BinderPromptPlan


_SCFV_CDR_RUN_NAMES = ("hcdr1", "hcdr2", "hcdr3", "lcdr1", "lcdr2", "lcdr3")


def _progress(message: str) -> None:
    print(f"[esmfold2-pipeline] {message}", file=sys.stderr, flush=True)


def _design_backend() -> str:
    backend = os.environ.get(_DESIGN_BACKEND_ENV, "local").strip().lower()
    if backend not in _SUPPORTED_DESIGN_BACKENDS:
        choices = ", ".join(sorted(_SUPPORTED_DESIGN_BACKENDS))
        raise ValueError(
            f"{_DESIGN_BACKEND_ENV} must be one of: {choices}; got {backend!r}"
        )
    return backend


def _build_design_spec(
    *,
    campaign_dir: str | Path,
    candidate_id: str,
    shard_id: str,
    seed: int,
    esm_repo: str | Path | None,
    gpu_id: str | None,
    steps: int,
    target_name: str | None,
    binder_name: str,
    critic_name: str,
    binder_scaffold: str | None,
    binder_framework_name: str | None,
    binder_framework_source: str | None,
    binder_framework_template: str | None,
    binder_framework_cdr_lengths: dict[str, tuple[int, int]] | None,
    binder_framework_sequence: str | None,
    binder_framework_cdr_indices: tuple[int, ...] | None,
    target_sequence: str | None,
    binder_length_range: tuple[int, int] | None,
    is_antibody: bool | None,
    inversion_model_name: str,
    structure_target: PreparedTarget | None,
    target_structure_indexing: str,
    conditioning_mode: str,
    conditioning_assembly: bool,
    conditioning_chain_pairs: tuple[tuple[str, str], ...] | None,
    hotspot_contact_weight: float,
    hotspot_contact_cutoff_angstrom: float | None,
    hotspot_distogram_contact_cutoff_angstrom: float,
    hotspot_critic_contact_cutoff_angstrom: float | None,
    hotspot_num_contacts: int,
    hotspot_contact_probability_target: float,
    hotspot_loss_mode: str,
    binder_target_contact_mode: str,
    mosaic_cdr_contact_weight: float,
    mosaic_cdr_contact_cutoff_angstrom: float,
    mosaic_cdr_num_target_contacts: int,
    mosaic_framework_contact_penalty_weight: float,
    mosaic_framework_contact_penalty_cutoff_angstrom: float,
    mosaic_framework_contact_probability_threshold: float,
    mosaic_framework_contact_penalty_scope: str,
    target_geometry_drift: TargetGeometryDriftConfig | None,
    artifact_stem: str | None,
    disable_hf_xet: bool,
) -> DesignSpec:
    return DesignSpec(
        campaign_dir=Path(campaign_dir),
        candidate_id=candidate_id,
        shard_id=shard_id,
        seed=seed,
        esm_repo=esm_repo,
        gpu_id=gpu_id,
        steps=steps,
        target_name=target_name,
        binder_name=binder_name,
        critic_name=critic_name,
        binder_scaffold=binder_scaffold,
        binder_framework_name=binder_framework_name,
        binder_framework_source=binder_framework_source,
        binder_framework_template=binder_framework_template,
        binder_framework_cdr_lengths=binder_framework_cdr_lengths,
        binder_framework_sequence=binder_framework_sequence,
        binder_framework_cdr_indices=binder_framework_cdr_indices,
        target_sequence=target_sequence,
        binder_length_range=binder_length_range,
        is_antibody=is_antibody,
        inversion_model_name=inversion_model_name,
        structure_target=structure_target,
        target_structure_indexing=target_structure_indexing,
        conditioning_mode=conditioning_mode,
        conditioning_assembly=conditioning_assembly,
        conditioning_chain_pairs=conditioning_chain_pairs,
        hotspot_contact_weight=hotspot_contact_weight,
        hotspot_contact_cutoff_angstrom=hotspot_contact_cutoff_angstrom,
        hotspot_distogram_contact_cutoff_angstrom=(
            hotspot_distogram_contact_cutoff_angstrom
        ),
        hotspot_critic_contact_cutoff_angstrom=hotspot_critic_contact_cutoff_angstrom,
        hotspot_num_contacts=hotspot_num_contacts,
        hotspot_contact_probability_target=hotspot_contact_probability_target,
        hotspot_loss_mode=hotspot_loss_mode,
        binder_target_contact_mode=binder_target_contact_mode,
        mosaic_cdr_contact_weight=mosaic_cdr_contact_weight,
        mosaic_cdr_contact_cutoff_angstrom=mosaic_cdr_contact_cutoff_angstrom,
        mosaic_cdr_num_target_contacts=mosaic_cdr_num_target_contacts,
        mosaic_framework_contact_penalty_weight=(
            mosaic_framework_contact_penalty_weight
        ),
        mosaic_framework_contact_penalty_cutoff_angstrom=(
            mosaic_framework_contact_penalty_cutoff_angstrom
        ),
        mosaic_framework_contact_probability_threshold=(
            mosaic_framework_contact_probability_threshold
        ),
        mosaic_framework_contact_penalty_scope=(
            mosaic_framework_contact_penalty_scope
        ),
        target_geometry_drift=target_geometry_drift,
        artifact_stem=artifact_stem,
        disable_hf_xet=disable_hf_xet,
    )


def _validate_design_spec(spec: DesignSpec) -> DesignSpec:
    if spec.steps <= 0:
        raise ValueError("steps must be positive")
    if spec.hotspot_contact_weight < 0:
        raise ValueError("hotspot_contact_weight must be non-negative")

    hotspot_distogram_contact_cutoff_angstrom = (
        spec.hotspot_distogram_contact_cutoff_angstrom
        if spec.hotspot_distogram_contact_cutoff_angstrom is not None
        else DEFAULT_HOTSPOT_DISTOGRAM_CONTACT_CUTOFF_ANGSTROM
    )
    hotspot_critic_contact_cutoff_angstrom = (
        spec.hotspot_critic_contact_cutoff_angstrom
        if spec.hotspot_critic_contact_cutoff_angstrom is not None
        else spec.hotspot_contact_cutoff_angstrom
        if spec.hotspot_contact_cutoff_angstrom is not None
        else DEFAULT_HOTSPOT_CRITIC_CONTACT_CUTOFF_ANGSTROM
    )
    hotspot_loss_mode = spec.hotspot_loss_mode or DEFAULT_HOTSPOT_LOSS_MODE
    target_geometry_drift = spec.target_geometry_drift or TargetGeometryDriftConfig()
    binder_target_contact_mode = (
        spec.binder_target_contact_mode or DEFAULT_BINDER_TARGET_CONTACT_MODE
    )

    if hotspot_distogram_contact_cutoff_angstrom <= 0:
        raise ValueError("hotspot_distogram_contact_cutoff_angstrom must be positive")
    if hotspot_critic_contact_cutoff_angstrom <= 0:
        raise ValueError("hotspot_critic_contact_cutoff_angstrom must be positive")
    if spec.hotspot_num_contacts <= 0:
        raise ValueError("hotspot_num_contacts must be positive")
    if not 0 < spec.hotspot_contact_probability_target <= 1:
        raise ValueError(
            "hotspot_contact_probability_target must be greater than 0 and at most 1"
        )
    if hotspot_loss_mode not in HOTSPOT_LOSS_MODES:
        choices = ", ".join(sorted(HOTSPOT_LOSS_MODES))
        raise ValueError(f"hotspot_loss_mode must be one of: {choices}")
    if binder_target_contact_mode not in BINDER_TARGET_CONTACT_MODES:
        choices = ", ".join(sorted(BINDER_TARGET_CONTACT_MODES))
        raise ValueError(f"binder_target_contact_mode must be one of: {choices}")
    if (
        binder_target_contact_mode == "mosaic_cdr"
        and binder_code(spec.binder_scaffold or spec.binder_name)
        not in {"scfv", "vhh"}
    ):
        raise ValueError("mosaic_cdr contact mode requires an scFv or VHH binder")
    if spec.mosaic_cdr_contact_weight < 0:
        raise ValueError("mosaic_cdr_contact_weight must be non-negative")
    if spec.mosaic_cdr_contact_cutoff_angstrom <= 0:
        raise ValueError("mosaic_cdr_contact_cutoff_angstrom must be positive")
    if spec.mosaic_cdr_num_target_contacts <= 0:
        raise ValueError("mosaic_cdr_num_target_contacts must be positive")
    if spec.mosaic_framework_contact_penalty_weight < 0:
        raise ValueError(
            "mosaic_framework_contact_penalty_weight must be non-negative"
        )
    if spec.mosaic_framework_contact_penalty_cutoff_angstrom <= 0:
        raise ValueError(
            "mosaic_framework_contact_penalty_cutoff_angstrom must be positive"
        )
    if not 0 < spec.mosaic_framework_contact_probability_threshold <= 1:
        raise ValueError(
            "mosaic_framework_contact_probability_threshold must be greater than "
            "0 and at most 1"
        )
    if (
        spec.mosaic_framework_contact_penalty_scope
        not in MOSAIC_FRAMEWORK_CONTACT_PENALTY_SCOPES
    ):
        choices = ", ".join(sorted(MOSAIC_FRAMEWORK_CONTACT_PENALTY_SCOPES))
        raise ValueError(
            "mosaic_framework_contact_penalty_scope must be one of: "
            f"{choices}"
        )
    if target_geometry_drift.weight < 0:
        raise ValueError("target_geometry_drift.weight must be non-negative")
    if target_geometry_drift.tolerance_angstrom <= 0:
        raise ValueError("target_geometry_drift.tolerance_angstrom must be positive")
    if target_geometry_drift.stiffness_angstrom <= 0:
        raise ValueError("target_geometry_drift.stiffness_angstrom must be positive")
    if target_geometry_drift.enabled and spec.structure_target is None:
        raise ValueError("target_geometry_drift requires structure_target")
    if spec.binder_length_range is not None:
        low, high = spec.binder_length_range
        if low <= 0 or high <= 0:
            raise ValueError("binder_length_range values must be positive")
        if high < low:
            raise ValueError("binder_length_range max must be >= min")
    if spec.conditioning_assembly and spec.conditioning_mode != "distogram":
        raise ValueError("conditioning_assembly requires conditioning_mode='distogram'")
    if spec.conditioning_assembly and (
        spec.structure_target is None or len(spec.structure_target.chains) < 2
    ):
        raise ValueError("conditioning_assembly requires at least two target chains")

    return replace(
        spec,
        hotspot_distogram_contact_cutoff_angstrom=(
            hotspot_distogram_contact_cutoff_angstrom
        ),
        hotspot_critic_contact_cutoff_angstrom=hotspot_critic_contact_cutoff_angstrom,
        hotspot_loss_mode=hotspot_loss_mode,
        binder_target_contact_mode=binder_target_contact_mode,
        target_geometry_drift=target_geometry_drift,
    )


def _run_local_binder_design_artifact(spec: DesignSpec) -> DesignCandidateArtifact:
    spec = _validate_design_spec(spec)
    _disable_hf_xet_if_requested(spec.disable_hf_xet)
    _restrict_cuda_if_requested(spec.gpu_id)

    _target_name_for_design, target_sequence_for_design = _target_design_inputs(spec)
    prompt_plan = prepare_binder_prompt_plan(
        binder_name=spec.binder_name,
        binder_scaffold=spec.binder_scaffold,
        binder_framework_name=spec.binder_framework_name,
        binder_framework_source=spec.binder_framework_source,
        binder_framework_template=spec.binder_framework_template,
        binder_framework_cdr_lengths=spec.binder_framework_cdr_lengths,
        binder_framework_sequence=spec.binder_framework_sequence,
        binder_framework_cdr_indices=spec.binder_framework_cdr_indices,
        seed=spec.seed,
        is_antibody=spec.is_antibody,
        binder_length_range=spec.binder_length_range,
        local_miniprotein=True,
        binder_prompt_factories=None,
    )
    _validate_mosaic_cdr_prompt(spec, prompt_plan)
    target_geometry_drift = spec.target_geometry_drift or TargetGeometryDriftConfig()
    target_geometry_drift_indices = _target_geometry_drift_indices_for_spec(
        spec,
        target_geometry_drift=target_geometry_drift,
    )
    local_execution = _run_local_design(
        spec,
        prompt_plan=prompt_plan,
        target_sequence_for_design=target_sequence_for_design,
        target_geometry_drift_indices=target_geometry_drift_indices,
    )
    if isinstance(local_execution, _LocalDesignExecution):
        design_run = local_execution.design_run
        binder_design = local_execution.binder_design
    else:
        design_run = local_execution
        binder_design = None
    return _artifact_from_design_run(
        spec=spec,
        binder_design=binder_design,
        prompt_plan=prompt_plan,
        target_sequence_for_design=target_sequence_for_design,
        target_geometry_drift_indices=target_geometry_drift_indices,
        design_run=design_run,
    )


def _validate_mosaic_cdr_prompt(
    spec: DesignSpec,
    prompt_plan: BinderPromptPlan,
) -> None:
    if spec.binder_target_contact_mode != "mosaic_cdr":
        return
    if not prompt_plan.cdr_indices:
        raise ValueError("mosaic_cdr contact mode requires resolved CDR indices")


def _run_local_design(
    spec: DesignSpec,
    *,
    prompt_plan: BinderPromptPlan,
    target_sequence_for_design: str | None,
    target_geometry_drift_indices: tuple[int, ...],
) -> _LocalDesignExecution:
    conditioning_enabled = (
        spec.structure_target is not None and spec.conditioning_mode == "distogram"
    )
    if spec.conditioning_mode not in {"none", "distogram"}:
        raise NotImplementedError(
            "local design backend does not yet support target conditioning"
        )
    if prompt_plan.binder_sequence is None:
        raise NotImplementedError(
            "local design backend requires a local binder prompt sequence"
        )

    runtime = _get_or_load_local_design_runtime(spec)
    binder_design = runtime.binder_design
    runtime_models = runtime.runtime_models
    target_sequence = target_sequence_for_design
    if target_sequence is None:
        if spec.target_name is None:
            raise ValueError("local design backend requires a target name or sequence")
        target_sequences = getattr(binder_design, "TARGET_SEQUENCES", {})
        try:
            target_sequence = target_sequences[spec.target_name]
        except KeyError as exc:
            raise ValueError(f"unknown target_name: {spec.target_name}") from exc

    design_run = run_gradient_design_loop(
        target_sequence=target_sequence,
        binder_sequence=prompt_plan.binder_sequence,
        is_antibody=prompt_plan.is_antibody,
        seed=spec.seed,
        steps=spec.steps,
        batch_size=1,
        inversion_models=runtime_models.inversion_models,
        critic_models=runtime_models.critic_models,
        esmc_model=runtime_models.esmc_model,
        fold_complex=_local_fold_complex_callback(
            binder_design,
            structure_target=spec.structure_target,
            condition_distograms=conditioning_enabled,
            condition_assembly=spec.conditioning_assembly,
            conditioning_chain_pairs=spec.conditioning_chain_pairs,
        ),
        compute_structure_losses=_local_structure_loss_callback(
            binder_design,
            structure_target=spec.structure_target,
            target_geometry_drift=(
                spec.target_geometry_drift or TargetGeometryDriftConfig()
            ),
            target_geometry_drift_indices=target_geometry_drift_indices,
            hotspot_contact_weight=spec.hotspot_contact_weight,
            hotspot_distogram_contact_cutoff_angstrom=(
                spec.hotspot_distogram_contact_cutoff_angstrom
                or DEFAULT_HOTSPOT_DISTOGRAM_CONTACT_CUTOFF_ANGSTROM
            ),
            hotspot_num_contacts=spec.hotspot_num_contacts,
            hotspot_contact_probability_target=(
                spec.hotspot_contact_probability_target
            ),
            hotspot_loss_mode=spec.hotspot_loss_mode or DEFAULT_HOTSPOT_LOSS_MODE,
            binder_contact_indices=prompt_plan.cdr_indices or None,
            binder_target_contact_mode=spec.binder_target_contact_mode,
            mosaic_cdr_contact_weight=spec.mosaic_cdr_contact_weight,
            mosaic_cdr_contact_cutoff_angstrom=(
                spec.mosaic_cdr_contact_cutoff_angstrom
            ),
            mosaic_cdr_num_target_contacts=spec.mosaic_cdr_num_target_contacts,
            mosaic_framework_contact_penalty_weight=(
                spec.mosaic_framework_contact_penalty_weight
            ),
            mosaic_framework_contact_penalty_cutoff_angstrom=(
                spec.mosaic_framework_contact_penalty_cutoff_angstrom
            ),
            mosaic_framework_contact_probability_threshold=(
                spec.mosaic_framework_contact_probability_threshold
            ),
            mosaic_framework_contact_penalty_scope=(
                spec.mosaic_framework_contact_penalty_scope
            ),
        ),
        compute_plm_loss=lambda **kwargs: design_plm.compute_esmc_pseudoperplexity_nll(
            **kwargs,
            torch_module=binder_design.torch,
            functional=binder_design.F,
            tokenizer_factory=binder_design.ESMCTokenizer,
        ),
        build_complex=binder_design.build_complex,
        compute_distogram_iptm_proxy=(
            lambda distogram_logits, target_length, binder_sequence, is_antibody: (
                design_metrics.compute_distogram_iptm_proxy(
                    distogram_logits,
                    target_length,
                    binder_sequence,
                    is_antibody,
                    cdr_indices=prompt_plan.cdr_indices or None,
                    bin_distance=design_losses.get_mid_points(binder_design.torch),
                    torch_module=binder_design.torch,
                )
            )
        ),
        torch_module=binder_design.torch,
        functional=binder_design.F,
        optim_module=binder_design.optim,
        seed_context=binder_design.seed_context,
    )
    return _LocalDesignExecution(
        design_run=design_run,
        binder_design=binder_design,
    )


def _local_structure_loss_callback(
    binder_design,
    *,
    structure_target: PreparedTarget | None,
    target_geometry_drift: TargetGeometryDriftConfig,
    target_geometry_drift_indices: tuple[int, ...],
    hotspot_contact_weight: float,
    hotspot_distogram_contact_cutoff_angstrom: float,
    hotspot_num_contacts: int,
    hotspot_contact_probability_target: float,
    hotspot_loss_mode: str,
    binder_contact_indices: tuple[int, ...] | None,
    binder_target_contact_mode: str = DEFAULT_BINDER_TARGET_CONTACT_MODE,
    mosaic_cdr_contact_weight: float = DEFAULT_MOSAIC_CDR_CONTACT_WEIGHT,
    mosaic_cdr_contact_cutoff_angstrom: float = (
        DEFAULT_MOSAIC_CDR_CONTACT_CUTOFF_ANGSTROM
    ),
    mosaic_cdr_num_target_contacts: int = DEFAULT_MOSAIC_CDR_NUM_TARGET_CONTACTS,
    mosaic_framework_contact_penalty_weight: float = (
        DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_WEIGHT
    ),
    mosaic_framework_contact_penalty_cutoff_angstrom: float = (
        DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_CUTOFF_ANGSTROM
    ),
    mosaic_framework_contact_probability_threshold: float = (
        DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PROBABILITY_THRESHOLD
    ),
    mosaic_framework_contact_penalty_scope: str = (
        DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_SCOPE
    ),
):
    drift_reference_distances: np.ndarray | None = None
    drift_pair_mask: np.ndarray | None = None
    if (
        structure_target is not None
        and target_geometry_drift.enabled
        and target_geometry_drift.weight > 0
    ):
        drift_reference_distances = _target_reference_distance_matrix(structure_target)
        drift_pair_mask = _target_geometry_drift_pair_mask(
            target_geometry_drift_indices,
            target_length=sum(len(chain.residues) for chain in structure_target.chains),
        )
        if not drift_pair_mask.any():
            raise ValueError("target geometry drift loss selected no valid residue pairs")

    hotspot_indices = (
        _target_global_hotspot_indices(structure_target)
        if structure_target is not None
        else ()
    )

    def compute_structure_losses(distogram_logits, binder_length: int) -> dict[str, Any]:
        bin_distance = design_losses.get_mid_points(binder_design.torch)
        use_mosaic_cdr = binder_target_contact_mode == "mosaic_cdr"
        losses = design_losses.compute_design_structure_losses(
            distogram_logits,
            binder_length,
            torch_module=binder_design.torch,
            bin_distance=bin_distance,
            target_geometry_reference_distances=drift_reference_distances,
            target_geometry_pair_mask=drift_pair_mask,
            target_geometry_weight=target_geometry_drift.weight
            if drift_reference_distances is not None
            else 0.0,
            target_geometry_tolerance_angstrom=(
                target_geometry_drift.tolerance_angstrom
            ),
            target_geometry_stiffness_angstrom=(
                target_geometry_drift.stiffness_angstrom
            ),
            hotspot_indices=hotspot_indices,
            hotspot_contact_weight=0.0 if use_mosaic_cdr else hotspot_contact_weight,
            hotspot_contact_cutoff_angstrom=(
                hotspot_distogram_contact_cutoff_angstrom
            ),
            hotspot_num_contacts=hotspot_num_contacts,
            hotspot_contact_probability_target=(
                hotspot_contact_probability_target
            ),
            hotspot_loss_mode=hotspot_loss_mode,
            binder_contact_indices=binder_contact_indices,
            include_inter_contact=not use_mosaic_cdr,
        )
        if not use_mosaic_cdr:
            return losses
        if not binder_contact_indices:
            raise ValueError("mosaic_cdr contact mode requires CDR contact indices")
        mosaic_loss = design_losses.compute_mosaic_cdr_contact_loss(
            binder_design.torch,
            distogram_logits,
            binder_length,
            cdr_indices=binder_contact_indices,
            contact_cutoff_angstrom=mosaic_cdr_contact_cutoff_angstrom,
            num_target_contacts=mosaic_cdr_num_target_contacts,
            hotspot_indices=hotspot_indices,
            bin_distances=bin_distance,
        )
        losses["mosaic_cdr_contact_loss"] = mosaic_loss
        losses["total_loss"] = (
            losses["total_loss"] + mosaic_cdr_contact_weight * mosaic_loss
        )
        if mosaic_framework_contact_penalty_weight > 0:
            framework_penalty_hotspot_indices = _framework_penalty_hotspot_indices(
                hotspot_indices=hotspot_indices,
                scope=mosaic_framework_contact_penalty_scope,
            )
            framework_penalty = design_losses.compute_framework_contact_penalty_loss(
                binder_design.torch,
                distogram_logits,
                binder_length,
                cdr_indices=binder_contact_indices,
                contact_cutoff_angstrom=(
                    mosaic_framework_contact_penalty_cutoff_angstrom
                ),
                num_target_contacts=mosaic_cdr_num_target_contacts,
                contact_probability_threshold=(
                    mosaic_framework_contact_probability_threshold
                ),
                hotspot_indices=framework_penalty_hotspot_indices,
                bin_distances=bin_distance,
            )
            losses["mosaic_framework_contact_penalty_loss"] = framework_penalty
            losses["total_loss"] = (
                losses["total_loss"]
                + mosaic_framework_contact_penalty_weight * framework_penalty
            )
        return losses

    return compute_structure_losses


def _local_fold_complex_callback(
    binder_design,
    *,
    structure_target: PreparedTarget | None,
    condition_distograms: bool,
    condition_assembly: bool,
    conditioning_chain_pairs: tuple[tuple[str, str], ...] | None,
):
    def fold_complex(
        model,
        target_seq: str,
        target_one_hot,
        design,
        *,
        num_loops: int = 0,
        num_sampling_steps: int = 1,
        calculate_confidence: bool = False,
        seed: int | None = None,
    ) -> dict:
        if structure_target is None:
            return _fold_and_get_distogram_for_sequence_target(
                binder_design,
                model,
                target_seq,
                target_one_hot,
                design,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                calculate_confidence=calculate_confidence,
                seed=seed,
            )

        expected_target_seq = _structure_target_sequence(structure_target)
        if target_seq != expected_target_seq:
            raise ValueError(
                "structure target sequence changed during design: "
                f"expected length {len(expected_target_seq)}, got {len(target_seq)}"
            )
        return _fold_and_get_distogram_for_structure_target(
            binder_design,
            model,
            target_one_hot,
            design,
            structure_target=structure_target,
            condition_distograms=condition_distograms,
            condition_assembly=condition_assembly,
            conditioning_chain_pairs=conditioning_chain_pairs,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            calculate_confidence=calculate_confidence,
            seed=seed,
        )

    return fold_complex


def _target_design_inputs(spec: DesignSpec) -> tuple[str | None, str | None]:
    if spec.structure_target is not None:
        return None, _structure_target_sequence(spec.structure_target)
    if spec.target_sequence is not None:
        return None, spec.target_sequence
    return spec.target_name, None


def _target_geometry_drift_indices_for_spec(
    spec: DesignSpec,
    *,
    target_geometry_drift: TargetGeometryDriftConfig,
) -> tuple[int, ...]:
    if spec.structure_target is None or not target_geometry_drift.enabled:
        return ()
    return resolve_target_geometry_drift_indices(
        spec.structure_target,
        target_geometry_drift.regions,
        structure_indexing=spec.target_structure_indexing,
        field_name="loss.target_geometry_drift.regions",
    )


def run_binder_design_artifact(
    *,
    campaign_dir: str | Path,
    candidate_id: str,
    shard_id: str,
    seed: int,
    esm_repo: str | Path | None,
    gpu_id: str | None,
    steps: int,
    target_name: str | None,
    binder_name: str,
    critic_name: str,
    binder_scaffold: str | None = None,
    binder_framework_name: str | None = None,
    binder_framework_source: str | None = None,
    binder_framework_template: str | None = None,
    binder_framework_cdr_lengths: dict[str, tuple[int, int]] | None = None,
    binder_framework_sequence: str | None = None,
    binder_framework_cdr_indices: tuple[int, ...] | None = None,
    target_sequence: str | None = None,
    binder_length_range: tuple[int, int] | None = None,
    is_antibody: bool | None = None,
    inversion_model_name: str = DEFAULT_ESMFOLD2_INVERSION_MODEL,
    structure_target: PreparedTarget | None = None,
    target_structure_indexing: str = "auto",
    conditioning_mode: str = "none",
    conditioning_assembly: bool = False,
    conditioning_chain_pairs: tuple[tuple[str, str], ...] | None = None,
    hotspot_contact_weight: float = DEFAULT_HOTSPOT_CONTACT_WEIGHT,
    hotspot_contact_cutoff_angstrom: float | None = None,
    hotspot_distogram_contact_cutoff_angstrom: float = (
        DEFAULT_HOTSPOT_DISTOGRAM_CONTACT_CUTOFF_ANGSTROM
    ),
    hotspot_critic_contact_cutoff_angstrom: float | None = None,
    hotspot_num_contacts: int = 1,
    hotspot_contact_probability_target: float = 0.6,
    hotspot_loss_mode: str = DEFAULT_HOTSPOT_LOSS_MODE,
    binder_target_contact_mode: str = DEFAULT_BINDER_TARGET_CONTACT_MODE,
    mosaic_cdr_contact_weight: float = DEFAULT_MOSAIC_CDR_CONTACT_WEIGHT,
    mosaic_cdr_contact_cutoff_angstrom: float = (
        DEFAULT_MOSAIC_CDR_CONTACT_CUTOFF_ANGSTROM
    ),
    mosaic_cdr_num_target_contacts: int = DEFAULT_MOSAIC_CDR_NUM_TARGET_CONTACTS,
    mosaic_framework_contact_penalty_weight: float = (
        DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_WEIGHT
    ),
    mosaic_framework_contact_penalty_cutoff_angstrom: float = (
        DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_CUTOFF_ANGSTROM
    ),
    mosaic_framework_contact_probability_threshold: float = (
        DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PROBABILITY_THRESHOLD
    ),
    mosaic_framework_contact_penalty_scope: str = (
        DEFAULT_MOSAIC_FRAMEWORK_CONTACT_PENALTY_SCOPE
    ),
    target_geometry_drift: TargetGeometryDriftConfig | None = None,
    artifact_stem: str | None = None,
    disable_hf_xet: bool = True,
) -> DesignCandidateArtifact:
    """Run the selected design backend and publish durable artifacts + scalars."""

    spec = _build_design_spec(
        campaign_dir=campaign_dir,
        candidate_id=candidate_id,
        shard_id=shard_id,
        seed=seed,
        esm_repo=esm_repo,
        gpu_id=gpu_id,
        steps=steps,
        target_name=target_name,
        binder_name=binder_name,
        critic_name=critic_name,
        binder_scaffold=binder_scaffold,
        binder_framework_name=binder_framework_name,
        binder_framework_source=binder_framework_source,
        binder_framework_template=binder_framework_template,
        binder_framework_cdr_lengths=binder_framework_cdr_lengths,
        binder_framework_sequence=binder_framework_sequence,
        binder_framework_cdr_indices=binder_framework_cdr_indices,
        target_sequence=target_sequence,
        binder_length_range=binder_length_range,
        is_antibody=is_antibody,
        inversion_model_name=inversion_model_name,
        structure_target=structure_target,
        target_structure_indexing=target_structure_indexing,
        conditioning_mode=conditioning_mode,
        conditioning_assembly=conditioning_assembly,
        conditioning_chain_pairs=conditioning_chain_pairs,
        hotspot_contact_weight=hotspot_contact_weight,
        hotspot_contact_cutoff_angstrom=hotspot_contact_cutoff_angstrom,
        hotspot_distogram_contact_cutoff_angstrom=(
            hotspot_distogram_contact_cutoff_angstrom
        ),
        hotspot_critic_contact_cutoff_angstrom=(
            hotspot_critic_contact_cutoff_angstrom
        ),
        hotspot_num_contacts=hotspot_num_contacts,
        hotspot_contact_probability_target=hotspot_contact_probability_target,
        hotspot_loss_mode=hotspot_loss_mode,
        binder_target_contact_mode=binder_target_contact_mode,
        mosaic_cdr_contact_weight=mosaic_cdr_contact_weight,
        mosaic_cdr_contact_cutoff_angstrom=mosaic_cdr_contact_cutoff_angstrom,
        mosaic_cdr_num_target_contacts=mosaic_cdr_num_target_contacts,
        mosaic_framework_contact_penalty_weight=(
            mosaic_framework_contact_penalty_weight
        ),
        mosaic_framework_contact_penalty_cutoff_angstrom=(
            mosaic_framework_contact_penalty_cutoff_angstrom
        ),
        mosaic_framework_contact_probability_threshold=(
            mosaic_framework_contact_probability_threshold
        ),
        mosaic_framework_contact_penalty_scope=(
            mosaic_framework_contact_penalty_scope
        ),
        target_geometry_drift=target_geometry_drift,
        artifact_stem=artifact_stem,
        disable_hf_xet=disable_hf_xet,
    )
    spec = _validate_design_spec(spec)
    campaign_dir = spec.campaign_dir
    candidate_id = spec.candidate_id
    shard_id = spec.shard_id
    seed = spec.seed
    esm_repo = spec.esm_repo
    gpu_id = spec.gpu_id
    steps = spec.steps
    target_name = spec.target_name
    binder_name = spec.binder_name
    critic_name = spec.critic_name
    binder_scaffold = spec.binder_scaffold
    binder_framework_name = spec.binder_framework_name
    binder_framework_source = spec.binder_framework_source
    binder_framework_template = spec.binder_framework_template
    binder_framework_cdr_lengths = spec.binder_framework_cdr_lengths
    binder_framework_sequence = spec.binder_framework_sequence
    binder_framework_cdr_indices = spec.binder_framework_cdr_indices
    target_sequence = spec.target_sequence
    binder_length_range = spec.binder_length_range
    is_antibody = spec.is_antibody
    inversion_model_name = spec.inversion_model_name or DEFAULT_ESMFOLD2_INVERSION_MODEL
    structure_target = spec.structure_target
    target_structure_indexing = spec.target_structure_indexing
    conditioning_mode = spec.conditioning_mode
    conditioning_assembly = spec.conditioning_assembly
    conditioning_chain_pairs = spec.conditioning_chain_pairs
    hotspot_contact_weight = spec.hotspot_contact_weight
    hotspot_contact_cutoff_angstrom = spec.hotspot_contact_cutoff_angstrom
    hotspot_distogram_contact_cutoff_angstrom = (
        spec.hotspot_distogram_contact_cutoff_angstrom
    )
    hotspot_critic_contact_cutoff_angstrom = (
        spec.hotspot_critic_contact_cutoff_angstrom
    )
    hotspot_num_contacts = spec.hotspot_num_contacts
    hotspot_contact_probability_target = spec.hotspot_contact_probability_target
    hotspot_loss_mode = spec.hotspot_loss_mode
    binder_target_contact_mode = spec.binder_target_contact_mode
    mosaic_cdr_contact_weight = spec.mosaic_cdr_contact_weight
    mosaic_cdr_contact_cutoff_angstrom = spec.mosaic_cdr_contact_cutoff_angstrom
    mosaic_cdr_num_target_contacts = spec.mosaic_cdr_num_target_contacts
    mosaic_framework_contact_penalty_weight = (
        spec.mosaic_framework_contact_penalty_weight
    )
    mosaic_framework_contact_penalty_cutoff_angstrom = (
        spec.mosaic_framework_contact_penalty_cutoff_angstrom
    )
    mosaic_framework_contact_probability_threshold = (
        spec.mosaic_framework_contact_probability_threshold
    )
    mosaic_framework_contact_penalty_scope = (
        spec.mosaic_framework_contact_penalty_scope
    )
    target_geometry_drift = spec.target_geometry_drift or TargetGeometryDriftConfig()
    artifact_stem = spec.artifact_stem
    disable_hf_xet = spec.disable_hf_xet
    backend = _design_backend()
    if backend == "local":
        return _run_local_binder_design_artifact(spec)

    if steps <= 0:
        raise ValueError("steps must be positive")
    if hotspot_contact_weight < 0:
        raise ValueError("hotspot_contact_weight must be non-negative")
    if hotspot_critic_contact_cutoff_angstrom is None:
        hotspot_critic_contact_cutoff_angstrom = (
            hotspot_contact_cutoff_angstrom
            if hotspot_contact_cutoff_angstrom is not None
            else DEFAULT_HOTSPOT_CRITIC_CONTACT_CUTOFF_ANGSTROM
        )
    if hotspot_distogram_contact_cutoff_angstrom <= 0:
        raise ValueError("hotspot_distogram_contact_cutoff_angstrom must be positive")
    if hotspot_critic_contact_cutoff_angstrom <= 0:
        raise ValueError("hotspot_critic_contact_cutoff_angstrom must be positive")
    if hotspot_num_contacts <= 0:
        raise ValueError("hotspot_num_contacts must be positive")
    if not 0 < hotspot_contact_probability_target <= 1:
        raise ValueError(
            "hotspot_contact_probability_target must be greater than 0 and at most 1"
        )
    if hotspot_loss_mode not in HOTSPOT_LOSS_MODES:
        choices = ", ".join(sorted(HOTSPOT_LOSS_MODES))
        raise ValueError(f"hotspot_loss_mode must be one of: {choices}")
    if target_geometry_drift is None:
        target_geometry_drift = TargetGeometryDriftConfig()
    if target_geometry_drift.weight < 0:
        raise ValueError("target_geometry_drift.weight must be non-negative")
    if target_geometry_drift.tolerance_angstrom <= 0:
        raise ValueError("target_geometry_drift.tolerance_angstrom must be positive")
    if target_geometry_drift.stiffness_angstrom <= 0:
        raise ValueError("target_geometry_drift.stiffness_angstrom must be positive")
    if target_geometry_drift.enabled and structure_target is None:
        raise ValueError("target_geometry_drift requires structure_target")
    if binder_length_range is not None:
        low, high = binder_length_range
        if low <= 0 or high <= 0:
            raise ValueError("binder_length_range values must be positive")
        if high < low:
            raise ValueError("binder_length_range max must be >= min")
    if conditioning_assembly and conditioning_mode != "distogram":
        raise ValueError("conditioning_assembly requires conditioning_mode='distogram'")
    if conditioning_assembly and (
        structure_target is None or len(structure_target.chains) < 2
    ):
        raise ValueError("conditioning_assembly requires at least two target chains")

    _disable_hf_xet_if_requested(disable_hf_xet)
    _restrict_cuda_if_requested(gpu_id)
    _progress(
        f"starting shard {shard_id} candidate={candidate_id} seed={seed} gpu={gpu_id or 'default'}"
    )
    _progress("loading ESM tutorial binder_design module")
    binder_design = load_binder_design_module(esm_repo)
    _progress("loaded ESM tutorial binder_design module")
    target_name_for_design, target_sequence_for_design = _target_design_inputs(spec)

    _progress(
        "preparing binder prompt "
        f"scaffold={binder_scaffold or binder_name} framework={binder_framework_name or binder_name}"
    )
    prompt_plan = _prepare_binder_prompt_plan(
        binder_design,
        binder_name=binder_name,
        binder_scaffold=binder_scaffold,
        binder_framework_name=binder_framework_name,
        binder_framework_source=binder_framework_source,
        binder_framework_template=binder_framework_template,
        binder_framework_cdr_lengths=binder_framework_cdr_lengths,
        binder_framework_sequence=binder_framework_sequence,
        binder_framework_cdr_indices=binder_framework_cdr_indices,
        seed=seed,
        is_antibody=is_antibody,
    )
    _validate_mosaic_cdr_prompt(spec, prompt_plan)
    _progress(
        "prepared binder prompt "
        f"binder_sequence={'provided' if prompt_plan.binder_sequence else 'factory'} "
        f"cdr_positions={len(prompt_plan.cdr_indices)}"
    )
    _progress(
        "loading ESMFold2/ESMC models; first run may download large checkpoints "
        "and spend several minutes before GPU memory increases"
    )
    app = _load_tutorial_app(
        binder_design,
        inversion_model_name=inversion_model_name,
        critic_name=critic_name,
        steps=steps,
    )
    _progress(
        f"loaded models inversion={inversion_model_name} critic={critic_name}"
    )
    fold_capture = _FoldCapture()
    conditioning_enabled = (
        structure_target is not None and conditioning_mode == "distogram"
    )
    target_geometry_drift_indices = _target_geometry_drift_indices_for_spec(
        spec,
        target_geometry_drift=target_geometry_drift,
    )
    if binder_target_contact_mode == "mosaic_cdr":
        target_contact_loss_context = _patched_structure_losses_for_mosaic_cdr(
            binder_design,
            structure_target=structure_target,
            cdr_indices=prompt_plan.cdr_indices,
            mosaic_cdr_contact_weight=mosaic_cdr_contact_weight,
            mosaic_cdr_contact_cutoff_angstrom=mosaic_cdr_contact_cutoff_angstrom,
            mosaic_cdr_num_target_contacts=mosaic_cdr_num_target_contacts,
            mosaic_framework_contact_penalty_weight=(
                mosaic_framework_contact_penalty_weight
            ),
            mosaic_framework_contact_penalty_cutoff_angstrom=(
                mosaic_framework_contact_penalty_cutoff_angstrom
            ),
            mosaic_framework_contact_probability_threshold=(
                mosaic_framework_contact_probability_threshold
            ),
            mosaic_framework_contact_penalty_scope=(
                mosaic_framework_contact_penalty_scope
            ),
        )
    elif binder_target_contact_mode == "legacy":
        target_contact_loss_context = _patched_structure_losses_for_hotspots(
            binder_design,
            structure_target=structure_target,
            hotspot_contact_weight=hotspot_contact_weight,
            hotspot_distogram_contact_cutoff_angstrom=(
                hotspot_distogram_contact_cutoff_angstrom
            ),
            hotspot_num_contacts=hotspot_num_contacts,
            hotspot_contact_probability_target=hotspot_contact_probability_target,
            hotspot_loss_mode=hotspot_loss_mode,
            binder_contact_indices=prompt_plan.cdr_indices or None,
        )
    else:
        target_contact_loss_context = nullcontext()

    with _patched_binder_length_range(
        binder_design,
        binder_name=binder_name,
        length_range=binder_length_range,
    ):
        with _patched_structure_losses_for_target_geometry_drift(
            binder_design,
            structure_target=structure_target,
            drift_config=target_geometry_drift,
            selected_indices=target_geometry_drift_indices,
        ):
            with target_contact_loss_context:
                with _patched_fold_with_distogram_conditioning(
                    binder_design,
                    structure_target=structure_target,
                    enabled=conditioning_enabled,
                    condition_assembly=conditioning_assembly,
                    conditioning_chain_pairs=conditioning_chain_pairs,
                    capture=fold_capture,
                ):
                    with _patched_antibody_cdr_indices(
                        binder_design,
                        cdr_indices=prompt_plan.cdr_indices,
                        binder_length=(
                            len(prompt_plan.binder_sequence)
                            if prompt_plan.binder_sequence is not None
                            else None
                        ),
                    ):
                        _progress(
                            f"starting ESMFold2 design steps={steps} target={target_name or 'custom'}"
                        )
                        best_sequences, _trajectory, critic_results = app.design(
                            target_name=target_name_for_design,
                            target_sequence=target_sequence_for_design,
                            binder_name=prompt_plan.binder_name,
                            binder_sequence=prompt_plan.binder_sequence,
                            is_antibody=prompt_plan.is_antibody,
                            seed=seed,
                            batch_size=1,
                        )
                    _progress("finished ESMFold2 design and critic evaluation")
    design_run = DesignRunResult(
        best_sequences=best_sequences,
        trajectory=_trajectory,
        critic_results=critic_results,
        last_design_fold=fold_capture.last_design_fold,
        last_confidence_fold=fold_capture.last_confidence_fold,
    )
    return _artifact_from_design_run(
        spec=spec,
        binder_design=binder_design,
        prompt_plan=prompt_plan,
        target_sequence_for_design=target_sequence_for_design,
        target_geometry_drift_indices=target_geometry_drift_indices,
        design_run=design_run,
    )


def _artifact_from_design_run(
    *,
    spec: DesignSpec,
    binder_design: Any | None,
    prompt_plan: BinderPromptPlan,
    target_sequence_for_design: str | None,
    target_geometry_drift_indices: tuple[int, ...],
    design_run: DesignRunResult,
) -> DesignCandidateArtifact:
    if not design_run.critic_results:
        raise RuntimeError("ESMFold2 design produced no critic results")

    target_geometry_drift = spec.target_geometry_drift or TargetGeometryDriftConfig()
    hotspot_distogram_contact_cutoff_angstrom = (
        spec.hotspot_distogram_contact_cutoff_angstrom
    )
    hotspot_critic_contact_cutoff_angstrom = (
        spec.hotspot_critic_contact_cutoff_angstrom
    )
    if hotspot_distogram_contact_cutoff_angstrom is None:
        hotspot_distogram_contact_cutoff_angstrom = (
            DEFAULT_HOTSPOT_DISTOGRAM_CONTACT_CUTOFF_ANGSTROM
        )
    if hotspot_critic_contact_cutoff_angstrom is None:
        hotspot_critic_contact_cutoff_angstrom = (
            spec.hotspot_contact_cutoff_angstrom
            if spec.hotspot_contact_cutoff_angstrom is not None
            else DEFAULT_HOTSPOT_CRITIC_CONTACT_CUTOFF_ANGSTROM
        )
    hotspot_loss_mode = spec.hotspot_loss_mode or DEFAULT_HOTSPOT_LOSS_MODE
    binder_target_contact_mode = (
        spec.binder_target_contact_mode or DEFAULT_BINDER_TARGET_CONTACT_MODE
    )
    inversion_model_name = spec.inversion_model_name or DEFAULT_ESMFOLD2_INVERSION_MODEL
    target_global_hotspots: tuple[int, ...] = ()

    critic_result = _select_critic_result(
        design_run.critic_results,
        spec.critic_name,
    )
    fallback_sequence = (
        design_run.best_sequences[0] if design_run.best_sequences else ""
    )
    complex_sequence = str(critic_result.get("designed_sequence") or fallback_sequence)
    designed_target_sequence, designed_sequence = _split_complex_sequence(
        complex_sequence
    )
    cdr_sequences = _mutable_run_sequences(
        designed_sequence,
        prompt_plan.cdr_indices,
        cdr_names=prompt_plan.cdr_report_names,
    )
    structure_path = structure_relpath(spec.artifact_stem or spec.candidate_id)

    root = Path(spec.campaign_dir)
    complex_obj = critic_result.get("complex")
    if complex_obj is None:
        raise RuntimeError("critic result did not include a ProteinComplex")
    pdb_text = complex_obj.to_pdb_string()
    plddt_b_factors = _plddt_b_factors_from_capture(
        design_run.last_confidence_fold,
    )
    if plddt_b_factors is not None:
        pdb_text = _rewrite_pdb_b_factors(pdb_text, plddt_b_factors)
    if spec.structure_target is not None:
        pdb_text = _rewrite_pdb_chain_ids_for_structure_target(
            pdb_text,
            spec.structure_target,
        )
    _progress(f"writing structure artifact {structure_path}")
    write_text_atomic(root / structure_path, pdb_text)

    design_metrics = {
        "target_name": spec.target_name,
        "target_input_mode": (
            "structure"
            if spec.structure_target is not None
            else "sequence"
            if target_sequence_for_design is not None
            else "name"
        ),
        "target_sequence_length": (
            len(target_sequence_for_design)
            if target_sequence_for_design is not None
            else None
        ),
        "binder_scaffold": spec.binder_scaffold,
        "binder_type": binder_code(spec.binder_scaffold or spec.binder_name),
        "framework": spec.binder_framework_name,
        "framework_name": spec.binder_framework_name,
        "framework_source": spec.binder_framework_source,
        "is_antibody": _to_scalar(critic_result.get("is_antibody"))
        if critic_result.get("is_antibody") is not None
        else prompt_plan.is_antibody,
        "cdr_indices": list(prompt_plan.cdr_indices)
        if prompt_plan.cdr_indices
        else None,
        "cdr_lengths": prompt_plan.cdr_lengths or None,
        "cdr_sequences": cdr_sequences or None,
        "binder_name": spec.binder_name,
        "binder_length_range": list(spec.binder_length_range)
        if spec.binder_length_range is not None
        else None,
        "inversion_model_name": inversion_model_name,
        "critic_name": spec.critic_name,
        "steps": spec.steps,
        "final_loss": _to_scalar(critic_result.get("final_loss")),
        "complex_sequence": complex_sequence,
    }
    if binder_target_contact_mode == "mosaic_cdr":
        design_metrics.update(
            {
                "binder_target_contact_mode": binder_target_contact_mode,
                "mosaic_cdr_contact_loss_enabled": True,
                "mosaic_cdr_contact_weight": spec.mosaic_cdr_contact_weight,
                "mosaic_cdr_contact_cutoff_angstrom": (
                    spec.mosaic_cdr_contact_cutoff_angstrom
                ),
                "mosaic_cdr_num_target_contacts": (
                    spec.mosaic_cdr_num_target_contacts
                ),
                "mosaic_framework_contact_penalty_enabled": (
                    spec.mosaic_framework_contact_penalty_weight > 0
                ),
                "mosaic_framework_contact_penalty_weight": (
                    spec.mosaic_framework_contact_penalty_weight
                ),
                "mosaic_framework_contact_penalty_cutoff_angstrom": (
                    spec.mosaic_framework_contact_penalty_cutoff_angstrom
                ),
                "mosaic_framework_contact_probability_threshold": (
                    spec.mosaic_framework_contact_probability_threshold
                ),
                "mosaic_framework_contact_penalty_scope": (
                    spec.mosaic_framework_contact_penalty_scope
                ),
            }
        )
    if designed_target_sequence is not None:
        design_metrics["designed_target_sequence"] = designed_target_sequence
    if spec.structure_target is not None:
        target_spans = _target_chain_spans(spec.structure_target)
        binder_chain_id = _binder_chain_id(
            [chain.canonical_chain_id for chain in spec.structure_target.chains]
        )
        target_hotspots = _target_hotspot_indices_by_chain(spec.structure_target)
        target_global_hotspots = _target_global_hotspot_indices(spec.structure_target)
        design_metrics.update(
            {
                "target_structure": str(spec.structure_target.source_path),
                "binder_chain_id": binder_chain_id,
                "target_chain_id": (
                    spec.structure_target.chains[0].canonical_chain_id
                ),
                "target_chain_ids": [
                    chain.canonical_chain_id for chain in spec.structure_target.chains
                ],
                "target_chains": [
                    {
                        "chain_id": chain.canonical_chain_id,
                        "auth_asym_id": chain.auth_asym_id,
                        "label_asym_id": chain.label_asym_id,
                        "length": len(chain.residues),
                    }
                    for chain in spec.structure_target.chains
                ],
                "target_chain_spans": [
                    {
                        "chain_id": span.chain_id,
                        "start": span.start,
                        "end": span.end,
                        "length": span.length,
                    }
                    for span in target_spans
                ],
                "target_length": sum(
                    len(chain.residues) for chain in spec.structure_target.chains
                ),
                "target_conditioning_mode": spec.conditioning_mode,
                "target_conditioning_assembly": spec.conditioning_assembly,
                "target_conditioning_chain_pairs": (
                    "auto"
                    if spec.conditioning_chain_pairs is None
                    else [list(pair) for pair in spec.conditioning_chain_pairs]
                ),
                "target_hotspot_indices": list(target_global_hotspots),
                "target_hotspots": target_hotspots,
                "target_hotspot_global_indices": list(target_global_hotspots),
                "hotspot_contact_weight": spec.hotspot_contact_weight,
                "hotspot_distogram_contact_cutoff_angstrom": (
                    hotspot_distogram_contact_cutoff_angstrom
                ),
                "hotspot_critic_contact_cutoff_angstrom": (
                    hotspot_critic_contact_cutoff_angstrom
                ),
                "hotspot_contact_cutoff_angstrom": (
                    hotspot_critic_contact_cutoff_angstrom
                ),
                "hotspot_num_contacts": spec.hotspot_num_contacts,
                "hotspot_contact_probability_target": (
                    spec.hotspot_contact_probability_target
                ),
                "hotspot_loss_mode": hotspot_loss_mode,
                "hotspot_contact_loss_enabled": bool(
                    target_global_hotspots and spec.hotspot_contact_weight > 0
                ),
                "target_geometry_drift_enabled": target_geometry_drift.enabled,
                "target_geometry_drift_weight": target_geometry_drift.weight,
                "target_geometry_drift_tolerance_angstrom": (
                    target_geometry_drift.tolerance_angstrom
                ),
                "target_geometry_drift_stiffness_angstrom": (
                    target_geometry_drift.stiffness_angstrom
                ),
                "target_geometry_drift_regions": _selector_map_for_metrics(
                    target_geometry_drift.regions
                ),
                "target_geometry_drift_selected_residue_count": len(
                    target_geometry_drift_indices
                ),
                "target_geometry_drift_selected_pair_count": (
                    _target_geometry_drift_pair_count(target_geometry_drift_indices)
                ),
            }
        )
        if design_run.last_design_fold is not None and binder_design is not None:
            design_metrics.update(
                _hotspot_design_contact_probability_metrics(
                    binder_design,
                    design_run.last_design_fold,
                    hotspot_indices=target_global_hotspots,
                    contact_cutoff_angstrom=hotspot_distogram_contact_cutoff_angstrom,
                    hotspot_num_contacts=spec.hotspot_num_contacts,
                    contact_probability_target=(
                        spec.hotspot_contact_probability_target
                    ),
                    hotspot_loss_mode=hotspot_loss_mode,
                    binder_contact_indices=prompt_plan.cdr_indices or None,
                )
            )
    if (
        binder_target_contact_mode == "mosaic_cdr"
        and design_run.last_design_fold is not None
        and binder_design is not None
        and prompt_plan.cdr_indices
    ):
        design_metrics.update(
            _mosaic_cdr_design_contact_probability_metrics(
                binder_design,
                design_run.last_design_fold,
                cdr_indices=prompt_plan.cdr_indices,
                hotspot_indices=target_global_hotspots,
                contact_cutoff_angstrom=spec.mosaic_cdr_contact_cutoff_angstrom,
                num_target_contacts=spec.mosaic_cdr_num_target_contacts,
                framework_contact_penalty_weight=(
                    spec.mosaic_framework_contact_penalty_weight
                ),
                framework_contact_penalty_cutoff_angstrom=(
                    spec.mosaic_framework_contact_penalty_cutoff_angstrom
                ),
                framework_contact_probability_threshold=(
                    spec.mosaic_framework_contact_probability_threshold
                ),
                framework_contact_penalty_scope=(
                    spec.mosaic_framework_contact_penalty_scope
                ),
            )
        )

    critic_metrics = _extract_metrics(critic_result, steps=spec.steps)
    if design_run.last_confidence_fold is not None:
        critic_metrics.update(
            _confidence_scalar_metrics_from_capture(design_run.last_confidence_fold)
        )
        critic_metrics.update(
            _plddt_metrics_from_capture(design_run.last_confidence_fold)
        )
    if (
        spec.structure_target is not None
        and design_run.last_confidence_fold is not None
    ):
        critic_metrics.update(
            _binder_target_iptm_metrics_from_capture(
                spec.structure_target,
                design_run.last_confidence_fold,
                complex_iptm=critic_metrics.get("iptm"),
            )
        )
        critic_metrics.update(
            _target_geometry_metrics_from_capture(
                spec.structure_target,
                design_run.last_confidence_fold,
            )
        )
        if target_geometry_drift.enabled:
            critic_metrics.update(
                _target_geometry_drift_metrics_from_capture(
                    spec.structure_target,
                    design_run.last_confidence_fold,
                    target_indices=target_geometry_drift_indices,
                )
            )
        critic_metrics.update(
            _hotspot_metrics_from_capture(
                spec.structure_target,
                design_run.last_confidence_fold,
                contact_cutoff_angstrom=hotspot_critic_contact_cutoff_angstrom,
                binder_contact_indices=prompt_plan.cdr_indices or None,
            )
        )
    _progress(f"finished artifact candidate={spec.candidate_id}")
    return DesignCandidateArtifact(
        candidate_id=spec.candidate_id,
        designed_sequence=designed_sequence,
        sequence_path=None,
        critic_name=spec.critic_name,
        structure_path=structure_path.as_posix(),
        design_metrics=design_metrics,
        critic_metrics=critic_metrics,
    )


def preflight_models(
    *,
    esm_repo: str | Path | None,
    gpu_id: str | None,
    inversion_model_name: str = DEFAULT_ESMFOLD2_INVERSION_MODEL,
    critic_name: str,
    steps: int = 1,
    disable_hf_xet: bool = True,
) -> ModelPreflightResult:
    """Load the model set used by the selected binder-design backend."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    _disable_hf_xet_if_requested(disable_hf_xet)
    _restrict_cuda_if_requested(gpu_id)
    if _design_backend() == "local":
        _progress("loading ESM runtime helpers for local model preflight")
        runtime = load_esm_folding_runtime(esm_repo)
        _progress(
            "loading ESMFold2/ESMC models for local preflight; first run may "
            "download large checkpoints and spend several minutes before GPU "
            "memory increases"
        )
        runtime_models = _load_local_runtime_models(
            runtime,
            inversion_model_name=inversion_model_name,
            critic_name=critic_name,
        )
        _progress("local model preflight loaded requested model set")
        return ModelPreflightResult(
            inversion_model_name=inversion_model_name,
            critic_name=critic_name,
            loaded_inversion_models=sorted(runtime_models.inversion_models.keys()),
            loaded_critic_models=sorted(runtime_models.critic_models.keys()),
            esmc_loaded=bool(runtime_models.esmc_model),
        )

    _progress("loading ESM tutorial binder_design module for model preflight")
    binder_design = load_binder_design_module(esm_repo)
    _progress(
        "loading ESMFold2/ESMC models for preflight; first run may download "
        "large checkpoints and spend several minutes before GPU memory increases"
    )
    app = _load_tutorial_app(
        binder_design,
        inversion_model_name=inversion_model_name,
        critic_name=critic_name,
        steps=steps,
    )
    _progress("model preflight loaded requested model set")
    return ModelPreflightResult(
        inversion_model_name=inversion_model_name,
        critic_name=critic_name,
        loaded_inversion_models=sorted(app.inversion_models.keys()),
        loaded_critic_models=sorted(app.hf_critic_models.keys()),
        esmc_loaded=bool(getattr(app, "esmc_model", None)),
    )


def _load_tutorial_app(
    binder_design,
    *,
    inversion_model_name: str,
    critic_name: str,
    steps: int,
):
    binder_design.STEPS = steps
    binder_design.LOG_INTERVAL = max(1, min(5, steps))

    app = binder_design.ESMFold2Design()
    app.inversion_model_names = [inversion_model_name]
    app.hero_critic_hf_paths = [critic_name]
    app.scaling_critic_hf_paths = []
    app.load(use_scaling_critics=False)
    return app


def _load_local_runtime_models(
    binder_design,
    *,
    inversion_model_name: str,
    critic_name: str,
) -> RuntimeModels:
    inversion_spec, critic_spec = _local_model_load_specs(
        inversion_model_name=inversion_model_name,
        critic_name=critic_name,
    )
    loaded_models: dict[_LocalModelLoadSpec, Any] = {}

    def load_model(model_spec: _LocalModelLoadSpec):
        if model_spec not in loaded_models:
            loaded_models[model_spec] = _load_local_hf_esmfold2_model(
                binder_design,
                model_spec.model_name,
                lm_dropout=model_spec.lm_dropout,
                cache_esmc=model_spec.cache_esmc,
                device=model_spec.device,
            )
        return loaded_models[model_spec]

    inversion_models = {
        inversion_model_name: load_model(inversion_spec),
    }
    critic_models = {
        critic_name: load_model(critic_spec),
    }
    if getattr(binder_design, "COMPILE", False):
        compile_model = getattr(binder_design, "_apply_torch_compile", None)
        if compile_model is None:
            raise RuntimeError("ESM runtime requested COMPILE but has no compiler hook")
        compiled_model_ids: set[int] = set()
        for model in inversion_models.values():
            model_id = id(model)
            if model_id in compiled_model_ids:
                continue
            compile_model(model)
            compiled_model_ids.add(model_id)

    esmc_model = binder_design.ESMCForMaskedLM.from_pretrained(
        "biohub/ESMC-6B",
        torch_dtype=binder_design.torch.float32,
    )
    esmc_model = esmc_model.cuda().eval().requires_grad_(False)
    return RuntimeModels(
        inversion_models=inversion_models,
        critic_models=critic_models,
        esmc_model=esmc_model,
        helpers={},
    )


def _get_or_load_local_design_runtime(spec: DesignSpec) -> _LocalDesignRuntime:
    if _local_runtime_cache_disabled():
        _progress("local ESMFold2/ESMC runtime cache disabled for this process")
        return _load_local_design_runtime(spec)

    key = _local_runtime_cache_key(spec)
    cached = _LOCAL_DESIGN_RUNTIME_CACHE.get(key)
    if cached is not None:
        _progress(
            "using cached ESMFold2/ESMC models for local design loop "
            f"gpu={spec.gpu_id or os.environ.get('CUDA_VISIBLE_DEVICES') or 'default'}"
        )
        return cached

    runtime = _load_local_design_runtime(spec)
    _LOCAL_DESIGN_RUNTIME_CACHE[key] = runtime
    return runtime


def _load_local_design_runtime(spec: DesignSpec) -> _LocalDesignRuntime:
    _progress("loading ESM runtime helpers for local design")
    binder_design = load_esm_folding_runtime(spec.esm_repo)
    _progress("loading ESMFold2/ESMC models for local design loop")
    runtime_models = _load_local_runtime_models(
        binder_design,
        inversion_model_name=(
            spec.inversion_model_name or DEFAULT_ESMFOLD2_INVERSION_MODEL
        ),
        critic_name=spec.critic_name,
    )
    return _LocalDesignRuntime(
        binder_design=binder_design,
        runtime_models=runtime_models,
    )


def _local_runtime_cache_disabled() -> bool:
    value = os.environ.get(_LOCAL_RUNTIME_CACHE_DISABLE_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _local_runtime_cache_key(spec: DesignSpec) -> _LocalRuntimeCacheKey:
    inversion_model_name = spec.inversion_model_name or DEFAULT_ESMFOLD2_INVERSION_MODEL
    inversion_spec, critic_spec = _local_model_load_specs(
        inversion_model_name=inversion_model_name,
        critic_name=spec.critic_name,
    )
    return _LocalRuntimeCacheKey(
        esm_repo=_local_runtime_esm_repo_key(spec.esm_repo),
        gpu_id=str(spec.gpu_id) if spec.gpu_id is not None else None,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        inversion_model=inversion_spec,
        critic_model=critic_spec,
    )


def _local_runtime_esm_repo_key(esm_repo: str | Path | None) -> str | None:
    if esm_repo is None:
        return None
    return str(Path(esm_repo).expanduser().resolve())


def _local_model_load_specs(
    *,
    inversion_model_name: str,
    critic_name: str,
) -> tuple[_LocalModelLoadSpec, _LocalModelLoadSpec]:
    return (
        _LocalModelLoadSpec(
            model_name=inversion_model_name,
            lm_dropout=_LOCAL_INVERSION_LM_DROPOUT,
            cache_esmc=_LOCAL_MODEL_CACHE_ESMC,
            device=_LOCAL_MODEL_DEVICE,
        ),
        _LocalModelLoadSpec(
            model_name=critic_name,
            lm_dropout=_LOCAL_CRITIC_LM_DROPOUT,
            cache_esmc=_LOCAL_MODEL_CACHE_ESMC,
            device=_LOCAL_MODEL_DEVICE,
        ),
    )


def _load_local_hf_esmfold2_model(
    binder_design,
    model_name: str,
    *,
    lm_dropout: float,
    cache_esmc: bool,
    device: str,
):
    global _LOCAL_ESMC_CACHE

    repo_id = f"biohub/{model_name}"
    model = binder_design.ESMFold2ExperimentalModel.from_pretrained(
        repo_id,
        load_esmc=not cache_esmc,
    )
    if cache_esmc:
        if _LOCAL_ESMC_CACHE is None:
            model.load_esmc(model.config.esmc_id)
            _LOCAL_ESMC_CACHE = model._esmc
        else:
            model._esmc = _LOCAL_ESMC_CACHE
    model.configure_lm_dropout(
        lm_dropout,
        force_lm_dropout_during_inference=True,
    )
    kernel_backend = (
        "cuequivariance" if getattr(binder_design, "CUE_AVAILABLE", False) else None
    )
    model.set_kernel_backend(kernel_backend)
    return model.to(device=device).eval().requires_grad_(False)


def _select_critic_result(results: list[dict[str, Any]], critic_name: str) -> dict[str, Any]:
    for result in results:
        if result.get("critic_name") == critic_name:
            return result
    return results[0]


def _split_complex_sequence(sequence: str) -> tuple[str | None, str]:
    parts = sequence.rsplit("|", 1)
    if len(parts) == 1:
        return None, sequence
    target, binder = parts
    if not binder:
        raise RuntimeError("designed complex sequence has an empty binder sequence")
    return target, binder


def _prepare_binder_prompt_plan(
    binder_design,
    *,
    binder_name: str,
    binder_scaffold: str | None,
    binder_framework_name: str | None,
    binder_framework_source: str | None,
    binder_framework_template: str | None,
    binder_framework_cdr_lengths: dict[str, tuple[int, int]] | None,
    binder_framework_sequence: str | None,
    binder_framework_cdr_indices: tuple[int, ...] | None,
    seed: int,
    is_antibody: bool | None,
) -> _BinderPromptPlan:
    factories = getattr(binder_design, "BINDER_PROMPT_FACTORIES", None)
    return prepare_binder_prompt_plan(
        binder_name=binder_name,
        binder_scaffold=binder_scaffold,
        binder_framework_name=binder_framework_name,
        binder_framework_source=binder_framework_source,
        binder_framework_template=binder_framework_template,
        binder_framework_cdr_lengths=binder_framework_cdr_lengths,
        binder_framework_sequence=binder_framework_sequence,
        binder_framework_cdr_indices=binder_framework_cdr_indices,
        seed=seed,
        is_antibody=is_antibody,
        binder_prompt_factories=factories if isinstance(factories, dict) else None,
    )


def _sample_antibody_template(
    template: str,
    cdr_lengths: dict[str, tuple[int, int]],
    *,
    seed: int,
) -> str:
    return sample_antibody_template(template, cdr_lengths, seed=seed)


def _sample_scfv_template(
    template: str,
    cdr_lengths: dict[str, tuple[int, int]],
    *,
    seed: int,
) -> str:
    return sample_scfv_template(template, cdr_lengths, seed=seed)


def _antibody_cdr_report_names(binder_type: str) -> tuple[str, ...]:
    return antibody_cdr_report_names(binder_type)


def _mutable_run_lengths(
    prompt: str,
    *,
    cdr_names: tuple[str, ...] = _SCFV_CDR_RUN_NAMES,
) -> dict[str, int]:
    return mutable_run_lengths(prompt, cdr_names=cdr_names)


def _mutable_run_sequences(
    sequence: str,
    cdr_indices: tuple[int, ...],
    *,
    cdr_names: tuple[str, ...] = _SCFV_CDR_RUN_NAMES,
) -> dict[str, str]:
    return mutable_run_sequences(sequence, cdr_indices, cdr_names=cdr_names)


def _contiguous_mutable_runs(prompt: str) -> list[tuple[int, int]]:
    return contiguous_mutable_runs(prompt)


def _contiguous_index_runs(indices: tuple[int, ...]) -> list[tuple[int, int]]:
    return contiguous_index_runs(indices)


def _cdr_prompt_from_indices(
    sequence: str,
    cdr_indices: tuple[int, ...],
) -> str:
    return cdr_prompt_from_indices(sequence, cdr_indices)


def _scfv_cdr_prompt_from_indices(
    sequence: str,
    cdr_indices: tuple[int, ...],
) -> str:
    return scfv_cdr_prompt_from_indices(sequence, cdr_indices)


def _structure_target_sequence(structure_target: PreparedTarget) -> str:
    return "".join(chain.sequence for chain in structure_target.chains)


def _target_chain_spans(structure_target: PreparedTarget) -> tuple[_ChainSpan, ...]:
    spans: list[_ChainSpan] = []
    start = 0
    for chain in structure_target.chains:
        end = start + len(chain.residues)
        spans.append(
            _ChainSpan(
                chain_id=chain.canonical_chain_id,
                auth_asym_id=chain.auth_asym_id,
                label_asym_id=chain.label_asym_id,
                start=start,
                end=end,
            )
        )
        start = end
    return tuple(spans)


def _target_hotspot_indices_by_chain(
    structure_target: PreparedTarget,
) -> dict[str, list[int]]:
    return {
        chain.canonical_chain_id: list(chain.hotspot_indices)
        for chain in structure_target.chains
        if chain.hotspot_indices
    }


def _target_global_hotspot_indices(structure_target: PreparedTarget) -> tuple[int, ...]:
    offset_by_chain = {
        span.chain_id: span.start for span in _target_chain_spans(structure_target)
    }
    global_indices: list[int] = []
    for chain in structure_target.chains:
        offset = offset_by_chain[chain.canonical_chain_id]
        global_indices.extend(offset + index for index in chain.hotspot_indices)
    return tuple(sorted(global_indices))


def _framework_penalty_hotspot_indices(
    *,
    hotspot_indices: tuple[int, ...],
    scope: str,
) -> tuple[int, ...]:
    if scope == "auto":
        return hotspot_indices
    if scope == "hotspot":
        if not hotspot_indices:
            raise ValueError(
                "mosaic_framework_contact_penalty_scope=hotspot requires "
                "target.hotspots"
            )
        return hotspot_indices
    if scope == "target_all":
        return ()
    choices = ", ".join(sorted(MOSAIC_FRAMEWORK_CONTACT_PENALTY_SCOPES))
    raise ValueError(
        f"mosaic_framework_contact_penalty_scope must be one of: {choices}"
    )


def _framework_penalty_target_scope_label(
    hotspot_indices: tuple[int, ...],
) -> str:
    return "target_hotspots" if hotspot_indices else "target_all"


def _extract_metrics(result: dict[str, Any], *, steps: int) -> dict[str, float | int | str | None]:
    metrics: dict[str, float | int | str | None] = {"steps": steps}
    for key in (
        "iptm",
        "ptm",
        "plddt",
        "distogram_iptm_proxy",
        "cdr_distogram_iptm_proxy",
        "final_loss",
    ):
        if key in result:
            metrics[key] = _to_scalar(result[key])
    return metrics


def _confidence_scalar_metrics_from_capture(
    fold_result: dict[str, Any],
) -> dict[str, float | int | str | None]:
    metrics: dict[str, float | int | str | None] = {}
    for key in ("ptm",):
        value = fold_result.get(key)
        if value is None:
            output = fold_result.get("output") or {}
            value = output.get(key)
        scalar = _to_scalar(value)
        if scalar is not None:
            metrics[key] = scalar
    return metrics


def _plddt_metrics_from_capture(fold_result: dict[str, Any]) -> dict[str, Any]:
    values = _plddt_b_factors_from_capture(fold_result)
    if values is None:
        return {}

    target_length, binder_length = _fold_target_binder_lengths(fold_result)
    complex_values = values[: target_length + binder_length]
    target_values = values[:target_length] if target_length else np.asarray([])
    binder_values = (
        values[target_length : target_length + binder_length]
        if binder_length
        else np.asarray([])
    )
    complex_mean = _finite_mean(complex_values)
    metrics: dict[str, Any] = {
        "plddt": complex_mean,
        "plddt_complex": complex_mean,
        "plddt_target": _finite_mean(target_values),
        "plddt_binder": _finite_mean(binder_values),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def _plddt_b_factors_from_capture(
    fold_result: dict[str, Any] | None,
) -> np.ndarray | None:
    if fold_result is None:
        return None
    plddt = fold_result.get("plddt")
    if plddt is None:
        output = fold_result.get("output") or {}
        plddt = output.get("plddt")
    if plddt is None:
        return None

    values = _to_numpy_array(plddt).astype(float)
    while values.ndim > 1:
        values = values[0]
    values = values.reshape(-1)
    target_length, binder_length = _fold_target_binder_lengths(fold_result)
    total_length = target_length + binder_length
    if total_length > 0:
        values = values[:total_length]
    return _scale_plddt_to_100(values)


def _fold_target_binder_lengths(fold_result: dict[str, Any]) -> tuple[int, int]:
    seq_list = fold_result.get("seq_list") or []
    if not seq_list:
        return 0, 0
    target_sequence, binder_sequence = _split_complex_sequence(str(seq_list[0]))
    target_length = len(target_sequence.replace("|", "")) if target_sequence else 0
    binder_length = len(binder_sequence)
    return target_length, binder_length


def _scale_plddt_to_100(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmax(finite)) <= 1.5:
        return values * 100.0
    return values


def _finite_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _rewrite_pdb_b_factors(
    pdb_text: str,
    plddt_per_residue: np.ndarray,
) -> str:
    if plddt_per_residue.size == 0:
        return pdb_text

    lines: list[str] = []
    residue_key: tuple[str, str, str] | None = None
    residue_index = -1
    for raw_line in pdb_text.splitlines(keepends=True):
        line = raw_line
        if raw_line.startswith(("ATOM  ", "HETATM")):
            newline = "\n" if raw_line.endswith("\n") else ""
            body = raw_line[:-1] if newline else raw_line
            body = body.ljust(66)
            current_key = (body[21], body[22:26], body[26])
            if current_key != residue_key:
                residue_index += 1
                residue_key = current_key
            if residue_index < plddt_per_residue.size:
                value = float(plddt_per_residue[residue_index])
                if np.isfinite(value):
                    body = f"{body[:60]}{value:6.2f}{body[66:]}"
                    line = f"{body}{newline}"
        lines.append(line)
    return "".join(lines)


def _binder_target_iptm_metrics_from_capture(
    structure_target: PreparedTarget,
    fold_result: dict[str, Any],
    *,
    complex_iptm: Any,
) -> dict[str, Any]:
    """Scope iPTM to binder-target chain pairs when ESMFold2 exposes pair scores."""

    metrics: dict[str, Any] = {}
    raw_complex_iptm = _to_scalar(complex_iptm)
    if raw_complex_iptm is not None:
        metrics["complex_iptm"] = raw_complex_iptm

    pair_chains_iptm = fold_result.get("pair_chains_iptm")
    if pair_chains_iptm is None:
        output = fold_result.get("output") or {}
        pair_chains_iptm = output.get("pair_chains_iptm")
    if pair_chains_iptm is None:
        if raw_complex_iptm is not None:
            metrics["iptm_scope"] = "complex"
        return metrics

    matrix = _to_numpy_array(pair_chains_iptm).astype(float)
    if matrix.ndim == 3:
        matrix = matrix[0]
    if matrix.ndim != 2:
        if raw_complex_iptm is not None:
            metrics["iptm_scope"] = "complex"
        return metrics

    target_chains = list(structure_target.chains)
    binder_index = len(target_chains)
    if matrix.shape[0] <= binder_index or matrix.shape[1] <= binder_index:
        if raw_complex_iptm is not None:
            metrics["iptm_scope"] = "complex"
        return metrics

    by_chain: dict[str, float] = {}
    weighted_values: list[tuple[float, int]] = []
    for target_index, chain in enumerate(target_chains):
        pair_values = [
            matrix[binder_index, target_index],
            matrix[target_index, binder_index],
        ]
        finite_values = [
            float(value)
            for value in pair_values
            if np.isfinite(value)
        ]
        if not finite_values:
            continue
        value = float(np.mean(finite_values))
        by_chain[chain.canonical_chain_id] = value
        weighted_values.append((value, len(chain.residues)))

    if not weighted_values:
        if raw_complex_iptm is not None:
            metrics["iptm_scope"] = "complex"
        return metrics

    total_weight = sum(weight for _value, weight in weighted_values)
    binder_target_iptm = sum(
        value * weight for value, weight in weighted_values
    ) / total_weight
    binder_target_iptm_unweighted = float(
        np.mean([value for value, _weight in weighted_values])
    )
    metrics.update(
        {
            "iptm": float(binder_target_iptm),
            "iptm_scope": "binder_target",
            "binder_target_iptm": float(binder_target_iptm),
            "binder_target_iptm_unweighted": binder_target_iptm_unweighted,
            "binder_target_iptm_by_chain": by_chain,
        }
    )
    return metrics


def _to_scalar(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (float, int, str)):
        return value
    return str(value)


def _restrict_cuda_if_requested(gpu_id: str | None) -> None:
    if gpu_id is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def _disable_hf_xet_if_requested(disable_hf_xet: bool) -> None:
    if disable_hf_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"


class _FoldCapture:
    def __init__(self) -> None:
        self.last_design_fold: dict[str, Any] | None = None
        self.last_confidence_fold: dict[str, Any] | None = None


@contextmanager
def _patched_binder_length_range(
    binder_design,
    *,
    binder_name: str,
    length_range: tuple[int, int] | None,
) -> Iterator[None]:
    if length_range is None:
        yield
        return
    if binder_name != "minibinder":
        raise ValueError("binder_length_range is currently only supported for minibinder")

    factories = getattr(binder_design, "BINDER_PROMPT_FACTORIES", None)
    if not isinstance(factories, dict) or binder_name not in factories:
        raise ValueError(f"binder factory not found: {binder_name}")
    factory = factories[binder_name]
    original_ranges = dict(getattr(factory, "length_ranges"))
    patched_ranges = dict(original_ranges)
    patched_ranges["seq"] = length_range
    try:
        patched_factory = replace(factory, length_ranges=patched_ranges)
    except TypeError:
        factory.length_ranges = patched_ranges
        try:
            yield
        finally:
            factory.length_ranges = original_ranges
        return

    factories[binder_name] = patched_factory
    try:
        yield
    finally:
        factories[binder_name] = factory


@contextmanager
def _patched_antibody_cdr_indices(
    binder_design,
    *,
    cdr_indices: tuple[int, ...],
    binder_length: int | None,
) -> Iterator[None]:
    if not cdr_indices or binder_length is None:
        yield
        return

    original = getattr(binder_design, "_cdr_indices", None)
    if original is None:
        yield
        return

    def cdr_indices_for_template_position(binder_sequence: str) -> list[int]:
        if len(binder_sequence) == binder_length:
            return list(cdr_indices)
        return original(binder_sequence)

    binder_design._cdr_indices = cdr_indices_for_template_position
    try:
        yield
    finally:
        binder_design._cdr_indices = original


@contextmanager
def _patched_structure_losses_for_target_geometry_drift(
    binder_design,
    *,
    structure_target: PreparedTarget | None,
    drift_config: TargetGeometryDriftConfig,
    selected_indices: tuple[int, ...],
) -> Iterator[None]:
    if not drift_config.enabled or drift_config.weight <= 0:
        yield
        return
    if structure_target is None:
        raise ValueError("target geometry drift loss requires a structure target")
    target_length = sum(len(chain.residues) for chain in structure_target.chains)
    reference_distances = _target_reference_distance_matrix(structure_target)
    pair_mask = _target_geometry_drift_pair_mask(
        selected_indices,
        target_length=target_length,
    )
    if not pair_mask.any():
        raise ValueError("target geometry drift loss selected no valid residue pairs")

    original = binder_design.compute_structure_losses

    def compute_structure_losses(distogram_logits, binder_length: int) -> dict:
        losses = original(distogram_logits, binder_length)
        drift_loss, drift_rmse = _compute_target_geometry_drift_loss(
            binder_design,
            distogram_logits,
            binder_length,
            reference_distances=reference_distances,
            pair_mask=pair_mask,
            tolerance_angstrom=drift_config.tolerance_angstrom,
            stiffness_angstrom=drift_config.stiffness_angstrom,
        )
        losses["target_geometry_drift_loss"] = drift_loss
        losses["target_geometry_drift_rmse"] = drift_rmse
        losses["total_loss"] = losses["total_loss"] + drift_config.weight * drift_loss
        return losses

    binder_design.compute_structure_losses = compute_structure_losses
    try:
        yield
    finally:
        binder_design.compute_structure_losses = original


def _compute_target_geometry_drift_loss(
    binder_design,
    distogram_logits,
    binder_length: int,
    *,
    reference_distances: np.ndarray,
    pair_mask: np.ndarray,
    tolerance_angstrom: float,
    stiffness_angstrom: float,
):
    return design_losses.compute_target_geometry_drift_loss(
        binder_design.torch,
        distogram_logits,
        binder_length,
        reference_distances=reference_distances,
        pair_mask=pair_mask,
        tolerance_angstrom=tolerance_angstrom,
        stiffness_angstrom=stiffness_angstrom,
        bin_distances=binder_design.get_mid_points(),
    )


def _compute_target_geometry_drift_hinge_loss(
    torch,
    drift_rmse,
    *,
    tolerance_angstrom: float,
    stiffness_angstrom: float,
):
    return design_losses.compute_target_geometry_drift_hinge_loss(
        torch,
        drift_rmse,
        tolerance_angstrom=tolerance_angstrom,
        stiffness_angstrom=stiffness_angstrom,
    )


def _target_reference_distance_matrix(structure_target: PreparedTarget) -> np.ndarray:
    coords = experimental_representative_coords(structure_target).astype(np.float32)
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1)).astype(np.float32)


def _target_geometry_drift_pair_mask(
    selected_indices: tuple[int, ...],
    *,
    target_length: int,
) -> np.ndarray:
    selected = np.zeros(target_length, dtype=bool)
    for index in selected_indices:
        if index < 0 or index >= target_length:
            raise ValueError(
                f"target geometry drift index {index} is outside target length "
                f"{target_length}"
            )
        selected[index] = True
    mask = np.logical_and(selected[:, None], selected[None, :])
    return np.triu(mask, k=1)


def _target_geometry_drift_pair_count(selected_indices: tuple[int, ...]) -> int:
    count = len(set(selected_indices))
    return count * (count - 1) // 2


def _selector_map_for_metrics(
    selectors: dict[str, tuple[str, ...]] | None,
) -> dict[str, list[str]] | None:
    if not selectors:
        return None
    return {chain: list(values) for chain, values in selectors.items()}


@contextmanager
def _patched_structure_losses_for_hotspots(
    binder_design,
    *,
    structure_target: PreparedTarget | None,
    hotspot_contact_weight: float,
    hotspot_distogram_contact_cutoff_angstrom: float,
    hotspot_num_contacts: int,
    hotspot_contact_probability_target: float,
    hotspot_loss_mode: str,
    binder_contact_indices: tuple[int, ...] | None = None,
) -> Iterator[None]:
    hotspot_indices = (
        _target_global_hotspot_indices(structure_target)
        if structure_target is not None
        else ()
    )
    if not hotspot_indices or hotspot_contact_weight <= 0:
        yield
        return

    original = binder_design.compute_structure_losses

    def compute_structure_losses(distogram_logits, binder_length: int) -> dict:
        losses = original(distogram_logits, binder_length)
        hotspot_loss = _compute_hotspot_contact_loss(
            binder_design,
            distogram_logits,
            binder_length,
            hotspot_indices=hotspot_indices,
            contact_cutoff_angstrom=hotspot_distogram_contact_cutoff_angstrom,
            hotspot_num_contacts=hotspot_num_contacts,
            contact_probability_target=hotspot_contact_probability_target,
            hotspot_loss_mode=hotspot_loss_mode,
            binder_contact_indices=binder_contact_indices,
        )
        losses["hotspot_contact_loss"] = hotspot_loss
        losses["total_loss"] = (
            losses["total_loss"] + hotspot_contact_weight * hotspot_loss
        )
        return losses

    binder_design.compute_structure_losses = compute_structure_losses
    try:
        yield
    finally:
        binder_design.compute_structure_losses = original


@contextmanager
def _patched_structure_losses_for_mosaic_cdr(
    binder_design,
    *,
    structure_target: PreparedTarget | None,
    cdr_indices: tuple[int, ...],
    mosaic_cdr_contact_weight: float,
    mosaic_cdr_contact_cutoff_angstrom: float,
    mosaic_cdr_num_target_contacts: int,
    mosaic_framework_contact_penalty_weight: float,
    mosaic_framework_contact_penalty_cutoff_angstrom: float,
    mosaic_framework_contact_probability_threshold: float,
    mosaic_framework_contact_penalty_scope: str,
) -> Iterator[None]:
    if not cdr_indices:
        raise ValueError("mosaic_cdr contact mode requires CDR contact indices")
    hotspot_indices = (
        _target_global_hotspot_indices(structure_target)
        if structure_target is not None
        else ()
    )

    original = binder_design.compute_structure_losses

    def compute_structure_losses(distogram_logits, binder_length: int) -> dict:
        losses = original(distogram_logits, binder_length)
        if "inter_contact_loss" not in losses:
            raise RuntimeError(
                "mosaic_cdr contact mode requires the tutorial backend loss dict "
                "to include inter_contact_loss so legacy binder-target attraction "
                "can be removed"
            )
        losses["total_loss"] = (
            losses["total_loss"]
            - design_losses.LOSS_WEIGHTS["inter_contact"]
            * losses["inter_contact_loss"]
        )
        mosaic_loss = _compute_mosaic_cdr_contact_loss(
            binder_design,
            distogram_logits,
            binder_length,
            cdr_indices=cdr_indices,
            contact_cutoff_angstrom=mosaic_cdr_contact_cutoff_angstrom,
            num_target_contacts=mosaic_cdr_num_target_contacts,
            hotspot_indices=hotspot_indices,
        )
        losses["mosaic_cdr_contact_loss"] = mosaic_loss
        losses["total_loss"] = (
            losses["total_loss"] + mosaic_cdr_contact_weight * mosaic_loss
        )
        if mosaic_framework_contact_penalty_weight > 0:
            framework_penalty_hotspot_indices = _framework_penalty_hotspot_indices(
                hotspot_indices=hotspot_indices,
                scope=mosaic_framework_contact_penalty_scope,
            )
            framework_penalty = _compute_framework_contact_penalty_loss(
                binder_design,
                distogram_logits,
                binder_length,
                cdr_indices=cdr_indices,
                contact_cutoff_angstrom=(
                    mosaic_framework_contact_penalty_cutoff_angstrom
                ),
                num_target_contacts=mosaic_cdr_num_target_contacts,
                contact_probability_threshold=(
                    mosaic_framework_contact_probability_threshold
                ),
                hotspot_indices=framework_penalty_hotspot_indices,
            )
            losses["mosaic_framework_contact_penalty_loss"] = framework_penalty
            losses["total_loss"] = (
                losses["total_loss"]
                + mosaic_framework_contact_penalty_weight * framework_penalty
            )
        return losses

    binder_design.compute_structure_losses = compute_structure_losses
    try:
        yield
    finally:
        binder_design.compute_structure_losses = original


def _compute_hotspot_contact_loss(
    binder_design,
    distogram_logits,
    binder_length: int,
    *,
    hotspot_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    hotspot_num_contacts: int,
    contact_probability_target: float,
    hotspot_loss_mode: str,
    binder_contact_indices: tuple[int, ...] | None = None,
):
    if hotspot_loss_mode == "probability_hinge":
        return _compute_hotspot_contact_probability_hinge_loss(
            binder_design,
            distogram_logits,
            binder_length,
            hotspot_indices=hotspot_indices,
            contact_cutoff_angstrom=contact_cutoff_angstrom,
            hotspot_num_contacts=hotspot_num_contacts,
            contact_probability_target=contact_probability_target,
            binder_contact_indices=binder_contact_indices,
        )
    if hotspot_loss_mode == "entropy_hotspot":
        return _compute_hotspot_entropy_contact_loss(
            binder_design,
            distogram_logits,
            binder_length,
            hotspot_indices=hotspot_indices,
            contact_cutoff_angstrom=contact_cutoff_angstrom,
            hotspot_num_contacts=hotspot_num_contacts,
            binder_contact_indices=binder_contact_indices,
        )

    choices = ", ".join(sorted(HOTSPOT_LOSS_MODES))
    raise ValueError(f"hotspot_loss_mode must be one of: {choices}")


def _compute_hotspot_entropy_contact_loss(
    binder_design,
    distogram_logits,
    binder_length: int,
    *,
    hotspot_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    hotspot_num_contacts: int,
    binder_contact_indices: tuple[int, ...] | None = None,
):
    return design_losses.compute_hotspot_entropy_contact_loss(
        binder_design.torch,
        distogram_logits,
        binder_length,
        hotspot_indices=hotspot_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        hotspot_num_contacts=hotspot_num_contacts,
        binder_contact_indices=binder_contact_indices,
        bin_distances=_distogram_bin_midpoints(binder_design),
    )


def _compute_hotspot_contact_probability_hinge_loss(
    binder_design,
    distogram_logits,
    binder_length: int,
    *,
    hotspot_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    hotspot_num_contacts: int,
    contact_probability_target: float,
    binder_contact_indices: tuple[int, ...] | None = None,
):
    return design_losses.compute_hotspot_contact_probability_hinge_loss(
        binder_design.torch,
        distogram_logits,
        binder_length,
        hotspot_indices=hotspot_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        hotspot_num_contacts=hotspot_num_contacts,
        contact_probability_target=contact_probability_target,
        binder_contact_indices=binder_contact_indices,
        bin_distances=_distogram_bin_midpoints(binder_design),
    )


def _hotspot_contact_probability_scores(
    binder_design,
    distogram_logits,
    binder_length: int,
    *,
    hotspot_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    hotspot_num_contacts: int,
    binder_contact_indices: tuple[int, ...] | None = None,
):
    return design_losses.hotspot_contact_probability_scores(
        binder_design.torch,
        distogram_logits,
        binder_length,
        hotspot_indices=hotspot_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        hotspot_num_contacts=hotspot_num_contacts,
        binder_contact_indices=binder_contact_indices,
        bin_distances=_distogram_bin_midpoints(binder_design),
    )


def _compute_mosaic_cdr_contact_loss(
    binder_design,
    distogram_logits,
    binder_length: int,
    *,
    cdr_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    num_target_contacts: int,
    hotspot_indices: tuple[int, ...] = (),
):
    return design_losses.compute_mosaic_cdr_contact_loss(
        binder_design.torch,
        distogram_logits,
        binder_length,
        cdr_indices=cdr_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        num_target_contacts=num_target_contacts,
        hotspot_indices=hotspot_indices,
        bin_distances=_distogram_bin_midpoints(binder_design),
    )


def _mosaic_cdr_contact_probability_scores(
    binder_design,
    distogram_logits,
    binder_length: int,
    *,
    cdr_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    num_target_contacts: int,
    hotspot_indices: tuple[int, ...] = (),
):
    return design_losses.mosaic_cdr_contact_probability_scores(
        binder_design.torch,
        distogram_logits,
        binder_length,
        cdr_indices=cdr_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        num_target_contacts=num_target_contacts,
        hotspot_indices=hotspot_indices,
        bin_distances=_distogram_bin_midpoints(binder_design),
    )


def _compute_framework_contact_penalty_loss(
    binder_design,
    distogram_logits,
    binder_length: int,
    *,
    cdr_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    num_target_contacts: int,
    contact_probability_threshold: float,
    hotspot_indices: tuple[int, ...] = (),
):
    return design_losses.compute_framework_contact_penalty_loss(
        binder_design.torch,
        distogram_logits,
        binder_length,
        cdr_indices=cdr_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        num_target_contacts=num_target_contacts,
        contact_probability_threshold=contact_probability_threshold,
        hotspot_indices=hotspot_indices,
        bin_distances=_distogram_bin_midpoints(binder_design),
    )


def _framework_contact_probability_scores(
    binder_design,
    distogram_logits,
    binder_length: int,
    *,
    cdr_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    num_target_contacts: int,
    hotspot_indices: tuple[int, ...] = (),
):
    return design_losses.framework_contact_probability_scores(
        binder_design.torch,
        distogram_logits,
        binder_length,
        cdr_indices=cdr_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        num_target_contacts=num_target_contacts,
        hotspot_indices=hotspot_indices,
        bin_distances=_distogram_bin_midpoints(binder_design),
    )


def _distogram_bin_midpoints(binder_design):
    get_mid_points = getattr(binder_design, "get_mid_points", None)
    if callable(get_mid_points):
        return get_mid_points()
    return design_losses.get_mid_points(binder_design.torch)


def _binder_contact_mask(
    torch,
    *,
    full_len: int,
    target_length: int,
    binder_length: int,
    binder_contact_indices: tuple[int, ...] | None,
    device,
):
    return design_losses.binder_contact_mask(
        torch,
        full_len=full_len,
        target_length=target_length,
        binder_length=binder_length,
        binder_contact_indices=binder_contact_indices,
        device=device,
    )


def _validate_binder_contact_indices(
    binder_contact_indices: tuple[int, ...],
    binder_length: int,
) -> None:
    design_losses.validate_binder_contact_indices(
        binder_contact_indices,
        binder_length,
    )


@contextmanager
def _patched_fold_with_distogram_conditioning(
    binder_design,
    *,
    structure_target: PreparedTarget | None,
    enabled: bool,
    condition_assembly: bool,
    conditioning_chain_pairs: tuple[tuple[str, str], ...] | None,
    capture: _FoldCapture,
) -> Iterator[None]:
    original = getattr(binder_design, "fold_and_get_distogram", None)
    if original is None:
        yield
        return

    def conditioned_fold_and_get_distogram(
        model,
        target_seq: str,
        target_one_hot,
        design,
        num_loops: int = 0,
        num_sampling_steps: int = 1,
        calculate_confidence: bool = False,
        seed: int | None = None,
    ) -> dict:
        if structure_target is None:
            result = original(
                model,
                target_seq,
                target_one_hot,
                design,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                calculate_confidence=calculate_confidence,
                seed=seed,
            )
        else:
            expected_target_seq = _structure_target_sequence(structure_target)
            if target_seq != expected_target_seq:
                raise ValueError(
                    "structure target sequence changed during design: "
                    f"expected length {len(expected_target_seq)}, got {len(target_seq)}"
                )
            result = _fold_and_get_distogram_for_structure_target(
                binder_design,
                model,
                target_one_hot,
                design,
                structure_target=structure_target,
                condition_distograms=enabled,
                condition_assembly=condition_assembly,
                conditioning_chain_pairs=conditioning_chain_pairs,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                calculate_confidence=calculate_confidence,
                seed=seed,
            )
        if num_loops == 1:
            capture.last_design_fold = result
        elif calculate_confidence:
            capture.last_confidence_fold = result
        return result

    binder_design.fold_and_get_distogram = conditioned_fold_and_get_distogram
    try:
        yield
    finally:
        binder_design.fold_and_get_distogram = original


def _fold_and_get_distogram_for_sequence_target(
    binder_design,
    model,
    target_seq: str,
    target_one_hot,
    design,
    *,
    num_loops: int = 0,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
) -> dict:
    from esm.models.esmfold2 import (  # type: ignore
        ProteinInput,
        StructurePredictionInput,
    )

    torch = binder_design.torch
    padded_design = binder_design.F.pad(design, (2, 11), mode="constant", value=0)

    token_lists = torch.argmax(padded_design, dim=-1)
    designed_sequences = [
        "".join(
            binder_design.PROTEIN_3TO1[binder_design.TOKENS[int(token.item())]]
            for token in token_list
        )
        for token_list in token_lists
    ]
    seq_list = [
        f"{target_seq}|{binder_sequence}"
        for binder_sequence in designed_sequences
    ]
    max_atoms = (
        None
        if len(seq_list) == 1
        else ((len(seq_list[0]) - 1) * 14) // 32 * 32
    )

    inputs_list = []
    for sequence in seq_list:
        sequences = {
            chain_sequence: [str(index)]
            for index, chain_sequence in enumerate(sequence.split("|"))
        }
        inputs_raw = StructurePredictionInput(
            sequences=[
                ProteinInput(id=chain_id, sequence=chain_sequence, msa=None)
                for chain_sequence, chain_id in sequences.items()
            ]
        )
        inputs_list.append(
            binder_design.prepare_esmfold2_tensors(inputs_raw, max_atoms=max_atoms)
        )

    inputs = {
        key: torch.stack([inp[key] for inp in inputs_list], dim=0).cuda()
        for key in inputs_list[0]
    }
    inputs["res_type_soft"] = torch.cat(
        (target_one_hot.repeat(design.size(0), 1, 1), padded_design), dim=1
    )

    with binder_design.seed_context(seed):
        output = model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=num_sampling_steps,
            num_loops=num_loops,
            calculate_confidence=calculate_confidence,
            seed=seed,
        )

    result: dict = {
        "distogram_logits": output["distogram_logits"],
        "inputs": inputs,
        "inputs_list": inputs_list,
        "output": output,
        "seq_list": seq_list,
    }
    if calculate_confidence:
        result.update(
            {
                "ptm": output.get("ptm"),
                "iptm": output.get("iptm"),
                "plddt": output.get("plddt"),
            }
        )
    return result


def _fold_and_get_distogram_for_structure_target(
    binder_design,
    model,
    target_one_hot,
    design,
    *,
    structure_target: PreparedTarget,
    condition_distograms: bool,
    condition_assembly: bool,
    conditioning_chain_pairs: tuple[tuple[str, str], ...] | None,
    num_loops: int = 0,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
) -> dict:
    from esm.models.esmfold2 import (  # type: ignore
        ProteinInput,
        StructurePredictionInput,
    )

    torch = binder_design.torch
    padding = (2, 11)
    padded_design = binder_design.F.pad(design, padding, mode="constant", value=0)
    target_chains = list(structure_target.chains)
    target_chain_ids = [chain.canonical_chain_id for chain in target_chains]
    target_sequences = [chain.sequence for chain in target_chains]
    target_length = sum(len(sequence) for sequence in target_sequences)
    if target_one_hot.shape[1] != target_length:
        raise ValueError(
            "target one-hot length does not match prepared target chains: "
            f"{target_one_hot.shape[1]} != {target_length}"
        )

    token_lists = torch.argmax(padded_design, dim=-1)
    designed_seq = [
        "".join(
            binder_design.PROTEIN_3TO1[binder_design.TOKENS[int(tkn.item())]]
            for tkn in token_list
        )
        for token_list in token_lists
    ]
    seq_list = ["|".join([*target_sequences, seq]) for seq in designed_seq]
    binder_length = len(designed_seq[0])
    total_length = target_length + binder_length
    max_atoms = None if len(seq_list) == 1 else (total_length * 14) // 32 * 32

    inputs_list = []
    binder_chain_id = _binder_chain_id(target_chain_ids)
    assembly_pairs = (
        _resolve_assembly_chain_pairs(structure_target, conditioning_chain_pairs)
        if condition_distograms and condition_assembly
        else ()
    )
    for binder_sequence in designed_seq:
        inputs_raw = StructurePredictionInput(
            sequences=[
                ProteinInput(id=chain_id, sequence=sequence, msa=None)
                for chain_id, sequence in zip(target_chain_ids, target_sequences)
            ]
            + [
                ProteinInput(id=binder_chain_id, sequence=binder_sequence, msa=None),
            ],
            distogram_conditioning=None,
        )
        inputs_list.append(
            binder_design.prepare_esmfold2_tensors(inputs_raw, max_atoms=max_atoms)
        )
        if condition_distograms:
            _apply_target_distogram_conditioning(
                inputs_list[-1],
                structure_target=structure_target,
                assembly_pairs=assembly_pairs,
            )

    _inspect_structure_target_distogram_tensors(
        inputs_list[0],
        structure_target=structure_target,
        binder_length=binder_length,
        expect_conditioned=condition_distograms,
        assembly_pairs=assembly_pairs,
    )
    inputs = {
        key: torch.stack([inp[key] for inp in inputs_list], dim=0).cuda()
        for key in inputs_list[0]
    }
    inputs["res_type_soft"] = torch.cat(
        (target_one_hot.repeat(design.size(0), 1, 1), padded_design), dim=1
    )

    template_pair_bias = _build_distogram_template_pair_bias(
        binder_design,
        model,
        inputs,
        structure_target=structure_target,
        binder_length=binder_length,
        enabled=condition_distograms,
        assembly_pairs=assembly_pairs,
    )
    with binder_design.seed_context(seed):
        with _patched_model_folding_trunk_with_distogram_template(
            model,
            template_pair_bias,
        ):
            output = model(
                **inputs,
                num_diffusion_samples=1,
                num_sampling_steps=num_sampling_steps,
                num_loops=num_loops,
                calculate_confidence=calculate_confidence,
                seed=seed,
            )

    result: dict = {
        "distogram_logits": output["distogram_logits"],
        "inputs": inputs,
        "inputs_list": inputs_list,
        "output": output,
        "seq_list": seq_list,
    }
    if calculate_confidence:
        result.update(
            {
                "ptm": output.get("ptm"),
                "iptm": output.get("iptm"),
                "pair_chains_iptm": output.get("pair_chains_iptm"),
                "plddt": output.get("plddt"),
            }
        )
    return result


def _inspect_structure_target_distogram_tensors(
    features: dict[str, Any],
    *,
    structure_target: PreparedTarget,
    binder_length: int,
    expect_conditioned: bool,
    assembly_pairs: tuple[tuple[Any, Any], ...] = (),
) -> None:
    disto_cond = _to_numpy_array(features["disto_cond"])
    disto_cond_mask = _to_numpy_array(features["disto_cond_mask"]).astype(bool)
    if disto_cond.ndim != 2 or disto_cond_mask.ndim != 2:
        raise ValueError("distogram conditioning tensors must be rank-2")
    if disto_cond.shape != disto_cond_mask.shape:
        raise ValueError(
            f"disto_cond shape {disto_cond.shape} does not match mask "
            f"{disto_cond_mask.shape}"
        )
    if disto_cond.shape[0] != disto_cond.shape[1]:
        raise ValueError(f"disto_cond must be square, got {disto_cond.shape}")

    spans = _target_chain_spans(structure_target)
    target_length = sum(span.length for span in spans)
    total_length = target_length + binder_length
    if disto_cond.shape != (total_length, total_length):
        raise ValueError(
            f"distogram conditioning shape {disto_cond.shape} does not match "
            f"target+binder length {(total_length, total_length)}"
        )

    expected = np.zeros_like(disto_cond_mask, dtype=bool)
    if expect_conditioned:
        _distances, expected = _target_distogram_conditioning_matrix(
            structure_target,
            binder_length=binder_length,
            assembly_pairs=assembly_pairs,
        )
    if not np.array_equal(disto_cond_mask, expected):
        actual_true = int(disto_cond_mask.sum())
        expected_true = int(expected.sum())
        target_binder_true = int(disto_cond_mask[:target_length, target_length:].sum())
        off_chain_true = int(
            disto_cond_mask[:target_length, :target_length].sum() - expected_true
        )
        raise ValueError(
            "unexpected distogram conditioning mask: "
            f"actual_true={actual_true}, expected_true={expected_true}, "
            f"target_binder_true={target_binder_true}, "
            f"off_chain_target_true={off_chain_true}"
        )


def _target_distogram_conditioning_matrix(
    structure_target: PreparedTarget,
    *,
    binder_length: int,
    assembly_pairs: tuple[tuple[Any, Any], ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    spans = _target_chain_spans(structure_target)
    target_length = sum(span.length for span in spans)
    total_length = target_length + binder_length
    distances = np.zeros((total_length, total_length), dtype=np.float32)
    mask = np.zeros((total_length, total_length), dtype=bool)

    span_by_chain = {span.chain_id: span for span in spans}
    for chain in structure_target.chains:
        span = span_by_chain[chain.canonical_chain_id]
        distogram = np.asarray(chain.distogram, dtype=np.float32)
        expected_shape = (span.length, span.length)
        if distogram.shape != expected_shape:
            raise ValueError(
                "target chain distogram shape does not match prepared residues: "
                f"{chain.canonical_chain_id} {distogram.shape} != {expected_shape}"
            )
        distogram_mask = np.asarray(chain.distogram_mask, dtype=bool)
        if distogram_mask.shape != expected_shape:
            raise ValueError(
                "target chain distogram mask shape does not match prepared residues: "
                f"{chain.canonical_chain_id} {distogram_mask.shape} != {expected_shape}"
            )
        distances[span.start : span.end, span.start : span.end] = distogram
        mask[span.start : span.end, span.start : span.end] = distogram_mask

    for chain_a, chain_b in assembly_pairs:
        span_a = span_by_chain[chain_a.canonical_chain_id]
        span_b = span_by_chain[chain_b.canonical_chain_id]
        pair_distogram = _compute_pair_distogram_array(chain_a, chain_b)
        pair_mask = _compute_pair_distogram_mask_array(chain_a, chain_b)
        expected_shape = (span_a.length, span_b.length)
        if pair_distogram.shape != expected_shape:
            raise ValueError(
                "assembly distogram shape does not match prepared residues: "
                f"{chain_a.canonical_chain_id}-{chain_b.canonical_chain_id} "
                f"{pair_distogram.shape} != {expected_shape}"
            )
        if pair_mask.shape != expected_shape:
            raise ValueError(
                "assembly distogram mask shape does not match prepared residues: "
                f"{chain_a.canonical_chain_id}-{chain_b.canonical_chain_id} "
                f"{pair_mask.shape} != {expected_shape}"
            )
        distances[span_a.start : span_a.end, span_b.start : span_b.end] = (
            pair_distogram
        )
        distances[span_b.start : span_b.end, span_a.start : span_a.end] = (
            pair_distogram.T
        )
        mask[span_a.start : span_a.end, span_b.start : span_b.end] = pair_mask
        mask[span_b.start : span_b.end, span_a.start : span_a.end] = pair_mask.T

    return distances, mask


def _build_distogram_template_pair_bias(
    binder_design,
    model,
    inputs: dict[str, Any],
    *,
    structure_target: PreparedTarget,
    binder_length: int,
    enabled: bool,
    assembly_pairs: tuple[tuple[Any, Any], ...] = (),
):
    if not enabled:
        return None

    distances_np, mask_np = _target_distogram_conditioning_matrix(
        structure_target,
        binder_length=binder_length,
        assembly_pairs=assembly_pairs,
    )
    if not mask_np.any():
        return None

    torch = binder_design.torch
    res_type_soft = inputs["res_type_soft"]
    batch_size = int(res_type_soft.shape[0])
    total_length = int(res_type_soft.shape[1])
    if distances_np.shape != (total_length, total_length):
        raise ValueError(
            "template distogram shape does not match model input length: "
            f"{distances_np.shape} != {(total_length, total_length)}"
        )

    if os.environ.get(_TEMPLATE_DISTOGRAM_INJECTION_DISABLE_ENV):
        _log_template_distogram_injection_once(
            model,
            enabled=False,
            reason=f"disabled by {_TEMPLATE_DISTOGRAM_INJECTION_DISABLE_ENV}",
            batch_size=batch_size,
            total_length=total_length,
            masked_pair_count=int(mask_np.sum()),
            assembly_pair_count=len(assembly_pairs),
        )
        return None

    embedding = _distogram_template_embedding(model)
    confidence_head = getattr(model, "confidence_head", None)
    boundaries = getattr(confidence_head, "boundaries", None)
    if boundaries is None:
        raise RuntimeError(
            "target distogram conditioning requires "
            "model.confidence_head.boundaries"
        )

    device = res_type_soft.device
    with torch.no_grad():
        distances = torch.as_tensor(
            distances_np,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device).unsqueeze(0)
        if batch_size != 1:
            distances = distances.repeat(batch_size, 1, 1)
            mask = mask.repeat(batch_size, 1, 1)
        boundaries = boundaries.to(device=device, dtype=distances.dtype)
        bins = (distances.unsqueeze(-1) > boundaries).sum(dim=-1).long()
        pair_bias = embedding(bins).detach()
        pair_bias = pair_bias * mask.unsqueeze(-1).to(dtype=pair_bias.dtype)
        pair_bias_float = pair_bias.float()
        active_values = pair_bias_float[mask.unsqueeze(-1).expand_as(pair_bias_float)]
        pair_bias_l2 = float(pair_bias_float.norm().item())
        pair_bias_abs_mean = (
            float(active_values.abs().mean().item())
            if int(active_values.numel())
            else 0.0
        )
    _log_template_distogram_injection_once(
        model,
        enabled=True,
        reason="using confidence_head.dist_bin_pairwise_embed",
        batch_size=batch_size,
        total_length=total_length,
        masked_pair_count=int(mask_np.sum()),
        assembly_pair_count=len(assembly_pairs),
        pair_bias_l2=pair_bias_l2,
        pair_bias_abs_mean=pair_bias_abs_mean,
    )
    return pair_bias


def _distogram_template_embedding(model):
    confidence_head = getattr(model, "confidence_head", None)
    embedding = getattr(confidence_head, "dist_bin_pairwise_embed", None)
    if embedding is None:
        raise RuntimeError(
            "target distogram conditioning requires "
            "model.confidence_head.dist_bin_pairwise_embed"
        )
    return embedding


def _log_template_distogram_injection_once(
    model,
    *,
    enabled: bool,
    reason: str,
    batch_size: int,
    total_length: int,
    masked_pair_count: int,
    assembly_pair_count: int,
    pair_bias_l2: float | None = None,
    pair_bias_abs_mean: float | None = None,
) -> None:
    key = (
        enabled,
        reason,
        batch_size,
        total_length,
        masked_pair_count,
        assembly_pair_count,
    )
    seen = getattr(model, _TEMPLATE_DISTOGRAM_LOG_KEYS_ATTR, set())
    if key in seen:
        return
    seen = set(seen)
    seen.add(key)
    setattr(model, _TEMPLATE_DISTOGRAM_LOG_KEYS_ATTR, seen)
    if enabled:
        _LOGGER.info(
            "target distogram injection enabled: reason=%s batch_size=%d "
            "total_length=%d masked_pairs=%d assembly_pairs=%d "
            "pair_bias_l2=%.4f pair_bias_abs_mean=%.4f",
            reason,
            batch_size,
            total_length,
            masked_pair_count,
            assembly_pair_count,
            0.0 if pair_bias_l2 is None else pair_bias_l2,
            0.0 if pair_bias_abs_mean is None else pair_bias_abs_mean,
        )
    else:
        _LOGGER.info(
            "target distogram injection disabled: reason=%s batch_size=%d "
            "total_length=%d masked_pairs=%d assembly_pairs=%d",
            reason,
            batch_size,
            total_length,
            masked_pair_count,
            assembly_pair_count,
        )


@contextmanager
def _patched_model_folding_trunk_with_distogram_template(
    model,
    template_pair_bias,
) -> Iterator[None]:
    if template_pair_bias is None:
        yield
        return

    folding_trunk = getattr(model, "folding_trunk", None)
    original_forward = getattr(folding_trunk, "forward", None)
    if original_forward is None:
        raise RuntimeError(
            "target distogram conditioning requires model.folding_trunk.forward"
        )

    def forward_with_template(pair, *args, **kwargs):
        if tuple(pair.shape[:3]) != tuple(template_pair_bias.shape[:3]):
            raise ValueError(
                "template distogram pair bias shape does not match folding trunk "
                f"input: {template_pair_bias.shape[:3]} != {pair.shape[:3]}"
            )
        bias = template_pair_bias.to(device=pair.device, dtype=pair.dtype)
        return original_forward(pair + bias, *args, **kwargs)

    folding_trunk.forward = forward_with_template
    try:
        yield
    finally:
        folding_trunk.forward = original_forward


def _apply_target_distogram_conditioning(
    features: dict[str, Any],
    *,
    structure_target: PreparedTarget,
    assembly_pairs: tuple[tuple[Any, Any], ...],
) -> None:
    chain_tokens = _token_indices_by_target_chain(features, structure_target)
    for chain in structure_target.chains:
        tokens = chain_tokens[chain.canonical_chain_id]
        distogram = np.asarray(chain.distogram, dtype=np.float32)
        mask = np.asarray(chain.distogram_mask, dtype=bool)
        if distogram.shape != (len(tokens), len(tokens)):
            raise ValueError(
                "target chain distogram shape does not match prepared target tokens: "
                f"{chain.canonical_chain_id} {distogram.shape} != "
                f"{(len(tokens), len(tokens))}"
            )
        if mask.shape != (len(tokens), len(tokens)):
            raise ValueError(
                "target chain distogram mask shape does not match prepared target tokens: "
                f"{chain.canonical_chain_id} {mask.shape} != "
                f"{(len(tokens), len(tokens))}"
            )
        binned = _bin_assembly_distogram(distogram)
        _assign_matrix_block(features["disto_cond"], tokens, tokens, binned)
        _assign_matrix_block(features["disto_cond_mask"], tokens, tokens, mask)

    _apply_assembly_distogram_conditioning(
        features,
        structure_target=structure_target,
        assembly_pairs=assembly_pairs,
    )


def _apply_assembly_distogram_conditioning(
    features: dict[str, Any],
    *,
    structure_target: PreparedTarget,
    assembly_pairs: tuple[tuple[Any, Any], ...],
) -> None:
    chain_tokens = _token_indices_by_target_chain(features, structure_target)
    for chain_a, chain_b in assembly_pairs:
        tokens_a = chain_tokens[chain_a.canonical_chain_id]
        tokens_b = chain_tokens[chain_b.canonical_chain_id]
        pair_distogram = _compute_pair_distogram_array(chain_a, chain_b)
        pair_mask = _compute_pair_distogram_mask_array(chain_a, chain_b)
        if pair_distogram.shape != (len(tokens_a), len(tokens_b)):
            raise ValueError(
                "assembly distogram shape does not match prepared target tokens: "
                f"{chain_a.canonical_chain_id}-{chain_b.canonical_chain_id} "
                f"{pair_distogram.shape} != {(len(tokens_a), len(tokens_b))}"
            )
        if pair_mask.shape != (len(tokens_a), len(tokens_b)):
            raise ValueError(
                "assembly distogram mask shape does not match prepared target tokens: "
                f"{chain_a.canonical_chain_id}-{chain_b.canonical_chain_id} "
                f"{pair_mask.shape} != {(len(tokens_a), len(tokens_b))}"
            )
        binned = _bin_assembly_distogram(pair_distogram)
        _assign_matrix_block(features["disto_cond"], tokens_a, tokens_b, binned)
        _assign_matrix_block(features["disto_cond"], tokens_b, tokens_a, binned.T)
        _assign_matrix_block(features["disto_cond_mask"], tokens_a, tokens_b, pair_mask)
        _assign_matrix_block(features["disto_cond_mask"], tokens_b, tokens_a, pair_mask.T)


def _token_indices_by_target_chain(
    features: dict[str, Any],
    structure_target: PreparedTarget,
) -> dict[str, np.ndarray]:
    asym_id = _to_numpy_array(features["asym_id"]).astype(int)
    if asym_id.ndim != 1:
        raise ValueError(f"asym_id must be rank-1 before batching, got {asym_id.shape}")
    result: dict[str, np.ndarray] = {}
    for asym_index, chain in enumerate(structure_target.chains):
        indices = np.flatnonzero(asym_id == asym_index).astype(np.int64)
        if indices.shape[0] != len(chain.residues):
            raise ValueError(
                f"prepared token count for chain {chain.canonical_chain_id} "
                f"{indices.shape[0]} does not match target length {len(chain.residues)}"
            )
        result[chain.canonical_chain_id] = indices
    return result


def _resolve_assembly_chain_pairs(
    structure_target: PreparedTarget,
    requested_pairs: tuple[tuple[str, str], ...] | None,
) -> tuple[tuple[Any, Any], ...]:
    if requested_pairs is None:
        chains = list(structure_target.chains)
        return tuple(
            (chain_a, chain_b)
            for left_index, chain_a in enumerate(chains)
            for chain_b in chains[left_index + 1 :]
        )

    pairs = []
    seen: set[tuple[str, str]] = set()
    for left, right in requested_pairs:
        chain_a = _resolve_target_chain(structure_target, left)
        chain_b = _resolve_target_chain(structure_target, right)
        if chain_a.canonical_chain_id == chain_b.canonical_chain_id:
            raise ValueError(
                "conditioning_chain_pairs cannot pair a target chain with itself: "
                f"{left}, {right}"
            )
        key = tuple(sorted((chain_a.canonical_chain_id, chain_b.canonical_chain_id)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((chain_a, chain_b))
    return tuple(pairs)


def _resolve_target_chain(structure_target: PreparedTarget, requested: str):
    matches = [
        chain
        for chain in structure_target.chains
        if requested
        in {
            chain.canonical_chain_id,
            chain.auth_asym_id,
            chain.label_asym_id,
        }
    ]
    if not matches:
        available = ", ".join(
            f"{chain.canonical_chain_id}(auth={chain.auth_asym_id},label={chain.label_asym_id})"
            for chain in structure_target.chains
        )
        raise ValueError(
            f"conditioning_chain_pairs references unknown chain {requested}; "
            f"available: {available}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"conditioning_chain_pairs chain {requested} is ambiguous; "
            "use a canonical chain id from check output"
        )
    return matches[0]


def _compute_pair_distogram_array(chain_a, chain_b) -> np.ndarray:
    coords_a = _representative_coords_for_chain(chain_a)
    coords_b = _representative_coords_for_chain(chain_b)
    delta = coords_a[:, None, :] - coords_b[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1)).astype(np.float32)


def _compute_pair_distogram_mask_array(chain_a, chain_b) -> np.ndarray:
    mask_a = _representative_coord_mask_for_chain(chain_a)
    mask_b = _representative_coord_mask_for_chain(chain_b)
    return np.logical_and(mask_a[:, None], mask_b[None, :])


def _representative_coords_for_chain(chain) -> np.ndarray:
    coords = []
    for residue in chain.residues:
        coords.append(residue.representative_coord or (0.0, 0.0, 0.0))
    return np.asarray(coords, dtype=np.float32)


def _representative_coord_mask_for_chain(chain) -> np.ndarray:
    if hasattr(chain, "representative_coord_mask"):
        return np.asarray(chain.representative_coord_mask, dtype=bool)
    return np.asarray(
        [residue.representative_coord is not None for residue in chain.residues],
        dtype=bool,
    )


def _bin_assembly_distogram(
    distogram: np.ndarray,
    *,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
) -> np.ndarray:
    boundaries = np.linspace(min_dist, max_dist, num_bins + 1, dtype=np.float32)
    binned = np.searchsorted(boundaries[:-1], distogram, side="left") - 1
    return np.clip(binned, 0, num_bins - 1).astype(np.int64)


def _assign_matrix_block(
    matrix,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
) -> None:
    if hasattr(matrix, "detach"):
        torch = __import__("torch")
        row_index = torch.as_tensor(rows, dtype=torch.long, device=matrix.device)
        col_index = torch.as_tensor(cols, dtype=torch.long, device=matrix.device)
        value_tensor = torch.as_tensor(values, dtype=matrix.dtype, device=matrix.device)
        matrix[row_index[:, None], col_index[None, :]] = value_tensor
        return
    matrix[np.ix_(rows, cols)] = values


def _target_geometry_metrics_from_capture(
    structure_target: PreparedTarget,
    fold_result: dict[str, Any],
) -> dict[str, float | int]:
    metrics = compute_fold_target_geometry_metrics(
        fold_result["inputs"],
        fold_result["output"],
        target_sequence=_structure_target_sequence(structure_target),
        experimental_coords=experimental_representative_coords(structure_target),
        experimental_coord_mask=np.asarray(
            [
                residue.representative_coord is not None
                for chain in structure_target.chains
                for residue in chain.residues
            ],
            dtype=bool,
        ),
    )
    return {
        "target_distance_rmse": metrics.target_distance_rmse,
        "target_aligned_rmsd": metrics.target_aligned_rmsd,
        "target_residue_count": metrics.target_residue_count,
        **compute_fold_target_geometry_diagnostics(
            fold_result["inputs"],
            fold_result["output"],
            prepared_target=structure_target,
        ),
    }


def _hotspot_design_contact_probability_metrics(
    binder_design,
    fold_result: dict[str, Any],
    *,
    hotspot_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    hotspot_num_contacts: int,
    contact_probability_target: float,
    hotspot_loss_mode: str,
    binder_contact_indices: tuple[int, ...] | None = None,
) -> dict[str, float | int | str | bool]:
    if not hotspot_indices:
        return {}
    binder_sequence = _binder_sequence_from_fold(fold_result)
    scores = _hotspot_contact_probability_scores(
        binder_design,
        fold_result["distogram_logits"],
        len(binder_sequence),
        hotspot_indices=hotspot_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        hotspot_num_contacts=hotspot_num_contacts,
        binder_contact_indices=binder_contact_indices,
    ).detach()
    first_batch_scores = scores[0]
    satisfied = bool((first_batch_scores >= contact_probability_target).all().item())
    return {
        "hotspot_design_contact_probability_max": float(
            first_batch_scores.max().item()
        ),
        "hotspot_design_contact_probability_mean": float(
            first_batch_scores.mean().item()
        ),
        "hotspot_design_contact_probability_min": float(
            first_batch_scores.min().item()
        ),
        "hotspot_design_contact_probability_target": float(
            contact_probability_target
        ),
        "hotspot_design_contact_cutoff_angstrom": float(contact_cutoff_angstrom),
        "hotspot_design_contact_satisfied": satisfied,
        "hotspot_num_contacts": int(hotspot_num_contacts),
        "hotspot_loss_mode": hotspot_loss_mode,
        "hotspot_design_contact_scope": (
            "binder_cdr" if binder_contact_indices is not None else "binder_all"
        ),
    }


def _mosaic_cdr_design_contact_probability_metrics(
    binder_design,
    fold_result: dict[str, Any],
    *,
    cdr_indices: tuple[int, ...],
    hotspot_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    num_target_contacts: int,
    framework_contact_penalty_weight: float,
    framework_contact_penalty_cutoff_angstrom: float,
    framework_contact_probability_threshold: float,
    framework_contact_penalty_scope: str,
) -> dict[str, float | int | str | bool]:
    binder_sequence = _binder_sequence_from_fold(fold_result)
    binder_length = len(binder_sequence)
    scores = _mosaic_cdr_contact_probability_scores(
        binder_design,
        fold_result["distogram_logits"],
        binder_length,
        cdr_indices=cdr_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        num_target_contacts=num_target_contacts,
        hotspot_indices=hotspot_indices,
    ).detach()
    first_batch_scores = scores[0]
    mosaic_loss = _compute_mosaic_cdr_contact_loss(
        binder_design,
        fold_result["distogram_logits"],
        binder_length,
        cdr_indices=cdr_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        num_target_contacts=num_target_contacts,
        hotspot_indices=hotspot_indices,
    ).detach()
    metrics: dict[str, float | int | str | bool] = {
        "mosaic_cdr_contact_scope": (
            "target_hotspots" if hotspot_indices else "target_all"
        ),
        "mosaic_cdr_contact_probability_max": float(
            first_batch_scores.max().item()
        ),
        "mosaic_cdr_contact_probability_mean": float(
            first_batch_scores.mean().item()
        ),
        "mosaic_cdr_contact_probability_min": float(
            first_batch_scores.min().item()
        ),
        "mosaic_cdr_contact_loss": float(mosaic_loss[0].item()),
        "mosaic_cdr_contact_cutoff_angstrom": float(contact_cutoff_angstrom),
        "mosaic_cdr_num_target_contacts": int(num_target_contacts),
    }
    if framework_contact_penalty_weight <= 0:
        return metrics

    framework_penalty_hotspot_indices = _framework_penalty_hotspot_indices(
        hotspot_indices=hotspot_indices,
        scope=framework_contact_penalty_scope,
    )
    framework_scores = _framework_contact_probability_scores(
        binder_design,
        fold_result["distogram_logits"],
        binder_length,
        cdr_indices=cdr_indices,
        contact_cutoff_angstrom=framework_contact_penalty_cutoff_angstrom,
        num_target_contacts=num_target_contacts,
        hotspot_indices=framework_penalty_hotspot_indices,
    ).detach()
    framework_penalty = _compute_framework_contact_penalty_loss(
        binder_design,
        fold_result["distogram_logits"],
        binder_length,
        cdr_indices=cdr_indices,
        contact_cutoff_angstrom=framework_contact_penalty_cutoff_angstrom,
        num_target_contacts=num_target_contacts,
        contact_probability_threshold=framework_contact_probability_threshold,
        hotspot_indices=framework_penalty_hotspot_indices,
    ).detach()
    metrics["mosaic_framework_contact_penalty_scope"] = framework_contact_penalty_scope
    metrics["mosaic_framework_contact_penalty_target_scope"] = (
        _framework_penalty_target_scope_label(framework_penalty_hotspot_indices)
    )
    metrics["mosaic_framework_contact_penalty_loss"] = float(
        framework_penalty[0].item()
    )
    if int(framework_scores.shape[-1]) > 0:
        first_batch_framework = framework_scores[0]
        metrics["mosaic_framework_contact_probability_max"] = float(
            first_batch_framework.max().item()
        )
        metrics["mosaic_framework_contact_probability_mean"] = float(
            first_batch_framework.mean().item()
        )
    return metrics


def _target_geometry_drift_metrics_from_capture(
    structure_target: PreparedTarget,
    fold_result: dict[str, Any],
    *,
    target_indices: tuple[int, ...],
) -> dict[str, float | int]:
    metrics = compute_fold_target_geometry_region_metrics(
        fold_result["inputs"],
        fold_result["output"],
        prepared_target=structure_target,
        target_indices=target_indices,
    )
    return {
        "target_geometry_drift_distance_rmse": metrics.target_distance_rmse,
        "target_geometry_drift_aligned_rmsd": metrics.target_aligned_rmsd,
        "target_geometry_drift_residue_count": metrics.target_residue_count,
    }


def _hotspot_metrics_from_capture(
    structure_target: PreparedTarget,
    fold_result: dict[str, Any],
    *,
    contact_cutoff_angstrom: float,
    binder_contact_indices: tuple[int, ...] | None = None,
) -> dict[str, float | int]:
    hotspot_indices = _target_global_hotspot_indices(structure_target)
    if not hotspot_indices:
        return {}

    metrics = compute_fold_hotspot_contact_metrics(
        fold_result["inputs"],
        fold_result["output"],
        target_sequence=_structure_target_sequence(structure_target),
        binder_sequence=_binder_sequence_from_fold(fold_result),
        hotspot_indices=hotspot_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
    )
    cdr_metrics = None
    if binder_contact_indices is not None:
        cdr_metrics = compute_fold_hotspot_contact_metrics(
            fold_result["inputs"],
            fold_result["output"],
            target_sequence=_structure_target_sequence(structure_target),
            binder_sequence=_binder_sequence_from_fold(fold_result),
            hotspot_indices=hotspot_indices,
            contact_cutoff_angstrom=contact_cutoff_angstrom,
            binder_indices=binder_contact_indices,
        )
    hotspot_by_chain = _hotspot_metrics_by_chain_from_capture(
        structure_target,
        fold_result,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
    )
    result = {
        "hotspot_critic_contact_cutoff_angstrom": (
            metrics.hotspot_contact_cutoff_angstrom
        ),
        "hotspot_contact_cutoff_angstrom": metrics.hotspot_contact_cutoff_angstrom,
        "hotspot_contact_fraction": metrics.hotspot_heavy_atom_contact_fraction,
        "hotspot_satisfaction": metrics.hotspot_heavy_atom_contact_fraction,
        "hotspot_heavy_atom_contact_fraction": metrics.hotspot_heavy_atom_contact_fraction,
        "hotspot_min_heavy_atom_distance_mean": (
            metrics.hotspot_min_heavy_atom_distance_mean
        ),
        "hotspot_min_heavy_atom_distance_min": (
            metrics.hotspot_min_heavy_atom_distance_min
        ),
        "hotspot_min_binder_distance_mean": (
            metrics.hotspot_min_heavy_atom_distance_mean
        ),
        "hotspot_min_binder_distance_min": metrics.hotspot_min_heavy_atom_distance_min,
        "hotspot_representative_contact_fraction": (
            metrics.hotspot_representative_contact_fraction
        ),
        "hotspot_min_representative_distance_mean": (
            metrics.hotspot_min_representative_distance_mean
        ),
        "hotspot_min_representative_distance_min": (
            metrics.hotspot_min_representative_distance_min
        ),
        "hotspot_count": metrics.hotspot_count,
        "hotspot_by_chain": hotspot_by_chain,
    }
    if cdr_metrics is not None:
        result.update(
            {
                "cdr_hotspot_pass": bool(
                    cdr_metrics.hotspot_heavy_atom_contact_fraction > 0
                ),
                "cdr_hotspot_distance_angstrom": (
                    cdr_metrics.hotspot_min_heavy_atom_distance_min
                ),
                "cdr_hotspot_contact_fraction": (
                    cdr_metrics.hotspot_heavy_atom_contact_fraction
                ),
                "cdr_hotspot_satisfaction": (
                    cdr_metrics.hotspot_heavy_atom_contact_fraction
                ),
                "cdr_hotspot_heavy_atom_contact_fraction": (
                    cdr_metrics.hotspot_heavy_atom_contact_fraction
                ),
                "cdr_hotspot_min_heavy_atom_distance_mean": (
                    cdr_metrics.hotspot_min_heavy_atom_distance_mean
                ),
                "cdr_hotspot_min_heavy_atom_distance_min": (
                    cdr_metrics.hotspot_min_heavy_atom_distance_min
                ),
                "cdr_hotspot_representative_contact_fraction": (
                    cdr_metrics.hotspot_representative_contact_fraction
                ),
                "cdr_hotspot_min_representative_distance_mean": (
                    cdr_metrics.hotspot_min_representative_distance_mean
                ),
                "cdr_hotspot_min_representative_distance_min": (
                    cdr_metrics.hotspot_min_representative_distance_min
                ),
                "cdr_hotspot_contact_cutoff_angstrom": (
                    cdr_metrics.hotspot_contact_cutoff_angstrom
                ),
                "cdr_hotspot_count": cdr_metrics.hotspot_count,
            }
        )
    return result


def _hotspot_metrics_by_chain_from_capture(
    structure_target: PreparedTarget,
    fold_result: dict[str, Any],
    *,
    contact_cutoff_angstrom: float,
) -> dict[str, dict[str, float | int | list[int]]]:
    spans = {span.chain_id: span for span in _target_chain_spans(structure_target)}
    target_sequence = _structure_target_sequence(structure_target)
    binder_sequence = _binder_sequence_from_fold(fold_result)
    by_chain: dict[str, dict[str, float | int | list[int]]] = {}
    for chain in structure_target.chains:
        if not chain.hotspot_indices:
            continue
        span = spans[chain.canonical_chain_id]
        global_indices = tuple(span.start + index for index in chain.hotspot_indices)
        metrics = compute_fold_hotspot_contact_metrics(
            fold_result["inputs"],
            fold_result["output"],
            target_sequence=target_sequence,
            binder_sequence=binder_sequence,
            hotspot_indices=global_indices,
            contact_cutoff_angstrom=contact_cutoff_angstrom,
        )
        by_chain[chain.canonical_chain_id] = {
            "hotspot_indices": list(chain.hotspot_indices),
            "hotspot_global_indices": list(global_indices),
            "hotspot_contact_cutoff_angstrom": metrics.hotspot_contact_cutoff_angstrom,
            "hotspot_heavy_atom_contact_fraction": (
                metrics.hotspot_heavy_atom_contact_fraction
            ),
            "hotspot_min_heavy_atom_distance_mean": (
                metrics.hotspot_min_heavy_atom_distance_mean
            ),
            "hotspot_min_heavy_atom_distance_min": (
                metrics.hotspot_min_heavy_atom_distance_min
            ),
            "hotspot_representative_contact_fraction": (
                metrics.hotspot_representative_contact_fraction
            ),
            "hotspot_min_representative_distance_mean": (
                metrics.hotspot_min_representative_distance_mean
            ),
            "hotspot_min_representative_distance_min": (
                metrics.hotspot_min_representative_distance_min
            ),
            "hotspot_count": metrics.hotspot_count,
        }
    return by_chain


def _binder_sequence_from_fold(fold_result: dict[str, Any]) -> str:
    seq_list = fold_result.get("seq_list") or []
    if not seq_list:
        raise ValueError("fold result did not include seq_list")
    sequence = str(seq_list[0])
    if "|" not in sequence:
        raise ValueError("fold result sequence does not contain target|binder separator")
    return sequence.rsplit("|", 1)[1]


def _rewrite_pdb_chain_ids_for_structure_target(
    pdb_text: str,
    structure_target: PreparedTarget,
) -> str:
    desired_chain_ids = [
        chain.canonical_chain_id for chain in structure_target.chains
    ] + [
        _binder_chain_id(
            [chain.canonical_chain_id for chain in structure_target.chains]
        )
    ]
    if not _valid_pdb_chain_id_set(desired_chain_ids):
        return pdb_text

    observed_chain_ids: list[str] = []
    seen: set[str] = set()
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) <= 21:
            continue
        chain_id = line[21]
        if chain_id not in seen:
            seen.add(chain_id)
            observed_chain_ids.append(chain_id)

    if len(observed_chain_ids) != len(desired_chain_ids):
        return pdb_text

    chain_map = dict(zip(observed_chain_ids, desired_chain_ids))
    rewritten = []
    for line in pdb_text.splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
            line = f"{line[:21]}{chain_map.get(line[21], line[21])}{line[22:]}"
        rewritten.append(line)
    return "".join(rewritten)


def _valid_pdb_chain_id_set(chain_ids: list[str]) -> bool:
    return (
        len(set(chain_ids)) == len(chain_ids)
        and all(len(chain_id) == 1 and not chain_id.isspace() for chain_id in chain_ids)
    )


def _binder_chain_id(target_chain_ids: str | list[str] | tuple[str, ...]) -> str:
    if isinstance(target_chain_ids, str):
        target_chain_id_set = {target_chain_ids}
    else:
        target_chain_id_set = set(target_chain_ids)
    candidates = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    for candidate in candidates:
        if candidate not in target_chain_id_set:
            return candidate
    joined = ", ".join(sorted(target_chain_id_set))
    raise ValueError(f"could not choose binder chain id distinct from {joined}")


def _to_numpy_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
