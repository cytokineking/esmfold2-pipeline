from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from esmfold2_pipeline.artifacts import write_json_atomic
from esmfold2_pipeline.config import (
    DEFAULT_ESMFOLD2_CRITIC_MODEL,
    CampaignConfig,
    load_campaign_config,
)
from esmfold2_pipeline.esm_adapter.imports import load_binder_design_module
from esmfold2_pipeline.structure import (
    PreparedTarget,
    StructureTargetError,
    parse_structure_target,
    write_target_artifacts,
)


DEFAULT_VALIDATION_BINDER_LENGTH = 64
DEFAULT_VALIDATION_CRITIC = DEFAULT_ESMFOLD2_CRITIC_MODEL


@dataclass(frozen=True)
class ConditioningValidationSettings:
    config_path: Path
    output_dir: Path
    esm_repo: Path | None
    gpu_id: str | None
    critic_name: str
    binder_sequence: str
    num_sampling_steps: int
    num_loops: int
    seed: int | None
    calculate_confidence: bool = False


@dataclass(frozen=True)
class DistogramTensorCheck:
    total_length: int
    target_length: int
    disto_cond_shape: tuple[int, int]
    disto_cond_mask_shape: tuple[int, int]
    disto_cond_mask_true: int
    target_block_true: int
    outside_target_block_true: int


@dataclass(frozen=True)
class FoldGeometryMetrics:
    target_distance_rmse: float
    target_aligned_rmsd: float
    target_residue_count: int


@dataclass(frozen=True)
class HotspotContactMetrics:
    hotspot_heavy_atom_contact_fraction: float
    hotspot_min_heavy_atom_distance_mean: float
    hotspot_min_heavy_atom_distance_min: float
    hotspot_representative_contact_fraction: float
    hotspot_min_representative_distance_mean: float
    hotspot_min_representative_distance_min: float
    hotspot_contact_cutoff_angstrom: float
    hotspot_count: int

    @property
    def hotspot_contact_fraction(self) -> float:
        return self.hotspot_heavy_atom_contact_fraction

    @property
    def hotspot_min_binder_distance_mean(self) -> float:
        return self.hotspot_min_heavy_atom_distance_mean

    @property
    def hotspot_min_binder_distance_min(self) -> float:
        return self.hotspot_min_heavy_atom_distance_min


@dataclass(frozen=True)
class ConditioningValidationResult:
    output_json: Path
    output_dir: Path
    target_chain_id: str
    target_length: int
    binder_chain_id: str
    binder_length: int
    critic_name: str
    baseline_tensor_check: DistogramTensorCheck
    conditioned_tensor_check: DistogramTensorCheck
    baseline_metrics: FoldGeometryMetrics
    conditioned_metrics: FoldGeometryMetrics
    distance_rmse_delta: float
    aligned_rmsd_delta: float
    conditioning_improved_distance_rmse: bool
    conditioning_improved_aligned_rmsd: bool


def compute_fold_target_geometry_metrics(
    inputs: dict[str, Any],
    output: dict[str, Any],
    *,
    target_sequence: str,
    experimental_coords: np.ndarray,
    experimental_coord_mask: np.ndarray | None = None,
) -> FoldGeometryMetrics:
    return _geometry_metrics_from_fold(
        inputs,
        output,
        target_sequence=target_sequence,
        experimental_coords=experimental_coords,
        experimental_coord_mask=experimental_coord_mask,
    )


def compute_fold_target_geometry_region_metrics(
    inputs: dict[str, Any],
    output: dict[str, Any],
    *,
    prepared_target: PreparedTarget,
    target_indices: tuple[int, ...],
) -> FoldGeometryMetrics:
    if len(target_indices) < 2:
        raise ValueError("target_indices must contain at least two residues")
    if len(set(target_indices)) != len(target_indices):
        raise ValueError("target_indices must not contain duplicates")

    target_sequence = "".join(chain.sequence for chain in prepared_target.chains)
    experimental_coords = _experimental_representative_coords(prepared_target)
    experimental_mask = _experimental_representative_coord_mask(prepared_target)
    for index in target_indices:
        if index < 0 or index >= len(target_sequence):
            raise ValueError(
                f"target index {index} is outside target length {len(target_sequence)}"
            )

    predicted_coords = _predicted_representative_coords_by_token(
        inputs,
        output,
        sequence=target_sequence,
    )
    if predicted_coords.shape != experimental_coords.shape:
        raise ValueError(
            f"predicted representative coords {predicted_coords.shape} do not match "
            f"experimental coords {experimental_coords.shape}"
        )

    index_array = np.asarray(target_indices, dtype=np.int64)
    index_array = index_array[experimental_mask[index_array]]
    if index_array.shape[0] < 2:
        raise ValueError("target_indices must contain at least two resolved residues")
    predicted_region = predicted_coords[index_array]
    experimental_region = experimental_coords[index_array]
    return FoldGeometryMetrics(
        target_distance_rmse=_distance_matrix_upper_rmse(
            predicted_region,
            experimental_region,
        ),
        target_aligned_rmsd=_aligned_rmsd(predicted_region, experimental_region),
        target_residue_count=int(len(target_indices)),
    )


def compute_fold_hotspot_contact_metrics(
    inputs: dict[str, Any],
    output: dict[str, Any],
    *,
    target_sequence: str,
    binder_sequence: str,
    hotspot_indices: tuple[int, ...],
    contact_cutoff_angstrom: float = 22.0,
    binder_indices: tuple[int, ...] | None = None,
) -> HotspotContactMetrics:
    return _hotspot_contact_metrics_from_fold(
        inputs,
        output,
        target_sequence=target_sequence,
        binder_sequence=binder_sequence,
        hotspot_indices=hotspot_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        binder_indices=binder_indices,
    )


def compute_fold_target_geometry_diagnostics(
    inputs: dict[str, Any],
    output: dict[str, Any],
    *,
    prepared_target: PreparedTarget,
) -> dict[str, Any]:
    target_sequence = "".join(chain.sequence for chain in prepared_target.chains)
    predicted_coords = _predicted_representative_coords_by_token(
        inputs,
        output,
        sequence=target_sequence,
    )
    experimental_coords = _experimental_representative_coords(prepared_target)
    experimental_mask = _experimental_representative_coord_mask(prepared_target)
    if predicted_coords.shape != experimental_coords.shape:
        raise ValueError(
            f"predicted representative coords {predicted_coords.shape} do not match "
            f"experimental coords {experimental_coords.shape}"
        )

    spans = _target_chain_spans(prepared_target)
    chain_geometry: dict[str, dict[str, float | int]] = {}
    for chain_id, start, end in spans:
        predicted_chain = predicted_coords[start:end]
        experimental_chain = experimental_coords[start:end]
        chain_mask = experimental_mask[start:end]
        resolved_predicted_chain = predicted_chain[chain_mask]
        resolved_experimental_chain = experimental_chain[chain_mask]
        chain_geometry[chain_id] = {
            "distance_rmse": _optional_distance_matrix_rmse(
                resolved_predicted_chain,
                resolved_experimental_chain,
            ),
            "aligned_rmsd": _optional_aligned_rmsd(
                resolved_predicted_chain,
                resolved_experimental_chain,
            ),
            "residue_count": int(chain_mask.sum()),
            "selected_residue_count": int(end - start),
        }

    assembly_geometry: dict[str, dict[str, float | int | None]] = {}
    for left_index, (left_id, left_start, left_end) in enumerate(spans):
        for right_id, right_start, right_end in spans[left_index + 1 :]:
            left_mask = experimental_mask[left_start:left_end]
            right_mask = experimental_mask[right_start:right_end]
            predicted_distances = _pairwise_distances(
                predicted_coords[left_start:left_end][left_mask],
                predicted_coords[right_start:right_end][right_mask],
            )
            experimental_distances = _pairwise_distances(
                experimental_coords[left_start:left_end][left_mask],
                experimental_coords[right_start:right_end][right_mask],
            )
            key = f"{left_id}__{right_id}"
            assembly_geometry[key] = {
                "pair_distance_rmse": _optional_distance_matrix_delta_rmse(
                    predicted_distances,
                    experimental_distances,
                ),
                "contact_recovery_8A": _contact_recovery(
                    predicted_distances,
                    experimental_distances,
                    cutoff=8.0,
                ),
                "contact_recovery_12A": _contact_recovery(
                    predicted_distances,
                    experimental_distances,
                    cutoff=12.0,
                ),
                "residue_pair_count": int(predicted_distances.size),
                "selected_residue_pair_count": int(
                    (left_end - left_start) * (right_end - right_start)
                ),
            }

    return {
        "target_chain_geometry": chain_geometry,
        "target_assembly_geometry": assembly_geometry,
    }


def experimental_representative_coords(prepared_target: PreparedTarget) -> np.ndarray:
    return _experimental_representative_coords(prepared_target)


def default_validation_binder_sequence(
    length: int = DEFAULT_VALIDATION_BINDER_LENGTH,
) -> str:
    if length <= 0:
        raise ValueError("binder length must be positive")
    motif = "GSGSAA"
    repeats = (length + len(motif) - 1) // len(motif)
    return (motif * repeats)[:length]


def validate_conditioning_config(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    esm_repo: str | Path | None = None,
    gpu_id: str | None = None,
    critic_name: str | None = None,
    binder_sequence: str | None = None,
    binder_length: int = DEFAULT_VALIDATION_BINDER_LENGTH,
    num_sampling_steps: int = 1,
    num_loops: int = 0,
    seed: int | None = 0,
    calculate_confidence: bool = False,
) -> ConditioningValidationResult:
    config = load_campaign_config(config_path)
    if config.target_structure is None:
        raise ValueError("validate-conditioning requires target.structure in the config")
    if config.target_structure.conditioning_mode != "distogram":
        raise ValueError("validate-conditioning requires target.conditioning.mode: distogram")
    if num_sampling_steps <= 0:
        raise ValueError("num_sampling_steps must be positive")
    if num_loops < 0:
        raise ValueError("num_loops must be non-negative")

    prepared_target = parse_structure_target(config.target_structure)
    if len(prepared_target.chains) != 1:
        raise StructureTargetError(
            "fold-only distogram validation currently supports exactly one target chain"
        )

    out_dir = _resolve_output_dir(config, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_artifacts = write_target_artifacts(
        prepared_target,
        out_dir / "target",
        conditioning_mode="distogram",
    )
    chain = prepared_target.chains[0]
    distogram_path = (
        target_artifacts.target_dir
        / "conditioning"
        / f"chain_{_safe_name(chain.canonical_chain_id)}_distogram.npy"
    )
    distogram_mask_path = (
        target_artifacts.target_dir
        / "conditioning"
        / f"chain_{_safe_name(chain.canonical_chain_id)}_distogram_mask.npy"
    )
    distogram = np.load(distogram_path).astype(np.float32, copy=False)
    distogram_mask = np.load(distogram_mask_path).astype(bool, copy=False)
    if distogram.shape != (len(chain.residues), len(chain.residues)):
        raise ValueError(
            f"distogram artifact shape {distogram.shape} does not match "
            f"target length {len(chain.residues)}"
        )
    if distogram_mask.shape != distogram.shape:
        raise ValueError(
            f"distogram mask artifact shape {distogram_mask.shape} does not match "
            f"distogram shape {distogram.shape}"
        )

    settings = ConditioningValidationSettings(
        config_path=Path(config_path),
        output_dir=out_dir,
        esm_repo=Path(esm_repo).expanduser() if esm_repo is not None else None,
        gpu_id=gpu_id,
        critic_name=critic_name or config.critic_name or DEFAULT_VALIDATION_CRITIC,
        binder_sequence=binder_sequence
        if binder_sequence is not None
        else default_validation_binder_sequence(binder_length),
        num_sampling_steps=num_sampling_steps,
        num_loops=num_loops,
        seed=seed,
        calculate_confidence=calculate_confidence,
    )
    _validate_protein_sequence(settings.binder_sequence, "binder sequence")

    fold_result = _run_fold_pair(
        settings=settings,
        prepared_target=prepared_target,
        target_distogram=distogram,
        target_distogram_mask=distogram_mask,
    )
    baseline_metrics = _geometry_metrics_from_fold(
        fold_result["baseline_inputs"],
        fold_result["baseline_output"],
        target_sequence=chain.sequence,
        experimental_coords=_experimental_representative_coords(prepared_target),
        experimental_coord_mask=_experimental_representative_coord_mask(prepared_target),
    )
    conditioned_metrics = _geometry_metrics_from_fold(
        fold_result["conditioned_inputs"],
        fold_result["conditioned_output"],
        target_sequence=chain.sequence,
        experimental_coords=_experimental_representative_coords(prepared_target),
        experimental_coord_mask=_experimental_representative_coord_mask(prepared_target),
    )

    distance_delta = (
        conditioned_metrics.target_distance_rmse - baseline_metrics.target_distance_rmse
    )
    rmsd_delta = conditioned_metrics.target_aligned_rmsd - baseline_metrics.target_aligned_rmsd
    result_without_path = {
        "settings": _settings_json(settings),
        "target": {
            "source_path": str(prepared_target.source_path),
            "chain_id": chain.canonical_chain_id,
            "length": len(chain.residues),
            "sequence": chain.sequence,
            "distogram_path": str(distogram_path),
        },
        "baseline_tensor_check": asdict(fold_result["baseline_tensor_check"]),
        "conditioned_tensor_check": asdict(fold_result["conditioned_tensor_check"]),
        "baseline_metrics": asdict(baseline_metrics),
        "conditioned_metrics": asdict(conditioned_metrics),
        "distance_rmse_delta": distance_delta,
        "aligned_rmsd_delta": rmsd_delta,
        "conditioning_improved_distance_rmse": distance_delta < 0,
        "conditioning_improved_aligned_rmsd": rmsd_delta < 0,
    }
    output_json = write_json_atomic(
        out_dir / "conditioning_validation.json",
        result_without_path,
    )
    return ConditioningValidationResult(
        output_json=output_json,
        output_dir=out_dir,
        target_chain_id=chain.canonical_chain_id,
        target_length=len(chain.residues),
        binder_chain_id=_binder_chain_id(chain.canonical_chain_id),
        binder_length=len(settings.binder_sequence),
        critic_name=settings.critic_name,
        baseline_tensor_check=fold_result["baseline_tensor_check"],
        conditioned_tensor_check=fold_result["conditioned_tensor_check"],
        baseline_metrics=baseline_metrics,
        conditioned_metrics=conditioned_metrics,
        distance_rmse_delta=distance_delta,
        aligned_rmsd_delta=rmsd_delta,
        conditioning_improved_distance_rmse=distance_delta < 0,
        conditioning_improved_aligned_rmsd=rmsd_delta < 0,
    )


def inspect_distogram_tensors(
    features: dict[str, Any],
    *,
    target_length: int,
    expect_conditioned: bool,
    expected_target_mask: np.ndarray | None = None,
) -> DistogramTensorCheck:
    disto_cond = _to_numpy(features["disto_cond"])
    disto_cond_mask = _to_numpy(features["disto_cond_mask"]).astype(bool)
    if disto_cond.ndim != 2:
        raise ValueError(f"disto_cond must be rank-2, got shape {disto_cond.shape}")
    if disto_cond_mask.ndim != 2:
        raise ValueError(
            f"disto_cond_mask must be rank-2, got shape {disto_cond_mask.shape}"
        )
    if disto_cond.shape != disto_cond_mask.shape:
        raise ValueError(
            f"disto_cond shape {disto_cond.shape} does not match mask "
            f"{disto_cond_mask.shape}"
        )
    total_length = int(disto_cond.shape[0])
    if disto_cond.shape[0] != disto_cond.shape[1]:
        raise ValueError(f"disto_cond must be square, got shape {disto_cond.shape}")
    if target_length <= 0 or target_length > total_length:
        raise ValueError(
            f"target_length must be in 1..{total_length}, got {target_length}"
        )

    target_block = disto_cond_mask[:target_length, :target_length]
    target_block_true = int(target_block.sum())
    mask_true = int(disto_cond_mask.sum())
    outside_target_block_true = mask_true - target_block_true
    if expected_target_mask is not None:
        expected_target_mask = np.asarray(expected_target_mask, dtype=bool)
        if expected_target_mask.shape != (target_length, target_length):
            raise ValueError(
                f"expected_target_mask shape {expected_target_mask.shape} does not "
                f"match target block {(target_length, target_length)}"
            )
        if not np.array_equal(target_block, expected_target_mask):
            raise ValueError(
                "target distogram conditioning mask does not match expected partial mask"
            )
    else:
        expected_target_true = target_length * target_length if expect_conditioned else 0
        if target_block_true != expected_target_true:
            raise ValueError(
                f"expected {expected_target_true} true distogram mask entries in target "
                f"block, found {target_block_true}"
            )
    if outside_target_block_true != 0:
        raise ValueError(
            "distogram conditioning mask has true entries outside the target-chain block"
        )

    return DistogramTensorCheck(
        total_length=total_length,
        target_length=target_length,
        disto_cond_shape=(int(disto_cond.shape[0]), int(disto_cond.shape[1])),
        disto_cond_mask_shape=(
            int(disto_cond_mask.shape[0]),
            int(disto_cond_mask.shape[1]),
        ),
        disto_cond_mask_true=mask_true,
        target_block_true=target_block_true,
        outside_target_block_true=outside_target_block_true,
    )


def _run_fold_pair(
    *,
    settings: ConditioningValidationSettings,
    prepared_target: PreparedTarget,
    target_distogram: np.ndarray,
    target_distogram_mask: np.ndarray,
) -> dict[str, Any]:
    if settings.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.gpu_id)

    binder_design = load_binder_design_module(settings.esm_repo)
    model = binder_design._load_hf_model(
        settings.critic_name,
        lm_dropout=0.25,
        cache_esmc=True,
        device="cuda",
    )

    baseline_features = _prepare_features(
        binder_design=binder_design,
        prepared_target=prepared_target,
        binder_sequence=settings.binder_sequence,
        target_distogram=None,
    )
    conditioned_features = _prepare_features(
        binder_design=binder_design,
        prepared_target=prepared_target,
        binder_sequence=settings.binder_sequence,
        target_distogram=target_distogram,
        target_distogram_mask=target_distogram_mask,
    )
    target_length = len(prepared_target.chains[0].residues)
    baseline_tensor_check = inspect_distogram_tensors(
        baseline_features,
        target_length=target_length,
        expect_conditioned=False,
    )
    conditioned_tensor_check = inspect_distogram_tensors(
        conditioned_features,
        target_length=target_length,
        expect_conditioned=True,
        expected_target_mask=target_distogram_mask,
    )

    baseline_inputs = _batch_cuda_inputs(baseline_features)
    conditioned_inputs = _batch_cuda_inputs(conditioned_features)
    with binder_design.seed_context(settings.seed):
        baseline_output = model(
            **baseline_inputs,
            num_diffusion_samples=1,
            num_sampling_steps=settings.num_sampling_steps,
            num_loops=settings.num_loops,
            calculate_confidence=settings.calculate_confidence,
            seed=settings.seed,
        )
    with binder_design.seed_context(settings.seed):
        conditioned_output = model(
            **conditioned_inputs,
            num_diffusion_samples=1,
            num_sampling_steps=settings.num_sampling_steps,
            num_loops=settings.num_loops,
            calculate_confidence=settings.calculate_confidence,
            seed=settings.seed,
        )

    return {
        "baseline_inputs": baseline_inputs,
        "conditioned_inputs": conditioned_inputs,
        "baseline_output": baseline_output,
        "conditioned_output": conditioned_output,
        "baseline_tensor_check": baseline_tensor_check,
        "conditioned_tensor_check": conditioned_tensor_check,
    }


def _prepare_features(
    *,
    binder_design,
    prepared_target: PreparedTarget,
    binder_sequence: str,
    target_distogram: np.ndarray | None,
    target_distogram_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    from esm.models.esmfold2 import (  # type: ignore
        ProteinInput,
        StructurePredictionInput,
    )

    target_chain = prepared_target.chains[0]
    binder_chain_id = _binder_chain_id(target_chain.canonical_chain_id)

    raw_input = StructurePredictionInput(
        sequences=[
            ProteinInput(
                id=target_chain.canonical_chain_id,
                sequence=target_chain.sequence,
                msa=None,
            ),
            ProteinInput(id=binder_chain_id, sequence=binder_sequence, msa=None),
        ],
        distogram_conditioning=None,
    )
    features = binder_design.prepare_esmfold2_tensors(raw_input)
    if target_distogram is not None:
        if target_distogram_mask is None:
            target_distogram_mask = np.ones_like(target_distogram, dtype=bool)
        _patch_single_chain_distogram_features(
            features,
            target_distogram=target_distogram,
            target_distogram_mask=target_distogram_mask,
        )
    return features


def _patch_single_chain_distogram_features(
    features: dict[str, Any],
    *,
    target_distogram: np.ndarray,
    target_distogram_mask: np.ndarray,
) -> None:
    target_length = int(target_distogram.shape[0])
    if target_distogram.shape != (target_length, target_length):
        raise ValueError(f"target_distogram must be square, got {target_distogram.shape}")
    if target_distogram_mask.shape != target_distogram.shape:
        raise ValueError(
            f"target_distogram_mask shape {target_distogram_mask.shape} does not "
            f"match distogram {target_distogram.shape}"
        )
    tokens = np.arange(target_length, dtype=np.int64)
    binned = _bin_distogram(target_distogram)
    _assign_matrix_block(features["disto_cond"], tokens, tokens, binned)
    _assign_matrix_block(
        features["disto_cond_mask"],
        tokens,
        tokens,
        target_distogram_mask.astype(bool, copy=False),
    )


def _bin_distogram(
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
        import torch  # type: ignore

        row_index = torch.as_tensor(rows, dtype=torch.long, device=matrix.device)
        col_index = torch.as_tensor(cols, dtype=torch.long, device=matrix.device)
        value_tensor = torch.as_tensor(values, dtype=matrix.dtype, device=matrix.device)
        matrix[row_index[:, None], col_index[None, :]] = value_tensor
        return
    matrix[np.ix_(rows, cols)] = values


def _batch_cuda_inputs(features: dict[str, Any]) -> dict[str, Any]:
    import torch  # type: ignore

    batched = {}
    for key, value in features.items():
        if torch.is_tensor(value):
            batched[key] = value.unsqueeze(0).cuda()
        else:
            batched[key] = value
    return batched


def _geometry_metrics_from_fold(
    inputs: dict[str, Any],
    output: dict[str, Any],
    *,
    target_sequence: str,
    experimental_coords: np.ndarray,
    experimental_coord_mask: np.ndarray | None = None,
) -> FoldGeometryMetrics:
    predicted_coords = _predicted_representative_coords(
        inputs,
        output,
        target_sequence=target_sequence,
    )
    if predicted_coords.shape != experimental_coords.shape:
        raise ValueError(
            f"predicted representative coords {predicted_coords.shape} do not match "
            f"experimental coords {experimental_coords.shape}"
        )
    if experimental_coord_mask is None:
        experimental_coord_mask = np.ones(experimental_coords.shape[0], dtype=bool)
    experimental_coord_mask = np.asarray(experimental_coord_mask, dtype=bool)
    if experimental_coord_mask.shape != (experimental_coords.shape[0],):
        raise ValueError(
            f"experimental_coord_mask shape {experimental_coord_mask.shape} does not "
            f"match target length {experimental_coords.shape[0]}"
        )
    predicted_coords = predicted_coords[experimental_coord_mask]
    experimental_coords = experimental_coords[experimental_coord_mask]
    if experimental_coords.shape[0] < 2:
        return FoldGeometryMetrics(
            target_distance_rmse=math.nan,
            target_aligned_rmsd=math.nan,
            target_residue_count=int(experimental_coords.shape[0]),
        )
    return FoldGeometryMetrics(
        target_distance_rmse=_distance_matrix_rmse(predicted_coords, experimental_coords),
        target_aligned_rmsd=_aligned_rmsd(predicted_coords, experimental_coords),
        target_residue_count=int(experimental_coords.shape[0]),
    )


def _predicted_representative_coords(
    inputs: dict[str, Any],
    output: dict[str, Any],
    *,
    target_sequence: str,
) -> np.ndarray:
    coords = _to_numpy(output["sample_atom_coords"])[0]
    atom_to_token = _to_numpy(inputs["atom_to_token"])[0].astype(int)
    ref_atom_name_chars = _to_numpy(inputs["ref_atom_name_chars"])[0]
    atom_attention_mask = _to_numpy(inputs["atom_attention_mask"])[0].astype(bool)
    target_length = len(target_sequence)

    token_atoms: list[dict[str, np.ndarray]] = [dict() for _ in range(target_length)]
    for atom_index, token_index in enumerate(atom_to_token):
        if token_index < 0 or token_index >= target_length:
            continue
        if not atom_attention_mask[atom_index]:
            continue
        atom_name = _decode_atom_name(ref_atom_name_chars[atom_index])
        token_atoms[token_index][atom_name] = coords[atom_index].astype(np.float64)

    representatives: list[np.ndarray] = []
    for index, aa in enumerate(target_sequence):
        atoms = token_atoms[index]
        atom_name = "CA" if aa == "G" or "CB" not in atoms else "CB"
        if atom_name not in atoms:
            raise ValueError(
                f"predicted target residue {index} lacks representative atom {atom_name}"
            )
        representatives.append(atoms[atom_name])
    return np.stack(representatives, axis=0)


def _hotspot_contact_metrics_from_fold(
    inputs: dict[str, Any],
    output: dict[str, Any],
    *,
    target_sequence: str,
    binder_sequence: str,
    hotspot_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    binder_indices: tuple[int, ...] | None,
) -> HotspotContactMetrics:
    if not hotspot_indices:
        raise ValueError("hotspot_indices cannot be empty")
    if contact_cutoff_angstrom <= 0:
        raise ValueError("contact_cutoff_angstrom must be positive")

    target_length = len(target_sequence)
    binder_length = len(binder_sequence)
    total_length = target_length + binder_length
    if binder_indices is None:
        binder_indices = tuple(range(binder_length))
    if not binder_indices:
        raise ValueError("binder_indices cannot be empty")
    for index in hotspot_indices:
        if index < 0 or index >= target_length:
            raise ValueError(
                f"hotspot index {index} is outside target length {target_length}"
            )
    for index in binder_indices:
        if index < 0 or index >= binder_length:
            raise ValueError(
                f"binder index {index} is outside binder length {binder_length}"
            )

    heavy_atom_coords_by_token = _predicted_heavy_atom_coords_by_token(
        inputs,
        output,
        sequence=target_sequence + binder_sequence,
    )
    if len(heavy_atom_coords_by_token) != total_length:
        raise ValueError(
            f"predicted heavy-atom coords length {len(heavy_atom_coords_by_token)} "
            f"does not match target+binder length {total_length}"
        )
    binder_heavy_atom_coords = []
    for index in binder_indices:
        binder_heavy_atom_coords.extend(
            coords for coords in heavy_atom_coords_by_token[target_length + index]
        )
    if not binder_heavy_atom_coords:
        raise ValueError("binder sequence has no predicted heavy atoms")
    binder_heavy_atom_coords_array = np.stack(binder_heavy_atom_coords, axis=0)

    heavy_atom_min_distances = []
    for index in hotspot_indices:
        hotspot_heavy_atom_coords = heavy_atom_coords_by_token[index]
        if hotspot_heavy_atom_coords.size == 0:
            raise ValueError(f"hotspot target residue {index} has no predicted heavy atoms")
        distances = _pairwise_distances(
            hotspot_heavy_atom_coords,
            binder_heavy_atom_coords_array,
        )
        heavy_atom_min_distances.append(float(np.min(distances)))
    heavy_atom_min_distances_array = np.asarray(
        heavy_atom_min_distances,
        dtype=np.float64,
    )

    representative_coords = _predicted_representative_coords_by_token(
        inputs,
        output,
        sequence=target_sequence + binder_sequence,
    )
    if representative_coords.shape[0] != total_length:
        raise ValueError(
            f"predicted representative coords length {representative_coords.shape[0]} "
            f"does not match target+binder length {total_length}"
        )

    target_hotspot_coords = representative_coords[list(hotspot_indices)]
    binder_coords = representative_coords[
        [target_length + index for index in binder_indices]
    ]
    if binder_coords.size == 0:
        raise ValueError("binder sequence cannot be empty")

    distances = _pairwise_distances(target_hotspot_coords, binder_coords)
    representative_min_distances = distances.min(axis=1)
    return HotspotContactMetrics(
        hotspot_heavy_atom_contact_fraction=float(
            np.mean(heavy_atom_min_distances_array <= contact_cutoff_angstrom)
        ),
        hotspot_min_heavy_atom_distance_mean=float(
            np.mean(heavy_atom_min_distances_array)
        ),
        hotspot_min_heavy_atom_distance_min=float(
            np.min(heavy_atom_min_distances_array)
        ),
        hotspot_representative_contact_fraction=float(
            np.mean(representative_min_distances <= contact_cutoff_angstrom)
        ),
        hotspot_min_representative_distance_mean=float(
            np.mean(representative_min_distances)
        ),
        hotspot_min_representative_distance_min=float(
            np.min(representative_min_distances)
        ),
        hotspot_contact_cutoff_angstrom=float(contact_cutoff_angstrom),
        hotspot_count=len(hotspot_indices),
    )


def _predicted_heavy_atom_coords_by_token(
    inputs: dict[str, Any],
    output: dict[str, Any],
    *,
    sequence: str,
) -> list[np.ndarray]:
    coords = _to_numpy(output["sample_atom_coords"])[0]
    atom_to_token = _to_numpy(inputs["atom_to_token"])[0].astype(int)
    ref_atom_name_chars = _to_numpy(inputs["ref_atom_name_chars"])[0]
    atom_attention_mask = _to_numpy(inputs["atom_attention_mask"])[0].astype(bool)

    token_atoms: list[list[np.ndarray]] = [[] for _ in range(len(sequence))]
    for atom_index, token_index in enumerate(atom_to_token):
        if token_index < 0 or token_index >= len(sequence):
            continue
        if not atom_attention_mask[atom_index]:
            continue
        atom_name = _decode_atom_name(ref_atom_name_chars[atom_index]).upper()
        if atom_name.startswith("H"):
            continue
        token_atoms[token_index].append(coords[atom_index].astype(np.float64))

    return [
        np.stack(token_coords, axis=0)
        if token_coords
        else np.empty((0, 3), dtype=np.float64)
        for token_coords in token_atoms
    ]


def _predicted_representative_coords_by_token(
    inputs: dict[str, Any],
    output: dict[str, Any],
    *,
    sequence: str,
) -> np.ndarray:
    coords = _to_numpy(output["sample_atom_coords"])[0]
    atom_to_token = _to_numpy(inputs["atom_to_token"])[0].astype(int)
    ref_atom_name_chars = _to_numpy(inputs["ref_atom_name_chars"])[0]
    atom_attention_mask = _to_numpy(inputs["atom_attention_mask"])[0].astype(bool)

    token_atoms: list[dict[str, np.ndarray]] = [dict() for _ in range(len(sequence))]
    for atom_index, token_index in enumerate(atom_to_token):
        if token_index < 0 or token_index >= len(sequence):
            continue
        if not atom_attention_mask[atom_index]:
            continue
        atom_name = _decode_atom_name(ref_atom_name_chars[atom_index])
        token_atoms[token_index][atom_name] = coords[atom_index].astype(np.float64)

    representatives: list[np.ndarray] = []
    for index, aa in enumerate(sequence):
        atoms = token_atoms[index]
        atom_name = "CA" if aa == "G" or "CB" not in atoms else "CB"
        if atom_name not in atoms:
            raise ValueError(
                f"predicted residue {index} lacks representative atom {atom_name}"
            )
        representatives.append(atoms[atom_name])
    return np.stack(representatives, axis=0)


def _experimental_representative_coords(prepared_target: PreparedTarget) -> np.ndarray:
    coords = []
    for chain in prepared_target.chains:
        for residue in chain.residues:
            coords.append(residue.representative_coord or (0.0, 0.0, 0.0))
    return np.array(coords, dtype=np.float64)


def _experimental_representative_coord_mask(prepared_target: PreparedTarget) -> np.ndarray:
    return np.asarray(
        [
            residue.representative_coord is not None
            for chain in prepared_target.chains
            for residue in chain.residues
        ],
        dtype=bool,
    )


def _target_chain_spans(prepared_target: PreparedTarget) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start = 0
    for chain in prepared_target.chains:
        end = start + len(chain.residues)
        spans.append((chain.canonical_chain_id, start, end))
        start = end
    return spans


def _distance_matrix_rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = _distance_matrix(a) - _distance_matrix(b)
    return float(math.sqrt(float(np.mean(diff * diff))))


def _optional_distance_matrix_rmse(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return None
    return _distance_matrix_rmse(a, b)


def _distance_matrix_upper_rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"coordinate arrays do not match: {a.shape} != {b.shape}")
    if a.shape[0] < 2:
        raise ValueError("at least two coordinates are required")
    diff = _distance_matrix(a) - _distance_matrix(b)
    mask = np.triu(np.ones(diff.shape, dtype=bool), k=1)
    return float(math.sqrt(float(np.mean(diff[mask] * diff[mask]))))


def _distance_matrix_delta_rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(math.sqrt(float(np.mean(diff * diff))))


def _optional_distance_matrix_delta_rmse(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size == 0 or b.size == 0:
        return None
    return _distance_matrix_delta_rmse(a, b)


def _optional_aligned_rmsd(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return None
    return _aligned_rmsd(a, b)


def _distance_matrix(coords: np.ndarray) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def _contact_recovery(
    predicted_distances: np.ndarray,
    experimental_distances: np.ndarray,
    *,
    cutoff: float,
) -> float | None:
    experimental_contacts = experimental_distances <= cutoff
    contact_count = int(experimental_contacts.sum())
    if contact_count == 0:
        return None
    predicted_contacts = predicted_distances <= cutoff
    recovered = predicted_contacts & experimental_contacts
    return float(recovered.sum() / contact_count)


def _aligned_rmsd(moving: np.ndarray, reference: np.ndarray) -> float:
    moving_centered = moving - moving.mean(axis=0)
    reference_centered = reference - reference.mean(axis=0)
    covariance = moving_centered.T @ reference_centered
    u, _s, vt = np.linalg.svd(covariance)
    determinant = np.linalg.det(u @ vt)
    correction = np.diag([1.0, 1.0, np.sign(determinant)])
    rotation = u @ correction @ vt
    aligned = moving_centered @ rotation
    diff = aligned - reference_centered
    return float(math.sqrt(float(np.mean(np.sum(diff * diff, axis=1)))))


def _decode_atom_name(chars: np.ndarray) -> str:
    return "".join(chr(int(char) + 32) for char in chars if int(char) != 0)


def _resolve_output_dir(config: CampaignConfig, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser()
    return config.output / "validation" / "conditioning"


def _settings_json(settings: ConditioningValidationSettings) -> dict[str, Any]:
    return {
        "config_path": str(settings.config_path),
        "output_dir": str(settings.output_dir),
        "esm_repo": str(settings.esm_repo) if settings.esm_repo is not None else None,
        "gpu_id": settings.gpu_id,
        "critic_name": settings.critic_name,
        "binder_sequence": settings.binder_sequence,
        "binder_length": len(settings.binder_sequence),
        "num_sampling_steps": settings.num_sampling_steps,
        "num_loops": settings.num_loops,
        "seed": settings.seed,
        "calculate_confidence": settings.calculate_confidence,
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _validate_protein_sequence(sequence: str, field_name: str) -> None:
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    if not sequence:
        raise ValueError(f"{field_name} cannot be empty")
    invalid = sorted(set(sequence) - allowed)
    if invalid:
        raise ValueError(f"{field_name} contains invalid amino acids: {''.join(invalid)}")


def _binder_chain_id(target_chain_id: str) -> str:
    for candidate in ("B", "Z", "binder"):
        if candidate != target_chain_id:
            return candidate
    raise ValueError(f"could not choose binder chain id distinct from {target_chain_id}")


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
