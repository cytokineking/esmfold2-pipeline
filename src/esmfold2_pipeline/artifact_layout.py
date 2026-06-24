from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ESMFOLD2_DIR = "esmfold2"
ESMFOLD2_STRUCTURES_DIR = Path(ESMFOLD2_DIR) / "structures"
ESMFOLD2_SELECTED_STRUCTURES_DIR = Path(ESMFOLD2_DIR) / "selected_structures"
ESMFOLD2_METRICS_CSV = "metrics_all.csv"
ESMFOLD2_SELECTED_DESIGNS_CSV = "selected_designs.csv"
ESMFOLD2_SUMMARY_JSON = "campaign_summary.json"
ESMFOLD2_SELECTED_MANIFEST_CSV = "selected_manifest.csv"

VALIDATION_DIR = "validation"
VALIDATION_STRUCTURES_DIR = "structures"
VALIDATION_STAGING_DIR = ".staging"
VALIDATION_PASSING_DIR = "passing"
VALIDATION_REJECTED_DIR = "rejected"
VALIDATION_MSA_CACHE_DIR = "msa_cache"
VALIDATION_RESULTS_CSV = "validation_results.csv"
VALIDATION_STRUCTURE_SAMPLES_CSV = "structure_samples.csv"
VALIDATION_SUMMARY_JSON = "validation_summary.json"

ANALYSIS_DIR = "analysis"
ANALYSIS_COMBINED_RANKING_CSV = "combined_ranking.csv"
ANALYSIS_RANKING_SUMMARY_JSON = "ranking_summary.json"
ANALYSIS_PLOTS_DIR = "plots"
ANALYSIS_TOP_RANKED_DIR = "top_ranked"


@dataclass(frozen=True)
class ArtifactNameSettings:
    shard_width: int = 3
    seed_width: int = 3
    candidate_width: int = 3


def artifact_name_settings(
    *,
    seeds: tuple[int, ...] | list[int],
    shard_count: int | None = None,
    candidate_count: int = 1,
) -> ArtifactNameSettings:
    if not seeds:
        raise ValueError("seeds cannot be empty")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    if shard_count is None:
        shard_count = len(seeds)
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    return ArtifactNameSettings(
        shard_width=max(3, len(str(shard_count - 1))),
        seed_width=max(3, max(len(str(seed)) for seed in seeds)),
        candidate_width=max(3, len(str(candidate_count - 1))),
    )


def artifact_stem(
    *,
    batch_index: int,
    seed: int,
    candidate_index: int,
    settings: ArtifactNameSettings | None = None,
) -> str:
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if candidate_index < 0:
        raise ValueError("candidate_index must be non-negative")
    if settings is None:
        settings = ArtifactNameSettings()
    return (
        f"s{batch_index:0{settings.shard_width}d}"
        f"_seed{seed:0{settings.seed_width}d}"
        f"_c{candidate_index:0{settings.candidate_width}d}"
    )


def structure_relpath(stem: str, *, suffix: str = ".pdb") -> Path:
    if not stem:
        raise ValueError("stem cannot be empty")
    if not suffix.startswith("."):
        raise ValueError("suffix must start with '.'")
    return ESMFOLD2_STRUCTURES_DIR / f"{stem}{suffix}"


def validator_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("model_name must contain at least one alphanumeric character")
    return slug


def validator_dir(model_name: str) -> Path:
    return Path(VALIDATION_DIR) / validator_slug(model_name)


def validator_structures_dir(model_name: str) -> Path:
    return validator_dir(model_name) / VALIDATION_STRUCTURES_DIR


def validator_structure_status_dir(model_name: str, status: str) -> Path:
    valid_status_dirs = {
        VALIDATION_PASSING_DIR,
        VALIDATION_REJECTED_DIR,
        VALIDATION_STAGING_DIR,
    }
    if status not in valid_status_dirs:
        raise ValueError(f"unsupported validator structure status directory: {status}")
    return validator_structures_dir(model_name) / status


def validator_msa_cache_dir(model_name: str) -> Path:
    return validator_dir(model_name) / VALIDATION_MSA_CACHE_DIR
