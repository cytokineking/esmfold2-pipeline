from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from esmfold2_pipeline.artifacts import write_text_atomic
from esmfold2_pipeline.config import CampaignConfig, load_campaign_config
from esmfold2_pipeline.db import CampaignStore, connect_database, initialize_database
from esmfold2_pipeline.planning.ids import shard_id
from esmfold2_pipeline.structure import (
    parse_structure_target,
    resolve_target_geometry_drift_indices,
    write_target_artifacts,
)


@dataclass(frozen=True)
class PlanResult:
    campaign_dir: Path
    shard_count: int
    config: CampaignConfig


def plan_campaign(config_path: str | Path, *, output_override: str | Path | None = None) -> PlanResult:
    source_path = Path(config_path)
    config_bytes = source_path.read_bytes()
    raw_config = yaml.safe_load(config_bytes.decode()) or {}
    config = load_campaign_config(source_path, output_override=output_override)
    campaign_dir = config.output
    resolved_config = _resolved_config_for_planning(
        config,
        raw_config,
        source_path.parent,
    )
    config_hash = _semantic_config_hash(resolved_config)
    existing_config_hashes = _existing_campaign_config_hashes(campaign_dir)
    if (
        existing_config_hashes is not None
        and config_hash not in existing_config_hashes
    ):
        raise ValueError(
            "existing campaign was planned from a different config: "
            f"{campaign_dir} (stored config_hash={existing_config_hashes[0]}, "
            f"incoming config_hash={config_hash})"
        )
    prepared_target = (
        parse_structure_target(config.target_structure)
        if config.target_structure is not None
        else None
    )
    if (
        prepared_target is not None
        and config.target_structure is not None
        and config.target_geometry_drift.enabled
    ):
        resolve_target_geometry_drift_indices(
            prepared_target,
            config.target_geometry_drift.regions,
            structure_indexing=config.target_structure.structure_indexing,
            field_name="loss.target_geometry_drift.regions",
        )

    campaign_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(campaign_dir / "config.yaml", config_bytes.decode())
    write_text_atomic(
        campaign_dir / "resolved_config.yaml",
        yaml.safe_dump(resolved_config, sort_keys=True),
    )
    if prepared_target is not None and config.target_structure is not None:
        write_target_artifacts(
            prepared_target,
            campaign_dir / "target",
            conditioning_mode=config.target_structure.conditioning_mode,
            conditioning_assembly=config.target_structure.conditioning_assembly,
            conditioning_chain_pairs=config.target_structure.conditioning_chain_pairs,
            partial_conditioning=config.target_structure.partial_conditioning,
            representative_atom=config.target_structure.representative_atom,
            require_resolved=config.target_structure.require_resolved,
        )

    conn = initialize_database(
        campaign_dir / "campaign.sqlite",
        config_hash=config_hash,
        resolved_config=resolved_config,
        software_versions={"esmfold2_pipeline": "0.1.0"},
    )
    store = CampaignStore(conn)
    for batch_index, seed in enumerate(config.seeds):
        store.create_shard(
            shard_id=shard_id(batch_index),
            seed=seed,
            batch_index=batch_index,
            target_key=_target_key(config),
            binder_key=config.binder_key_for_design_index(batch_index),
            critic_set=[config.critic_name],
        )
    conn.close()
    return PlanResult(campaign_dir=campaign_dir, shard_count=len(config.seeds), config=config)


def _resolved_config_for_planning(
    config: CampaignConfig,
    raw_config: Any,
    source_dir: Path,
) -> dict[str, Any]:
    resolved_config = config.to_resolved_dict()
    if isinstance(raw_config, dict) and isinstance(raw_config.get("validation"), dict):
        resolved_config["validation"] = _normalize_validation_paths(
            raw_config["validation"],
            source_dir,
        )
    return resolved_config


def _semantic_config_hash(resolved_config: dict[str, Any]) -> str:
    payload = json.dumps(
        resolved_config,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _existing_campaign_config_hashes(campaign_dir: Path) -> tuple[str, ...] | None:
    db_path = campaign_dir / "campaign.sqlite"
    if not db_path.exists():
        return None
    conn = connect_database(db_path)
    try:
        row = conn.execute("SELECT * FROM campaign WHERE id = 1").fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    stored_hash = str(row["config_hash"])
    hashes = [stored_hash]
    if "resolved_config_json" in row.keys():
        try:
            resolved_config = json.loads(row["resolved_config_json"] or "{}")
        except json.JSONDecodeError:
            resolved_config = None
        if isinstance(resolved_config, dict):
            hashes.append(_semantic_config_hash(resolved_config))
    return tuple(dict.fromkeys(hashes))


_VALIDATION_PATH_KEYS = {
    "checkpoint_dir",
    "msa_cache_root",
    "protenix_checkpoint_dir",
    "protenix_root",
    "root",
    "scratch_root",
    "target_msa_dir",
    "target_msa_map_csv",
}


def _normalize_validation_paths(value: Any, base_dir: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_validation_path_value(key, item, base_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_validation_paths(item, base_dir) for item in value]
    return value


def _normalize_validation_path_value(key: str, value: Any, base_dir: Path) -> Any:
    if isinstance(value, dict):
        return _normalize_validation_paths(value, base_dir)
    if isinstance(value, list):
        return [_normalize_validation_paths(item, base_dir) for item in value]
    if key not in _VALIDATION_PATH_KEYS or value in (None, ""):
        return value
    path = Path(str(value)).expanduser()
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _target_key(config: CampaignConfig) -> str:
    if config.target_structure is None:
        if config.target_sequence is not None:
            digest = hashlib.sha256(config.target_sequence.encode()).hexdigest()[:16]
            return f"target:sequence:{digest}:{config.target_name}"
        return f"target:name:{config.target_name}"
    return f"target:structure:{config.target_structure.path}:{config.target_name}"
