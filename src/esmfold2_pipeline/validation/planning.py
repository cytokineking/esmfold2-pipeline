from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from esmfold2_pipeline.db import CampaignStore, initialize_database
from esmfold2_pipeline.frameworks import (
    get_scfv_framework_template_cif,
    get_vhh_framework_template_cif,
)
from esmfold2_pipeline.reports.exports import (
    _completed_metric_rows,
    _deduplicate_by_sequence,
    _row_has_hotspot_contact,
    _should_require_hotspot_contact,
    _sort_key,
)
from esmfold2_pipeline.validation.msa import (
    DEFAULT_MSA_PAIRING_STRATEGY,
    BinderMsaMode,
    MsaMode,
    MsaPairingStrategy,
    ProtenixMsaConfig,
    msa_context_hash,
)

DEFAULT_VALIDATE_MODEL = "protenix-v2"
DEFAULT_VALIDATION_SEEDS = (101,)
DEFAULT_VALIDATION_N_SAMPLE = 1
DEFAULT_VALIDATION_N_STEP = 200
DEFAULT_VALIDATION_N_CYCLE = 10
DEFAULT_VALIDATION_TOKEN_LIMIT = 2560
DEFAULT_IPSAE_PAE_CUTOFF = 15.0
DEFAULT_IPSAE_DIST_CUTOFF = 15.0
VALIDATION_CONFIG_SCHEMA = "protenix_validation_v2"
UNSUPPORTED_PROTENIX_SCAFFOLDS = frozenset({"scfv", "sc_fv"})
TemplateMode = Literal["auto", "true", "false"]
TEMPLATE_MODES = frozenset({"auto", "true", "false"})


@dataclass(frozen=True)
class ValidationPlanConfig:
    model_name: str = DEFAULT_VALIDATE_MODEL
    top_k: int | None = None
    min_esm_iptm: float | None = None
    min_validation_iptm: float | None = None
    min_validation_ipsae: float | None = None
    require_hotspot_contact: Literal["auto", "always", "never"] = "auto"
    validation_hotspot_cutoff_angstrom: float | None = None
    protenix_command: tuple[str, ...] | None = None
    protenix_python: str | None = None
    protenix_root: Path | None = None
    checkpoint_dir: Path | None = None
    seeds: tuple[int, ...] = DEFAULT_VALIDATION_SEEDS
    n_sample: int = DEFAULT_VALIDATION_N_SAMPLE
    n_step: int = DEFAULT_VALIDATION_N_STEP
    n_cycle: int = DEFAULT_VALIDATION_N_CYCLE
    token_limit: int | None = DEFAULT_VALIDATION_TOKEN_LIMIT
    use_msa: bool = False
    use_template: TemplateMode = "auto"
    target_msa_mode: MsaMode = "none"
    binder_msa_mode: BinderMsaMode = "auto"
    target_msa_dir: Path | None = None
    target_msa_map_csv: Path | None = None
    msa_server_url: str | None = None
    msa_cache_root: Path | None = None
    msa_pairing_strategy: MsaPairingStrategy = DEFAULT_MSA_PAIRING_STRATEGY
    ipsae_script_path: Path | None = None
    ipsae_python: str | None = None
    ipsae_pae_cutoff: float = DEFAULT_IPSAE_PAE_CUTOFF
    ipsae_dist_cutoff: float = DEFAULT_IPSAE_DIST_CUTOFF
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None for all")
        if self.min_validation_ipsae is not None and self.min_validation_ipsae < 0:
            raise ValueError("min_validation_ipsae must be non-negative")
        if (
            self.validation_hotspot_cutoff_angstrom is not None
            and self.validation_hotspot_cutoff_angstrom <= 0
        ):
            raise ValueError("validation_hotspot_cutoff_angstrom must be positive")
        if self.protenix_command is not None and not self.protenix_command:
            raise ValueError("protenix_command cannot be empty")
        if self.protenix_command is not None and self.protenix_python is not None:
            raise ValueError("use protenix_command or protenix_python, not both")
        if not self.seeds:
            raise ValueError("at least one Protenix seed is required")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Protenix seeds must be non-negative")
        if self.n_sample <= 0:
            raise ValueError("n_sample must be positive")
        if self.n_step <= 0:
            raise ValueError("n_step must be positive")
        if self.n_cycle <= 0:
            raise ValueError("n_cycle must be positive")
        if self.token_limit is not None and self.token_limit <= 0:
            raise ValueError("token_limit must be positive when provided")
        if self.use_template not in TEMPLATE_MODES:
            raise ValueError("use_template must be one of: auto, true, false")
        if self.ipsae_pae_cutoff <= 0:
            raise ValueError("ipsae_pae_cutoff must be positive")
        if self.ipsae_dist_cutoff <= 0:
            raise ValueError("ipsae_dist_cutoff must be positive")
        self.msa_config()
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.require_hotspot_contact not in {"auto", "always", "never"}:
            raise ValueError(
                "require_hotspot_contact must be one of: auto, always, never"
            )

    def msa_config(self) -> ProtenixMsaConfig:
        return ProtenixMsaConfig(
            target_mode=self.target_msa_mode,
            binder_mode=self.binder_msa_mode,
            target_msa_dir=self.target_msa_dir,
            target_msa_map_csv=self.target_msa_map_csv,
            server_url=self.msa_server_url,
            cache_root=self.msa_cache_root,
            pairing_strategy=self.msa_pairing_strategy,
        )

    @property
    def validation_config_hash(self) -> str:
        payload = self.validation_config_payload()
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def validation_config_payload(self) -> dict[str, Any]:
        msa_config = self.msa_config()
        runtime_command = (
            self.protenix_command
            if self.protenix_command is not None
            else (
                (self.protenix_python, "-m", "runner.inference")
                if self.protenix_python is not None
                else None
            )
        )
        return {
            "schema": VALIDATION_CONFIG_SCHEMA,
            "scope": "binder_target",
            "model": {
                "name": self.model_name,
                "command": list(runtime_command) if runtime_command else None,
                "protenix_root": _path_context(self.protenix_root),
                "checkpoint_dir": _path_context(self.checkpoint_dir),
            },
            "runtime": {
                "seeds": list(self.seeds),
                "n_sample": self.n_sample,
                "n_step": self.n_step,
                "n_cycle": self.n_cycle,
                "token_limit": self.token_limit,
            },
            "msa": {
                "context_hash": msa_context_hash(msa_config),
                "use_msa": self.use_msa,
                "target_mode": self.target_msa_mode,
                "binder_mode": self.binder_msa_mode,
                "server_url": _normalized_server_url(self.msa_server_url),
                "pairing_strategy": self.msa_pairing_strategy,
                "target_msa_dir": _path_context(self.target_msa_dir),
                "target_msa_map_csv": _path_context(self.target_msa_map_csv),
                "cache_root": _path_context(self.msa_cache_root),
            },
            "template": {
                "use_template": self.use_template,
            },
            "filters": {
                "min_esm_iptm": self.min_esm_iptm,
                "min_validation_iptm": self.min_validation_iptm,
                "min_validation_ipsae": self.min_validation_ipsae,
                "require_hotspot_contact": self.require_hotspot_contact,
                "validation_hotspot_cutoff_angstrom": (
                    self.validation_hotspot_cutoff_angstrom
                ),
            },
            "ipsae": {
                "script_path": _path_context(self.ipsae_script_path),
                "python": self.ipsae_python,
                "pae_cutoff": self.ipsae_pae_cutoff,
                "dist_cutoff": self.ipsae_dist_cutoff,
            },
        }


@dataclass(frozen=True)
class ValidationPlanResult:
    campaign_dir: Path
    model_name: str
    validation_config_hash: str
    candidate_count: int
    selected_count: int
    created_count: int
    existing_count: int


def plan_validation_tasks(
    campaign_dir: str | Path,
    *,
    config: ValidationPlanConfig | None = None,
) -> ValidationPlanResult:
    root = Path(campaign_dir)
    db_path = root / "campaign.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"missing campaign database: {db_path}")
    if config is None:
        config = ValidationPlanConfig()

    rows = _validation_candidate_rows(root, config=config)
    selected = rows if config.top_k is None else rows[: config.top_k]
    _raise_for_unsupported_validation_scaffolds(selected, config=config)

    conn = initialize_database(db_path)
    store = CampaignStore(conn)
    try:
        created = 0
        existing = 0
        for rank, row in enumerate(selected, start=1):
            validation_id = validation_task_id(
                candidate_id=str(row["candidate_id"]),
                model_name=config.model_name,
                validation_config_hash=config.validation_config_hash,
            )
            was_created = store.create_validation_task(
                validation_id=validation_id,
                candidate_id=str(row["candidate_id"]),
                model_name=config.model_name,
                validation_config_hash=config.validation_config_hash,
                selection_rank=rank,
                max_attempts=config.max_attempts,
            )
            if was_created:
                created += 1
            else:
                existing += 1
    finally:
        conn.close()

    return ValidationPlanResult(
        campaign_dir=root,
        model_name=config.model_name,
        validation_config_hash=config.validation_config_hash,
        candidate_count=len(rows),
        selected_count=len(selected),
        created_count=created,
        existing_count=existing,
    )


def validation_task_id(
    *,
    candidate_id: str,
    model_name: str,
    validation_config_hash: str,
) -> str:
    readable = _safe_identifier(candidate_id, max_len=80)
    digest = hashlib.sha256(
        f"{candidate_id}|{model_name}|{validation_config_hash}".encode("utf-8")
    ).hexdigest()[:10]
    return f"val_{readable}_{digest}"


def _raise_for_unsupported_validation_scaffolds(
    rows: list[dict[str, Any]],
    *,
    config: ValidationPlanConfig,
) -> None:
    if not _is_protenix_model(config.model_name):
        return
    unsupported = [
        row
        for row in rows
        if _scaffold_key(row) in UNSUPPORTED_PROTENIX_SCAFFOLDS
        and (
            not _row_has_supported_framework_template(row, config=config)
            or _requests_unsupported_scfv_binder_msa(config)
        )
    ]
    if not unsupported:
        return
    candidate_ids = ", ".join(
        str(row["candidate_id"])
        for row in unsupported[:5]
        if row.get("candidate_id") is not None
    )
    suffix = (
        f" Selected incompatible candidates include: {candidate_ids}."
        if candidate_ids
        else ""
    )
    raise ValueError(
        "scFv Protenix validation requires a bundled scFv framework "
        "structural template, and scFv binder MSA support is not implemented. "
        "Use a bundled scFv framework with template-only binder validation, "
        "or set validation.msa.binder: none when adding target MSAs."
        f"{suffix}"
    )


def _row_has_supported_framework_template(
    row: dict[str, Any],
    *,
    config: ValidationPlanConfig,
) -> bool:
    if config.use_template == "false":
        return False
    if str(row.get("framework_source") or "").strip().lower() != "builtin":
        return False
    framework = str(row.get("framework") or "").strip()
    if not framework:
        return False
    try:
        scaffold = _scaffold_key(row)
        if scaffold in {"scfv", "sc_fv"}:
            return get_scfv_framework_template_cif(framework) is not None
        if scaffold == "vhh":
            return get_vhh_framework_template_cif(framework) is not None
    except KeyError:
        return False
    return False


def _requests_unsupported_scfv_binder_msa(config: ValidationPlanConfig) -> bool:
    if config.binder_msa_mode == "none":
        return False
    return bool(config.use_msa or config.binder_msa_mode == "single_sequence")


def _is_protenix_model(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return normalized == DEFAULT_VALIDATE_MODEL or normalized.startswith("protenix-")


def _scaffold_key(row: dict[str, Any]) -> str:
    scaffold = row.get("binder_scaffold")
    if scaffold in (None, ""):
        scaffold = row.get("binder_type")
    return str(scaffold or "").strip().lower()


def _validation_candidate_rows(
    root: Path,
    *,
    config: ValidationPlanConfig,
) -> list[dict[str, Any]]:
    rows = _completed_metric_rows(root)
    if config.min_esm_iptm is not None:
        rows = [
            row
            for row in rows
            if row["iptm"] is not None and float(row["iptm"]) >= config.min_esm_iptm
        ]

    if _should_require_hotspot_contact(
        root,
        rows,
        config.require_hotspot_contact,
    ):
        rows = [row for row in rows if _row_has_hotspot_contact(row)]

    deduped = _deduplicate_by_sequence(rows)
    return sorted(deduped, key=_sort_key)


def _safe_identifier(value: str, *, max_len: int) -> str:
    chars = [
        char.lower() if char.isalnum() else "_"
        for char in value.strip()
    ]
    text = "".join(chars).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    if not text:
        text = "candidate"
    return text[:max_len].strip("_") or "candidate"


def _normalized_server_url(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().rstrip("/")
    return text or None


def _path_context(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    expanded = Path(path).expanduser()
    payload: dict[str, Any] = {"path": str(expanded)}
    try:
        stat = expanded.stat()
    except OSError:
        payload["exists"] = False
        return payload

    payload["exists"] = True
    payload["kind"] = "dir" if expanded.is_dir() else "file"
    payload["mtime_ns"] = int(stat.st_mtime_ns)
    payload["size"] = int(stat.st_size)
    if expanded.is_dir():
        try:
            entries = []
            for child in sorted(expanded.iterdir(), key=lambda item: item.name):
                try:
                    child_stat = child.stat()
                except OSError:
                    continue
                entries.append(
                    {
                        "name": child.name,
                        "kind": "dir" if child.is_dir() else "file",
                        "mtime_ns": int(child_stat.st_mtime_ns),
                        "size": int(child_stat.st_size),
                    }
                )
            payload["entries"] = entries
        except OSError:
            pass
    return payload
