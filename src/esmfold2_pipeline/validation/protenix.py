from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Literal, Sequence
import uuid

from esmfold2_pipeline.artifact_layout import (
    VALIDATION_PASSING_DIR,
    VALIDATION_REJECTED_DIR,
    VALIDATION_STAGING_DIR,
    validator_dir,
    validator_slug,
    validator_structure_status_dir,
)
from esmfold2_pipeline.artifacts import write_bytes_atomic, write_json_atomic
from esmfold2_pipeline.db import CampaignStore, ValidationClaim, initialize_database
from esmfold2_pipeline.frameworks import (
    get_scfv_framework_template_cif,
    get_vhh_framework_template_cif,
)
from esmfold2_pipeline.validation.hotspots import (
    score_validation_hotspots,
    validation_hotspot_context,
)
from esmfold2_pipeline.validation.msa import (
    DEFAULT_MSA_PAIRING_STRATEGY,
    BinderMsaMode,
    MsaPair,
    MsaPairingStrategy,
    MsaMode,
    ProtenixMsaConfig,
    resolve_binder_msa_pairs,
    resolve_target_msa_pairs,
    write_msa_files_for_input,
)

DEFAULT_PROTENIX_SEEDS = (101,)
DEFAULT_PROTENIX_TOKEN_LIMIT = 2560
ProtenixTemplateMode = Literal["auto", "true", "false"]
PROTENIX_TEMPLATE_MODES = frozenset({"auto", "true", "false"})


@dataclass(frozen=True)
class ProtenixRunnerConfig:
    model_name: str = "protenix-v2"
    protenix_command: tuple[str, ...] | None = None
    checkpoint_dir: Path | None = None
    protenix_root: Path | None = None
    scratch_root: Path | None = None
    keep_debug: bool = False
    seeds: tuple[int, ...] = DEFAULT_PROTENIX_SEEDS
    n_sample: int = 1
    n_step: int = 200
    n_cycle: int = 10
    use_msa: bool = False
    use_template: ProtenixTemplateMode = "auto"
    target_msa_mode: MsaMode = "none"
    binder_msa_mode: BinderMsaMode = "auto"
    target_msa_dir: Path | None = None
    target_msa_map_csv: Path | None = None
    msa_server_url: str | None = None
    msa_cache_root: Path | None = None
    msa_pairing_strategy: MsaPairingStrategy = DEFAULT_MSA_PAIRING_STRATEGY
    msa_max_submit_retries: int = 6
    msa_max_status_polls: int = 120
    msa_status_poll_interval_seconds: float = 10.0
    msa_request_timeout_seconds: float = 20.0
    timeout_seconds: int = 7200
    heartbeat_interval_seconds: float = 30.0
    batch_size: int = 1
    max_tasks: int | None = None
    min_validation_iptm: float | None = None
    min_validation_ipsae: float | None = None
    ipsae_script_path: Path | None = None
    ipsae_python: str | None = None
    ipsae_pae_cutoff: float = 15.0
    ipsae_dist_cutoff: float = 15.0
    validation_hotspot_cutoff_angstrom: float | None = None
    token_limit: int | None = DEFAULT_PROTENIX_TOKEN_LIMIT
    env: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if self.protenix_command is not None and not self.protenix_command:
            raise ValueError("protenix_command cannot be empty")
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
        if self.use_template not in PROTENIX_TEMPLATE_MODES:
            raise ValueError("use_template must be one of: auto, true, false")
        self.msa_config()
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_tasks is not None and self.max_tasks < 0:
            raise ValueError("max_tasks must be non-negative when provided")
        if self.min_validation_ipsae is not None and self.min_validation_ipsae < 0:
            raise ValueError("min_validation_ipsae must be non-negative")
        if self.ipsae_pae_cutoff <= 0:
            raise ValueError("ipsae_pae_cutoff must be positive")
        if self.ipsae_dist_cutoff <= 0:
            raise ValueError("ipsae_dist_cutoff must be positive")
        if (
            self.validation_hotspot_cutoff_angstrom is not None
            and self.validation_hotspot_cutoff_angstrom <= 0
        ):
            raise ValueError("validation_hotspot_cutoff_angstrom must be positive")
        if self.token_limit is not None and self.token_limit <= 0:
            raise ValueError("token_limit must be positive when provided")

    def msa_config(self) -> ProtenixMsaConfig:
        return ProtenixMsaConfig(
            target_mode=self.target_msa_mode,
            binder_mode=self.binder_msa_mode,
            target_msa_dir=self.target_msa_dir,
            target_msa_map_csv=self.target_msa_map_csv,
            server_url=self.msa_server_url,
            cache_root=self.msa_cache_root,
            pairing_strategy=self.msa_pairing_strategy,
            max_submit_retries=self.msa_max_submit_retries,
            max_status_polls=self.msa_max_status_polls,
            status_poll_interval_seconds=self.msa_status_poll_interval_seconds,
            request_timeout_seconds=self.msa_request_timeout_seconds,
        )


@dataclass(frozen=True)
class ProtenixValidationRunResult:
    completed_tasks: int
    recorded_structures: int
    failed_tasks: int
    skipped_tasks: int
    retryable_failed_attempts: int
    skipped_no_pending: bool


@dataclass(frozen=True)
class ProtenixStructureResult:
    validation_id: str
    candidate_id: str
    structure_id: str
    seed: int
    sample_rank: int
    status: str
    cif_path: Path
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ProtenixCommandResult:
    returncode: int
    stdout_tail: str
    stderr_tail: str


@dataclass(frozen=True)
class ProtenixValidationResult:
    validation_id: str
    candidate_id: str
    structure_results: tuple[ProtenixStructureResult, ...]
    best_structure_id: str
    best_structure_path: str
    metrics: dict[str, Any]
    runtime_seconds: float


@dataclass(frozen=True)
class ProtenixTaskInput:
    validation_id: str
    candidate_id: str
    model_name: str
    selection_rank: int | None
    designed_sequence: str
    target_sequences: tuple[str, ...]
    target_labels: tuple[str, ...]
    seed: int
    binder_scaffold: str | None
    framework: str | None
    framework_source: str | None = None
    validation_config_hash: str = ""
    needs_config_suffix: bool = False

    @property
    def sample_name(self) -> str:
        return f"{_safe_identifier(self.validation_id, max_len=110)}_pred"

    @property
    def chain_role_map(self) -> dict[str, list[str]]:
        target_chains = [
            _chain_id_for_index(index)
            for index in range(1, 1 + len(self.target_sequences))
        ]
        return {"binder": ["A"], "target": target_chains}

    @property
    def token_count(self) -> int:
        return len(self.designed_sequence) + sum(len(seq) for seq in self.target_sequences)


@dataclass(frozen=True)
class ProtenixTemplateSpec:
    path: Path
    chain_ids: tuple[str, ...]
    template_ids: tuple[str, ...]
    source: str


def run_local_protenix_validation(
    campaign_dir: str | Path,
    *,
    worker_id: str = "protenix-validation-worker-0",
    gpu_id: str | None = None,
    config: ProtenixRunnerConfig | None = None,
) -> ProtenixValidationRunResult:
    """Claim pending validation tasks and run a local Protenix subprocess batch."""

    if config is None:
        config = ProtenixRunnerConfig()

    root = Path(campaign_dir)
    conn = initialize_database(root / "campaign.sqlite")
    store = CampaignStore(conn)
    completed = 0
    failed = 0
    skipped = 0
    retryable_failed_attempts = 0
    attempts_processed = 0
    recorded_structures = 0

    def record_failed_attempt(
        *,
        validation_id: str,
        attempt_id: int,
        error_message: str,
    ) -> None:
        nonlocal failed, retryable_failed_attempts, attempts_processed
        next_status = store.fail_validation_task(
            validation_id=validation_id,
            attempt_id=attempt_id,
            error_message=error_message,
        )
        attempts_processed += 1
        if next_status == "failed":
            failed += 1
        else:
            retryable_failed_attempts += 1

    try:
        while config.max_tasks is None or attempts_processed < config.max_tasks:
            remaining = (
                None
                if config.max_tasks is None
                else config.max_tasks - attempts_processed
            )
            batch_size = config.batch_size if remaining is None else min(config.batch_size, remaining)
            if batch_size <= 0:
                break
            store.reopen_ready_msa_jobs_with_missing_cache(base_dir=root)
            claims = store.claim_next_pending_validation_tasks(
                worker_id=worker_id,
                batch_size=batch_size,
                hostname=socket.gethostname(),
                pid=os.getpid(),
                gpu_id=gpu_id,
            )
            if not claims:
                break

            try:
                tasks = _fetch_task_inputs(conn, root=root, claims=claims)
            except Exception as exc:
                for claim in claims:
                    record_failed_attempt(
                        validation_id=claim.validation_id,
                        attempt_id=claim.attempt_id,
                        error_message=str(exc),
                    )
                continue

            runnable: list[tuple[ValidationClaim, ProtenixTaskInput]] = []
            for claim, task in zip(claims, tasks):
                skip_reason = _skip_reason(task, config=config)
                if skip_reason is not None:
                    store.skip_validation_task(
                        validation_id=claim.validation_id,
                        error_message=skip_reason,
                        attempt_id=claim.attempt_id,
                    )
                    skipped += 1
                    attempts_processed += 1
                else:
                    runnable.append((claim, task))

            if not runnable:
                continue

            scratch_dir = _scratch_dir(root, worker_id=worker_id, config=config)
            scratch_dir.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            input_json: Path | None = None
            output_dir: Path | None = None
            command: list[str] | None = None
            command_result: ProtenixCommandResult | None = None
            try:
                task_list = [task for _claim, task in runnable]
                template_specs = _template_specs_by_task(
                    root,
                    task_list,
                    config=config,
                )
                target_msas = _target_msas_by_task(
                    root,
                    task_list,
                    config=config,
                    template_specs=template_specs,
                )
                binder_msas = _binder_msas_by_task(
                    root,
                    task_list,
                    config=config,
                    template_specs=template_specs,
                    target_msas=target_msas,
                )
                input_json, sample_names, chain_maps = build_protenix_input_json(
                    task_list,
                    scratch_dir / "input",
                    target_msas=target_msas,
                    binder_msas=binder_msas,
                    template_specs=template_specs,
                )
                output_dir = scratch_dir / "output"
                output_dir.mkdir(parents=True, exist_ok=True)
                command = build_protenix_command(
                    input_json=input_json,
                    output_dir=output_dir,
                    config=config,
                    use_msa=bool(config.use_msa or target_msas or binder_msas),
                    use_template=bool(template_specs),
                )
                _heartbeat_validation_claims(store, runnable)
                command_result = _run_protenix_command(
                    command,
                    cwd=config.protenix_root,
                    env=config.env,
                    gpu_id=gpu_id,
                    timeout_seconds=config.timeout_seconds,
                    heartbeat_interval_seconds=config.heartbeat_interval_seconds,
                    on_heartbeat=lambda: _heartbeat_validation_claims(store, runnable),
                )
                runtime_seconds = time.monotonic() - start
                _write_validation_debug_metadata(
                    root=root,
                    scratch_dir=scratch_dir,
                    input_json=input_json,
                    output_dir=output_dir,
                    command=command,
                    command_result=command_result,
                    runnable=runnable,
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    config=config,
                )

                for claim, task in runnable:
                    try:
                        structures = collect_protenix_structures(
                            output_dir=output_dir,
                            task=task,
                            sample_name=sample_names[task.validation_id],
                            chain_role_map=chain_maps[task.validation_id],
                            config=config,
                            campaign_dir=root,
                        )
                        if not structures:
                            raise RuntimeError(
                                f"no Protenix CIF/summary outputs found for {task.validation_id}"
                            )
                        best = _best_structure(structures)
                        best_path: str | None = None
                        for structure in structures:
                            relpath = _publish_structure(
                                root=root,
                                store=store,
                                task=task,
                                structure=structure,
                            )
                            recorded_structures += 1
                            if structure.structure_id == best.structure_id:
                                best_path = relpath
                        if best_path is None:
                            raise RuntimeError(
                                f"best Protenix structure was not published for {task.validation_id}"
                            )
                        store.complete_validation_task(
                            validation_id=claim.validation_id,
                            attempt_id=claim.attempt_id,
                            output_structure_path=best_path,
                            metrics={
                                **best.metrics,
                                "best_structure_id": best.structure_id,
                                "structure_count": len(structures),
                            },
                            runtime_seconds=runtime_seconds,
                        )
                        completed += 1
                        attempts_processed += 1
                    except Exception as exc:
                        record_failed_attempt(
                            validation_id=claim.validation_id,
                            attempt_id=claim.attempt_id,
                            error_message=str(exc),
                        )
            except Exception as exc:
                _write_validation_debug_metadata(
                    root=root,
                    scratch_dir=scratch_dir,
                    input_json=input_json,
                    output_dir=output_dir,
                    command=command,
                    command_result=command_result,
                    runnable=runnable,
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    config=config,
                    error_message=str(exc),
                )
                for claim, _task in runnable:
                    record_failed_attempt(
                        validation_id=claim.validation_id,
                        attempt_id=claim.attempt_id,
                        error_message=str(exc),
                    )
            finally:
                if not config.keep_debug:
                    _cleanup_scratch_dir(scratch_dir, root=root, config=config)
    finally:
        conn.close()

    return ProtenixValidationRunResult(
        completed_tasks=completed,
        recorded_structures=recorded_structures,
        failed_tasks=failed,
        skipped_tasks=skipped,
        retryable_failed_attempts=retryable_failed_attempts,
        skipped_no_pending=attempts_processed == 0,
    )


def build_protenix_input_json(
    tasks: Sequence[ProtenixTaskInput],
    input_dir: str | Path,
    *,
    target_msas: dict[str, tuple[MsaPair | None, ...]] | None = None,
    binder_msas: dict[str, MsaPair | None] | None = None,
    template_specs: dict[str, tuple[ProtenixTemplateSpec, ...]] | None = None,
) -> tuple[Path, dict[str, str], dict[str, dict[str, list[str]]]]:
    if not tasks:
        raise ValueError("at least one validation task is required")

    root = Path(input_dir)
    samples: list[dict[str, Any]] = []
    sample_names: dict[str, str] = {}
    chain_maps: dict[str, dict[str, list[str]]] = {}

    for task in tasks:
        sample_name = task.sample_name
        sample_names[task.validation_id] = sample_name
        chain_maps[task.validation_id] = task.chain_role_map
        binder_chain = {
            "proteinChain": {
                "id": task.chain_role_map["binder"],
                "sequence": _normalize_sequence(task.designed_sequence),
                "count": 1,
            }
        }
        binder_msa = (binder_msas or {}).get(task.validation_id)
        if binder_msa is not None:
            binder_chain["proteinChain"].update(
                write_msa_files_for_input(
                    root,
                    prefix=f"{_safe_identifier(sample_name, max_len=80)}_binder",
                    msa=binder_msa,
                )
            )
        sequences: list[dict[str, Any]] = [binder_chain]
        task_target_msas = (target_msas or {}).get(task.validation_id)
        if (
            task_target_msas is not None
            and len(task_target_msas) != len(task.target_sequences)
        ):
            raise ValueError(
                f"target MSA count for {task.validation_id} does not match target sequence count"
            )
        for index, sequence in enumerate(task.target_sequences):
            chain = {
                "proteinChain": {
                    "id": [task.chain_role_map["target"][index]],
                    "sequence": _normalize_sequence(sequence),
                    "count": 1,
                }
            }
            msa = task_target_msas[index] if task_target_msas is not None else None
            if msa is not None:
                chain["proteinChain"].update(
                    write_msa_files_for_input(
                        root,
                        prefix=(
                            f"{_safe_identifier(sample_name, max_len=80)}"
                            f"_target{index}"
                        ),
                        msa=msa,
                    )
                )
            sequences.append(chain)
        sample: dict[str, Any] = {"name": sample_name, "sequences": sequences}
        task_template_specs = (template_specs or {}).get(task.validation_id)
        if task_template_specs:
            sample["templates"] = [
                _materialize_template_spec(
                    root,
                    sample_name=sample_name,
                    spec=spec,
                    index=index,
                )
                for index, spec in enumerate(task_template_specs)
            ]
        samples.append(sample)

    input_json = root / "input.json"
    write_json_atomic(input_json, samples, indent=2)
    return input_json, sample_names, chain_maps


def _materialize_template_spec(
    root: Path,
    *,
    sample_name: str,
    spec: ProtenixTemplateSpec,
    index: int,
) -> dict[str, Any]:
    source = Path(spec.path)
    if not source.exists():
        raise FileNotFoundError(f"Protenix template file does not exist: {source}")
    if len(spec.chain_ids) != len(spec.template_ids):
        raise ValueError(
            f"template mapping {spec.source} has {len(spec.chain_ids)} chain IDs "
            f"but {len(spec.template_ids)} template IDs"
        )
    suffix = source.suffix.lower()
    if suffix not in {".cif", ".mmcif", ".pdb"}:
        raise ValueError(f"unsupported Protenix structural template type: {source}")
    template_dir = root / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    dest = (
        template_dir
        / (
            f"{_safe_identifier(sample_name, max_len=70)}_"
            f"{index}_{_safe_identifier(spec.source, max_len=40)}{suffix}"
        )
    )
    shutil.copyfile(source, dest)
    payload_key = "pdb" if suffix == ".pdb" else "cif"
    return {
        payload_key: str(dest),
        "chain_id": list(spec.chain_ids),
        "template_id": list(spec.template_ids),
    }


def build_protenix_command(
    *,
    input_json: str | Path,
    output_dir: str | Path,
    config: ProtenixRunnerConfig,
    use_msa: bool | None = None,
    use_template: bool | None = None,
) -> list[str]:
    command = (
        list(config.protenix_command)
        if config.protenix_command is not None
        else [sys.executable, "-m", "runner.inference"]
    )
    command.extend(
        [
            "--input_json_path",
            str(input_json),
            "--dump_dir",
            str(output_dir),
            "--seeds",
            ",".join(str(seed) for seed in config.seeds),
            "--model_name",
            config.model_name,
            "--sample_diffusion.N_sample",
            str(config.n_sample),
            "--sample_diffusion.N_step",
            str(config.n_step),
            "--model.N_cycle",
            str(config.n_cycle),
            "--need_atom_confidence",
            "true",
            "--use_msa",
            str(_should_use_msa(config) if use_msa is None else bool(use_msa)).lower(),
            "--use_template",
            str(
                _should_use_template(config)
                if use_template is None
                else bool(use_template)
            ).lower(),
        ]
    )
    checkpoint_dir = config.checkpoint_dir or _checkpoint_dir_from_env()
    if checkpoint_dir is not None:
        command.extend(["--load_checkpoint_dir", str(checkpoint_dir)])
    return command


def collect_protenix_structures(
    *,
    output_dir: str | Path,
    task: ProtenixTaskInput,
    sample_name: str,
    chain_role_map: dict[str, list[str]],
    config: ProtenixRunnerConfig,
    campaign_dir: str | Path | None = None,
) -> tuple[ProtenixStructureResult, ...]:
    structures: list[ProtenixStructureResult] = []
    hotspot_context = (
        validation_hotspot_context(
            campaign_dir,
            chain_role_map=chain_role_map,
            contact_cutoff_angstrom=config.validation_hotspot_cutoff_angstrom,
        )
        if campaign_dir is not None
        else None
    )
    for seed, sample_rank, summary_path, full_data_path, cif_path in _iter_output_triples(
        Path(output_dir),
        sample_name=sample_name,
        seeds=config.seeds,
    ):
        summary = _load_json(summary_path)
        ipsae, ipsae_details = _scoped_ipsae_metric_for_outputs(
            summary,
            summary_path=summary_path,
            full_data_path=full_data_path,
            cif_path=cif_path,
            chain_role_map=chain_role_map,
            config=config,
        )
        metrics = _validation_metrics_from_summary(
            summary,
            chain_role_map=chain_role_map,
            min_validation_iptm=config.min_validation_iptm,
            min_validation_ipsae=config.min_validation_ipsae,
            ipsae=ipsae,
            ipsae_details=ipsae_details,
        )
        if hotspot_context is not None:
            metrics = _with_hotspot_validation_metrics(
                metrics,
                score_validation_hotspots(cif_path, context=hotspot_context),
            )
        status = "passing" if metrics.get("validation_passed") else "rejected"
        structures.append(
            ProtenixStructureResult(
                validation_id=task.validation_id,
                candidate_id=task.candidate_id,
                structure_id=f"seed{seed}_sample{sample_rank}",
                seed=seed,
                sample_rank=sample_rank,
                status=status,
                cif_path=cif_path,
                metrics={
                    **metrics,
                    "validation_model": task.model_name,
                    "validation_chain_role_map": chain_role_map,
                    "validation_metric_scope": "binder_target",
                    "source_summary_path": str(summary_path),
                    "source_cif_path": str(cif_path),
                    "binder_scaffold": task.binder_scaffold,
                    "framework": task.framework,
                },
            )
        )
    return tuple(sorted(structures, key=lambda item: (item.seed, item.sample_rank)))


def scoped_pair_metric(
    payload: dict[str, Any],
    *,
    keys: Sequence[str],
    chain_role_map: dict[str, Sequence[str]],
) -> dict[str, Any] | None:
    matrix = None
    metric_key = None
    for key in keys:
        converted = _convert_chain_pair_metric_to_matrix(payload.get(key))
        if converted is not None:
            matrix = converted
            metric_key = key
            break
    if matrix is None or metric_key is None:
        return None

    pairs: list[dict[str, Any]] = []
    for binder_chain in chain_role_map.get("binder", ()):
        for target_chain in chain_role_map.get("target", ()):
            values = _pair_values(matrix, binder_chain, target_chain)
            if not values:
                continue
            pairs.append(
                {
                    "binder_chain": binder_chain,
                    "target_chain": target_chain,
                    "value": min(values),
                    "directional_values": values,
                }
            )

    if not pairs:
        return None

    values = [float(pair["value"]) for pair in pairs]
    return {
        "source_key": metric_key,
        "value": min(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "pairs": pairs,
    }


def _fetch_task_inputs(
    conn,
    *,
    root: Path,
    claims: Sequence[ValidationClaim],
) -> list[ProtenixTaskInput]:
    target_sequences, target_labels = _target_sequences(root, conn)
    tasks: list[ProtenixTaskInput] = []
    for claim in claims:
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
        tasks.append(
            ProtenixTaskInput(
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
                designed_sequence=row["designed_sequence"],
                target_sequences=target_sequences,
                target_labels=target_labels,
                seed=int(row["seed"]),
                binder_scaffold=design_metrics.get("binder_scaffold"),
                framework=(
                    design_metrics.get("framework")
                    or design_metrics.get("framework_name")
                ),
                framework_source=design_metrics.get("framework_source"),
            )
        )
    return tasks


def _template_specs_by_task(
    root: Path,
    tasks: Sequence[ProtenixTaskInput],
    *,
    config: ProtenixRunnerConfig,
) -> dict[str, tuple[ProtenixTemplateSpec, ...]] | None:
    if config.use_template == "false":
        return None
    specs_by_task: dict[str, tuple[ProtenixTemplateSpec, ...]] = {}
    missing_required: list[str] = []
    for task in tasks:
        specs = _resolve_protenix_template_specs(root, task)
        if specs:
            specs_by_task[task.validation_id] = specs
        elif config.use_template == "true":
            missing_required.append(task.validation_id)
    if missing_required:
        joined = ", ".join(missing_required[:5])
        raise ValueError(
            "Protenix structural templates were required but no usable "
            f"target/framework template was found for validation task(s): {joined}"
        )
    return specs_by_task or None


def _resolve_protenix_template_specs(
    root: Path,
    task: ProtenixTaskInput,
) -> tuple[ProtenixTemplateSpec, ...]:
    specs: list[ProtenixTemplateSpec] = []
    target_spec = _target_template_spec(root, task)
    if target_spec is not None:
        specs.append(target_spec)
    framework_spec = _framework_template_spec(task)
    if framework_spec is not None:
        specs.append(framework_spec)
    return tuple(specs)


def _target_template_spec(
    root: Path,
    task: ProtenixTaskInput,
) -> ProtenixTemplateSpec | None:
    cif_path = root / "target" / "normalized_target.cif"
    summary_path = root / "target" / "chain_summary.json"
    if not cif_path.exists():
        return None
    if not summary_path.exists():
        raise FileNotFoundError(
            f"cannot use target structural template without chain summary: {summary_path}"
        )
    summary = _load_json(summary_path)
    chains = summary.get("chains")
    if not isinstance(chains, list):
        raise ValueError(f"{summary_path} does not contain a chains list")
    if len(chains) != len(task.target_sequences):
        raise ValueError(
            "target structural template chain count does not match Protenix "
            f"target sequence count for {task.validation_id}: "
            f"{len(chains)} template chain(s), {len(task.target_sequences)} sequence(s)"
        )
    template_ids: list[str] = []
    for index, chain in enumerate(chains):
        if not isinstance(chain, dict):
            raise ValueError(f"{summary_path}: chains[{index}] must be a mapping")
        template_sequence = chain.get("sequence")
        if isinstance(template_sequence, str) and template_sequence.strip():
            observed = _normalize_sequence(template_sequence)
            expected = _normalize_sequence(task.target_sequences[index])
            if observed != expected:
                raise ValueError(
                    "target structural template sequence does not match Protenix "
                    f"target sequence for {task.validation_id} chain {index}: "
                    f"{len(observed)} template residue(s), {len(expected)} input residue(s)"
                )
        template_id = chain.get("canonical_chain_id") or (
            task.target_labels[index] if index < len(task.target_labels) else None
        )
        if not template_id:
            raise ValueError(f"{summary_path}: chains[{index}] lacks canonical_chain_id")
        template_ids.append(str(template_id))
    return ProtenixTemplateSpec(
        path=cif_path,
        chain_ids=tuple(task.chain_role_map["target"]),
        template_ids=tuple(template_ids),
        source="target",
    )


def _framework_template_spec(task: ProtenixTaskInput) -> ProtenixTemplateSpec | None:
    path = _framework_template_path(task)
    if path is None:
        return None
    return ProtenixTemplateSpec(
        path=path,
        chain_ids=tuple(task.chain_role_map["binder"]),
        template_ids=("A",),
        source=f"framework:{_safe_identifier(task.framework or 'framework', max_len=60)}",
    )


def _framework_template_path(task: ProtenixTaskInput) -> Path | None:
    if str(task.framework_source or "").strip().lower() != "builtin":
        return None
    framework = str(task.framework or "").strip()
    if not framework:
        return None
    scaffold = _scaffold_key(task.binder_scaffold)
    try:
        if scaffold in {"scfv", "sc_fv"}:
            return get_scfv_framework_template_cif(framework)
        if scaffold == "vhh":
            return get_vhh_framework_template_cif(framework)
    except KeyError:
        return None
    return None


def _has_target_template(
    task: ProtenixTaskInput,
    template_specs: dict[str, tuple[ProtenixTemplateSpec, ...]] | None,
) -> bool:
    specs = (template_specs or {}).get(task.validation_id, ())
    covered = {
        chain_id
        for spec in specs
        if spec.source == "target"
        for chain_id in spec.chain_ids
    }
    return set(task.chain_role_map["target"]).issubset(covered)


def _has_framework_template(
    task: ProtenixTaskInput,
    template_specs: dict[str, tuple[ProtenixTemplateSpec, ...]] | None,
) -> bool:
    specs = (template_specs or {}).get(task.validation_id, ())
    covered = {
        chain_id
        for spec in specs
        if spec.source.startswith("framework:")
        for chain_id in spec.chain_ids
    }
    return set(task.chain_role_map["binder"]).issubset(covered)


def _target_msas_by_task(
    root: Path,
    tasks: Sequence[ProtenixTaskInput],
    *,
    config: ProtenixRunnerConfig,
    template_specs: dict[str, tuple[ProtenixTemplateSpec, ...]] | None = None,
) -> dict[str, tuple[MsaPair | None, ...]] | None:
    if config.target_msa_mode == "none":
        return None
    if not tasks:
        return None
    if not config.use_msa and all(
        _has_target_template(task, template_specs)
        for task in tasks
    ):
        return None
    first = tasks[0]
    target_name = _target_name(root)
    target_msas = resolve_target_msa_pairs(
        root,
        target_sequences=first.target_sequences,
        target_labels=first.target_labels,
        target_name=target_name,
        config=config.msa_config(),
    )
    return {task.validation_id: target_msas for task in tasks}


def _binder_msas_by_task(
    root: Path,
    tasks: Sequence[ProtenixTaskInput],
    *,
    config: ProtenixRunnerConfig,
    template_specs: dict[str, tuple[ProtenixTemplateSpec, ...]] | None = None,
    target_msas: dict[str, tuple[MsaPair | None, ...]] | None = None,
) -> dict[str, MsaPair | None] | None:
    if config.binder_msa_mode == "none":
        return None
    tasks_requiring_msas = _tasks_requiring_binder_msas(
        tasks,
        config=config,
        template_specs=template_specs,
        target_msas=target_msas,
    )
    if not tasks_requiring_msas:
        return None
    pairs = resolve_binder_msa_pairs(
        root,
        binders=tuple(
            (task.designed_sequence, task.binder_scaffold)
            for task in tasks_requiring_msas
        ),
        config=config.msa_config(),
    )
    return {
        task.validation_id: pair
        for task, pair in zip(tasks_requiring_msas, pairs)
    }


def _tasks_requiring_binder_msas(
    tasks: Sequence[ProtenixTaskInput],
    *,
    config: ProtenixRunnerConfig,
    template_specs: dict[str, tuple[ProtenixTemplateSpec, ...]] | None = None,
    target_msas: dict[str, tuple[MsaPair | None, ...]] | None = None,
) -> tuple[ProtenixTaskInput, ...]:
    if config.use_msa:
        return tuple(tasks)
    if config.binder_msa_mode == "single_sequence":
        return tuple(tasks)
    if target_msas is not None:
        return tuple(
            task
            for task in tasks
            if not _has_framework_template(task, template_specs)
        )
    return tuple(
        task
        for task in tasks
        if _scaffold_key(task.binder_scaffold) in {"vhh", "scfv", "sc_fv"}
        and not _has_framework_template(task, template_specs)
    )


def _target_name(root: Path) -> str | None:
    for name in ("resolved_config.yaml", "config.yaml"):
        path = root / name
        if not path.exists():
            continue
        try:
            import yaml

            payload = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        target = payload.get("target")
        if isinstance(target, dict) and target.get("name"):
            return str(target["name"])
    return None


def _target_sequences(root: Path, conn) -> tuple[tuple[str, ...], tuple[str, ...]]:
    row = conn.execute(
        "SELECT resolved_config_json FROM campaign WHERE id = 1"
    ).fetchone()
    resolved = json.loads(row["resolved_config_json"] or "{}") if row else {}
    target = resolved.get("target") if isinstance(resolved, dict) else None
    if isinstance(target, dict):
        direct = target.get("sequence")
        if isinstance(direct, str) and direct.strip():
            return (_normalize_sequence(direct),), ("B",)

        sequences = target.get("sequences")
        chains = target.get("chains")
        if isinstance(sequences, dict) and isinstance(chains, list):
            ordered = [
                _normalize_sequence(str(sequences[chain]))
                for chain in chains
                if chain in sequences and str(sequences[chain]).strip()
            ]
            if ordered:
                return tuple(ordered), tuple(str(chain) for chain in chains[: len(ordered)])

    summary_path = root / "target" / "chain_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        chains_payload = summary.get("chains")
        if isinstance(chains_payload, list):
            sequences_out: list[str] = []
            labels_out: list[str] = []
            for chain in chains_payload:
                if not isinstance(chain, dict):
                    continue
                sequence = chain.get("sequence")
                chain_id = chain.get("canonical_chain_id")
                if isinstance(sequence, str) and sequence.strip():
                    sequences_out.append(_normalize_sequence(sequence))
                    labels_out.append(str(chain_id or len(labels_out)))
            if sequences_out:
                return tuple(sequences_out), tuple(labels_out)

    raise ValueError(
        "cannot build Protenix input because the campaign does not expose target sequence metadata"
    )


def _should_use_msa(config: ProtenixRunnerConfig) -> bool:
    return bool(
        config.use_msa
        or config.target_msa_mode != "none"
        or config.binder_msa_mode == "single_sequence"
    )


def _should_use_template(config: ProtenixRunnerConfig) -> bool:
    return config.use_template == "true"


def _skip_reason(task: ProtenixTaskInput, *, config: ProtenixRunnerConfig) -> str | None:
    if (
        _scaffold_key(task.binder_scaffold) in {"scfv", "sc_fv"}
        and _requests_unsupported_scfv_binder_msa(config)
    ):
        return (
            "scFv binder MSA support is not implemented; use a bundled scFv "
            "framework with template-only binder validation, or set binder MSA "
            "mode to none when adding target MSAs"
        )
    if _is_unsupported_protenix_scaffold(
        task,
        model_name=config.model_name,
        use_template=config.use_template,
    ):
        return (
            "scFv Protenix validation requires a bundled scFv framework "
            "structural template; custom scFv frameworks still require MSA "
            "support that is not implemented"
        )
    if config.token_limit is not None and task.token_count > config.token_limit:
        return (
            f"validation input has {task.token_count} tokens, exceeding "
            f"{config.model_name} limit {config.token_limit}"
        )
    return None


def _is_unsupported_protenix_scaffold(
    task: ProtenixTaskInput,
    *,
    model_name: str,
    use_template: ProtenixTemplateMode,
) -> bool:
    normalized_model = model_name.strip().lower()
    if normalized_model != "protenix-v2" and not normalized_model.startswith(
        "protenix-"
    ):
        return False
    scaffold = _scaffold_key(task.binder_scaffold)
    if scaffold not in {"scfv", "sc_fv"}:
        return False
    if use_template == "false":
        return True
    return _framework_template_path(task) is None


def _requests_unsupported_scfv_binder_msa(config: ProtenixRunnerConfig) -> bool:
    if config.binder_msa_mode == "none":
        return False
    return bool(config.use_msa or config.binder_msa_mode == "single_sequence")


def _scaffold_key(binder_scaffold: str | None) -> str:
    return str(binder_scaffold or "").strip().lower()


def _scratch_dir(root: Path, *, worker_id: str, config: ProtenixRunnerConfig) -> Path:
    scratch_root = config.scratch_root or root / ".scratch" / "protenix_validation"
    return (
        scratch_root
        / _safe_identifier(worker_id, max_len=80)
        / f"batch_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )


def _cleanup_scratch_dir(
    scratch_dir: Path,
    *,
    root: Path,
    config: ProtenixRunnerConfig,
) -> None:
    shutil.rmtree(scratch_dir, ignore_errors=True)
    stop = (
        config.scratch_root.resolve()
        if config.scratch_root is not None
        else root.resolve()
    )
    current = scratch_dir.parent
    while True:
        try:
            resolved = current.resolve()
        except FileNotFoundError:
            resolved = current
        if resolved == stop:
            break
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _write_validation_debug_metadata(
    *,
    root: Path,
    scratch_dir: Path,
    input_json: Path | None,
    output_dir: Path | None,
    command: Sequence[str] | None,
    command_result: ProtenixCommandResult | None,
    runnable: Sequence[tuple[ValidationClaim, ProtenixTaskInput]],
    worker_id: str,
    gpu_id: str | None,
    config: ProtenixRunnerConfig,
    error_message: str | None = None,
) -> None:
    if not config.keep_debug:
        return
    model_name = runnable[0][1].model_name if runnable else config.model_name
    debug_dir = root / validator_dir(model_name) / "debug"
    batch_id = scratch_dir.name
    first_validation = runnable[0][0].validation_id if runnable else "validation"
    path = debug_dir / (
        f"{_safe_identifier(first_validation, max_len=80)}__{batch_id}.json"
    )
    payload: dict[str, Any] = {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "batch_id": batch_id,
        "scratch_dir": str(scratch_dir),
        "input_json_path": str(input_json) if input_json is not None else None,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "command": list(command) if command is not None else None,
        "attempts": [
            {
                "validation_id": claim.validation_id,
                "attempt_id": claim.attempt_id,
                "candidate_id": task.candidate_id,
                "selection_rank": task.selection_rank,
            }
            for claim, task in runnable
        ],
    }
    if command_result is not None:
        payload["protenix_returncode"] = command_result.returncode
        payload["stdout_tail"] = command_result.stdout_tail
        payload["stderr_tail"] = command_result.stderr_tail
    if error_message is not None:
        payload["error_message"] = error_message
    write_json_atomic(path, payload, indent=2)


def _run_protenix_command(
    command: Sequence[str],
    *,
    cwd: Path | None,
    env: dict[str, str] | None,
    gpu_id: str | None,
    timeout_seconds: int,
    heartbeat_interval_seconds: float,
    on_heartbeat: Callable[[], None] | None = None,
) -> ProtenixCommandResult:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    _remove_esm_repo_from_pythonpath(run_env)
    if command:
        _prepend_executable_dir_to_path(run_env, command[0])
    if gpu_id is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            env=run_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = _communicate_with_heartbeats(
            process,
            timeout_seconds=timeout_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            on_heartbeat=on_heartbeat,
        )
    except subprocess.TimeoutExpired as exc:
        if process is not None and process.poll() is None:
            process.kill()
            stdout, stderr = process.communicate()
        else:
            stdout = ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        raise RuntimeError(
            "Protenix inference timed out after "
            f"{timeout_seconds}s. stderr tail: {_tail_text(stderr)} "
            f"stdout tail: {_tail_text(stdout)}"
        ) from exc
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        raise

    returncode = 0 if process.returncode is None else int(process.returncode)
    if returncode != 0:
        stderr_tail = _tail_text(stderr)
        stdout_tail = _tail_text(stdout)
        raise RuntimeError(
            f"Protenix inference failed with exit code {returncode}. "
            f"stderr tail: {stderr_tail} stdout tail: {stdout_tail}"
        )
    return ProtenixCommandResult(
        returncode=returncode,
        stdout_tail=_tail_text(stdout),
        stderr_tail=_tail_text(stderr),
    )


def _remove_esm_repo_from_pythonpath(env: dict[str, str]) -> None:
    """Keep Biohub ESM from shadowing Protenix's fair-esm dependency."""
    pythonpath = env.get("PYTHONPATH")
    esm_repo = env.get("ESM_REPO")
    if not pythonpath or not esm_repo:
        return

    esm_repo_key = _normalized_pythonpath_entry(esm_repo)
    kept = [
        entry
        for entry in pythonpath.split(os.pathsep)
        if _normalized_pythonpath_entry(entry) != esm_repo_key
    ]
    if kept:
        env["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        env.pop("PYTHONPATH", None)


def _normalized_pythonpath_entry(value: str) -> str:
    if not value:
        return value
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except OSError:
        return value


def _communicate_with_heartbeats(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: int,
    heartbeat_interval_seconds: float,
    on_heartbeat: Callable[[], None] | None,
) -> tuple[str, str]:
    start = time.monotonic()
    deadline = start + timeout_seconds
    next_heartbeat = start + heartbeat_interval_seconds
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout_seconds)

        wait_for = min(max(0.05, next_heartbeat - now), remaining)
        try:
            stdout, stderr = process.communicate(timeout=wait_for)
            return stdout or "", stderr or ""
        except subprocess.TimeoutExpired:
            now = time.monotonic()
            if now >= deadline:
                raise
            if on_heartbeat is not None and now >= next_heartbeat:
                on_heartbeat()
                next_heartbeat = now + heartbeat_interval_seconds


def _heartbeat_validation_claims(
    store: CampaignStore,
    runnable: Sequence[tuple[ValidationClaim, ProtenixTaskInput]],
) -> None:
    for claim, _task in runnable:
        store.heartbeat_validation_task(
            validation_id=claim.validation_id,
            attempt_id=claim.attempt_id,
        )


def _iter_output_triples(
    output_dir: Path,
    *,
    sample_name: str,
    seeds: Sequence[int],
):
    for seed in seeds:
        pred_dir = output_dir / sample_name / f"seed_{seed}" / "predictions"
        if not pred_dir.exists():
            continue
        for summary_path in sorted(pred_dir.glob("*_summary_confidence_sample_*.json")):
            sample_rank = _sample_rank(summary_path)
            if sample_rank is None:
                continue
            cif_path = pred_dir / summary_path.name.replace(
                "_summary_confidence_sample_",
                "_sample_",
            ).replace(".json", ".cif")
            full_data_path = pred_dir / summary_path.name.replace(
                "_summary_confidence_sample_",
                "_full_data_sample_",
            )
            if not cif_path.exists():
                continue
            yield int(seed), int(sample_rank), summary_path, full_data_path, cif_path


def _scoped_ipsae_metric_for_outputs(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    full_data_path: Path,
    cif_path: Path,
    chain_role_map: dict[str, list[str]],
    config: ProtenixRunnerConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    details: dict[str, Any] = {}
    adapter_result = _run_ipsae_adapter(
        summary_path=summary_path,
        full_data_path=full_data_path,
        cif_path=cif_path,
        chain_role_map=chain_role_map,
        config=config,
    )
    if adapter_result.get("error"):
        details["validation_ipSAE_adapter_error"] = adapter_result["error"]
    else:
        adapter_metric = _ipsae_metric_from_adapter_result(adapter_result)
        if adapter_metric is not None:
            details["validation_ipSAE_adapter"] = "ipsae.py"
            return adapter_metric, details

    summary_metric = scoped_pair_metric(
        summary,
        keys=("chain_pair_ipsae", "chain_pair_ipSAE", "chain_pair_interface_ipsae"),
        chain_role_map=chain_role_map,
    )
    if summary_metric is not None:
        if "validation_ipSAE_adapter_error" in details:
            details["validation_ipSAE_fallback"] = "summary_chain_pair"
        return summary_metric, details
    return None, details


def _run_ipsae_adapter(
    *,
    summary_path: Path,
    full_data_path: Path,
    cif_path: Path,
    chain_role_map: dict[str, list[str]],
    config: ProtenixRunnerConfig,
) -> dict[str, Any]:
    script_path = _resolve_ipsae_script(config)
    if script_path is None:
        return {"error": "ipSAE adapter script not found"}
    script_path = script_path.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    full_data_path = full_data_path.expanduser().resolve()
    cif_path = cif_path.expanduser().resolve()
    try:
        adapter_dir = cif_path.parent / "_ipsae_adapter"
        adapted_full_path, _adapted_summary_path = _write_ipsae_adapter_files(
            full_data_path=full_data_path,
            summary_path=summary_path,
            out_dir=adapter_dir,
        )
    except Exception as exc:
        return {"error": f"ipSAE adapter input failed: {exc}"}

    command = [
        config.ipsae_python or sys.executable,
        str(script_path),
        str(adapted_full_path),
        str(cif_path),
        str(int(config.ipsae_pae_cutoff)),
        str(int(config.ipsae_dist_cutoff)),
    ]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=cif_path.parent,
        env=None if config.env is None else {**os.environ, **config.env},
        check=False,
    )
    if proc.returncode != 0:
        return {
            "error": f"ipSAE scorer failed: {proc.stderr[-500:] or proc.stdout[-500:]}",
            "raw_stdout": proc.stdout[-2000:],
            "raw_stderr": proc.stderr[-2000:],
        }

    txt_path = _ipsae_output_text_path(
        cif_path,
        pae_cutoff=config.ipsae_pae_cutoff,
        dist_cutoff=config.ipsae_dist_cutoff,
    )
    if not txt_path.exists():
        return {"error": "ipSAE output txt not found"}
    raw_text = txt_path.read_text()
    parsed = _parse_ipsae_output_filtered(
        output=raw_text,
        binder_chains=chain_role_map.get("binder", ()),
        partner_chains=chain_role_map.get("target", ()),
    )
    if not parsed:
        return {"error": "ipSAE produced no binder-target rows"}
    parsed["raw_text"] = raw_text
    return parsed


def _write_ipsae_adapter_files(
    *,
    full_data_path: Path,
    summary_path: Path,
    out_dir: Path,
) -> tuple[Path, Path]:
    full_data = _load_json(full_data_path)
    summary = _load_json(summary_path)

    pae = full_data.get("token_pair_pae")
    atom_plddt = full_data.get("atom_plddt")
    if pae is None or atom_plddt is None:
        raise ValueError("missing token_pair_pae or atom_plddt in Protenix full_data")

    atom_plddts = [float(value) for value in atom_plddt]
    if atom_plddts and max(atom_plddts) <= 1.5:
        atom_plddts = [value * 100.0 for value in atom_plddts]

    chain_pair_iptm = _convert_chain_pair_metric_to_matrix(
        summary.get("chain_pair_iptm")
    )
    adapted_full = {
        "pae": pae,
        "atom_plddts": atom_plddts,
    }
    adapted_summary = {
        "chain_pair_iptm": chain_pair_iptm or [[0.0, 0.0], [0.0, 0.0]],
    }

    adapted_full_path = out_dir / f"{full_data_path.stem}_ipsae.json"
    adapted_summary_path = out_dir / adapted_full_path.name.replace(
        "full_data",
        "summary_confidences",
    )
    write_json_atomic(adapted_full_path, adapted_full)
    write_json_atomic(adapted_summary_path, adapted_summary)
    return adapted_full_path, adapted_summary_path


def _resolve_ipsae_script(config: ProtenixRunnerConfig) -> Path | None:
    candidates: list[Path] = []
    if config.ipsae_script_path is not None:
        candidates.append(config.ipsae_script_path)
    for env_name in ("ESMFOLD2_IPSAE_SCRIPT", "PROTENIX_IPSAE_SCRIPT"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    candidates.append(Path(__file__).with_name("ipsae.py"))
    if config.protenix_root is not None:
        candidates.append(config.protenix_root / "ipsae.py")

    for path in candidates:
        expanded = path.expanduser()
        if expanded.exists():
            return expanded
    return None


def _ipsae_output_text_path(
    cif_path: Path,
    *,
    pae_cutoff: float,
    dist_cutoff: float,
) -> Path:
    pae_text = _ipsae_cutoff_text(pae_cutoff)
    dist_text = _ipsae_cutoff_text(dist_cutoff)
    expected = Path(str(cif_path).replace(".cif", f"_{pae_text}_{dist_text}.txt"))
    if expected.exists():
        return expected
    matches = sorted(cif_path.parent.glob(f"*_{pae_text}_{dist_text}.txt"))
    return matches[0] if matches else expected


def _ipsae_cutoff_text(value: float) -> str:
    integer = int(value)
    return f"{integer:02d}" if integer < 10 else str(integer)


def _parse_ipsae_output_filtered(
    *,
    output: str,
    binder_chains: Sequence[str],
    partner_chains: Sequence[str],
) -> dict[str, Any]:
    bset = set(binder_chains)
    pset = set(partner_chains)
    max_rows: list[dict[str, Any]] = []
    asym_values: list[float] = []

    for line in output.strip().splitlines():
        line = line.strip()
        if not line or line.startswith(("Chn1", "#")):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        chain1, chain2 = parts[0], parts[1]
        is_scoped_pair = (
            (chain1 in bset and chain2 in pset)
            or (chain2 in bset and chain1 in pset)
        )
        if not is_scoped_pair:
            continue
        row_type = parts[4].lower()
        try:
            if row_type == "max":
                row = {
                    "binder_chain": chain1 if chain1 in bset else chain2,
                    "target_chain": chain2 if chain1 in bset else chain1,
                    "chain1": chain1,
                    "chain2": chain2,
                    "ipSAE": float(parts[5]),
                    "ipSAE_d0chn": float(parts[6]),
                    "ipSAE_d0dom": float(parts[7]),
                    "ipTM_af": float(parts[8]),
                    "pDockQ": float(parts[10]),
                    "pDockQ2": float(parts[11]),
                    "LIS": float(parts[12]),
                }
                if len(parts) > 13:
                    row["n0res"] = int(parts[13])
                if len(parts) > 14:
                    row["n0chn"] = int(parts[14])
                max_rows.append(row)
            elif row_type == "asym":
                asym_values.append(float(parts[5]))
        except (TypeError, ValueError):
            continue

    result: dict[str, Any] = {}
    if max_rows:
        best = max(max_rows, key=lambda row: row.get("ipSAE", 0.0))
        result.update(best)
        result["rows"] = max_rows
    if asym_values:
        result["ipSAE_min"] = min(asym_values)
        result["ipSAE_max"] = max(asym_values)
    return result


def _ipsae_metric_from_adapter_result(
    result: dict[str, Any],
) -> dict[str, Any] | None:
    value = _float_or_none(result.get("ipSAE"))
    if value is None:
        return None
    min_value = _float_or_none(result.get("ipSAE_min"))
    max_value = _float_or_none(result.get("ipSAE_max"))
    pairs = result.get("rows") if isinstance(result.get("rows"), list) else []
    metric: dict[str, Any] = {
        "source_key": "ipsae.py",
        "value": value,
        "min": min_value if min_value is not None else value,
        "max": max_value if max_value is not None else value,
        "mean": None,
        "pairs": pairs,
    }
    metric["adapter_metrics"] = {
        key: result[key]
        for key in (
            "ipSAE_d0chn",
            "ipSAE_d0dom",
            "ipTM_af",
            "pDockQ",
            "pDockQ2",
            "LIS",
            "n0res",
            "n0chn",
        )
        if key in result
    }
    return metric


def _validation_metrics_from_summary(
    summary: dict[str, Any],
    *,
    chain_role_map: dict[str, list[str]],
    min_validation_iptm: float | None,
    min_validation_ipsae: float | None,
    ipsae: dict[str, Any] | None = None,
    ipsae_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    iptm = scoped_pair_metric(
        summary,
        keys=("chain_pair_iptm", "chain_pair_ipTM", "chain_pair_interface_iptm"),
        chain_role_map=chain_role_map,
    )

    metrics: dict[str, Any] = {
        "validation_global_iptm": _float_or_none(summary.get("iptm")),
        "ptm": _float_or_none(summary.get("ptm")),
        "ranking_score": _float_or_none(summary.get("ranking_score")),
    }
    fail_reasons: list[str] = []
    if iptm is None:
        fail_reasons.append("missing scoped binder-target chain_pair_iptm")
    else:
        metrics.update(
            {
                "validation_iptm": iptm["value"],
                "validation_iptm_min": iptm["min"],
                "validation_iptm_max": iptm["max"],
                "validation_iptm_mean": iptm["mean"],
                "validation_iptm_pairs": iptm["pairs"],
                "validation_iptm_source_key": iptm["source_key"],
            }
        )

    if ipsae_details:
        metrics.update(ipsae_details)
    if ipsae is None:
        missing_ipsae_reason = "missing scoped binder-target validation_ipSAE"
        metrics["validation_ipSAE_warning"] = missing_ipsae_reason
        if min_validation_ipsae is not None:
            fail_reasons.append(missing_ipsae_reason)
    else:
        adapter_metrics = ipsae.get("adapter_metrics")
        if isinstance(adapter_metrics, dict):
            metrics.update(
                {
                    f"validation_{key}": value
                    for key, value in adapter_metrics.items()
                }
            )
        metrics.update(
            {
                "validation_ipSAE": ipsae["value"],
                "validation_ipSAE_min": ipsae["min"],
                "validation_ipSAE_max": ipsae["max"],
                "validation_ipSAE_mean": ipsae.get("mean"),
                "validation_ipSAE_pairs": ipsae.get("pairs", []),
                "validation_ipSAE_source_key": ipsae["source_key"],
            }
        )

    if min_validation_ipsae is not None:
        metrics["min_validation_ipSAE"] = min_validation_ipsae

    validation_iptm = metrics.get("validation_iptm")
    if (
        min_validation_iptm is not None
        and validation_iptm is not None
        and float(validation_iptm) < min_validation_iptm
    ):
        fail_reasons.append(
            f"validation_iptm {float(validation_iptm):.4f} below threshold {min_validation_iptm:.4f}"
        )

    validation_ipsae = metrics.get("validation_ipSAE")
    if min_validation_ipsae is not None:
        if validation_ipsae is None:
            metrics["validation_ipSAE_pass"] = False
        else:
            validation_ipsae_pass = float(validation_ipsae) >= min_validation_ipsae
            metrics["validation_ipSAE_pass"] = validation_ipsae_pass
            if not validation_ipsae_pass:
                fail_reasons.append(
                    f"validation_ipSAE {float(validation_ipsae):.4f} below threshold {min_validation_ipsae:.4f}"
                )

    if fail_reasons:
        metrics["validation_passed"] = False
        metrics["fail_reason"] = "; ".join(fail_reasons)
    else:
        metrics["validation_passed"] = True
        metrics["pass_reason"] = "scoped Protenix validation metrics passed"
    return metrics


def _with_hotspot_validation_metrics(
    metrics: dict[str, Any],
    hotspot_metrics: dict[str, Any],
) -> dict[str, Any]:
    combined = {**metrics, **hotspot_metrics}
    if hotspot_metrics.get("validation_hotspot_pass") is False:
        combined["validation_passed"] = False
        fail_reasons = [
            reason
            for reason in (
                metrics.get("fail_reason"),
                hotspot_metrics.get("validation_hotspot_fail_reason"),
            )
            if reason
        ]
        combined["fail_reason"] = "; ".join(str(reason) for reason in fail_reasons)
        combined.pop("pass_reason", None)
    elif metrics.get("validation_passed"):
        reason = hotspot_metrics.get("validation_hotspot_pass_reason")
        if reason:
            combined["pass_reason"] = f"{metrics.get('pass_reason')}; {reason}"
    return combined


def _publish_structure(
    *,
    root: Path,
    store: CampaignStore,
    task: ProtenixTaskInput,
    structure: ProtenixStructureResult,
) -> str:
    stem = _structure_stem(task, structure)
    pending_relpath = validator_structure_status_dir(
        task.model_name,
        VALIDATION_STAGING_DIR,
    ) / f"{stem}.cif"
    final_relpath = validator_structure_status_dir(
        task.model_name,
        _status_dir(structure.status),
    ) / f"{stem}.cif"

    data = structure.cif_path.read_bytes()
    write_bytes_atomic(root / pending_relpath, data)
    store.record_validation_structure(
        validation_id=task.validation_id,
        structure_id=structure.structure_id,
        candidate_id=task.candidate_id,
        model_name=task.model_name,
        seed=structure.seed,
        sample_rank=structure.sample_rank,
        status="pending",
        structure_path=pending_relpath.as_posix(),
        metrics=structure.metrics,
    )

    write_bytes_atomic(root / final_relpath, data)
    store.record_validation_structure(
        validation_id=task.validation_id,
        structure_id=structure.structure_id,
        candidate_id=task.candidate_id,
        model_name=task.model_name,
        seed=structure.seed,
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


def _best_structure(
    structures: Sequence[ProtenixStructureResult],
) -> ProtenixStructureResult:
    if not structures:
        raise ValueError("cannot select best structure from an empty list")
    return max(
        structures,
        key=lambda item: (
            item.status == "passing",
            _metric_sort_value(item.metrics, "validation_iptm"),
            _metric_sort_value(item.metrics, "validation_ipSAE"),
            _metric_sort_value(item.metrics, "ranking_score"),
            -item.sample_rank,
        ),
    )


def _metric_sort_value(metrics: dict[str, Any], key: str) -> float:
    value = _float_or_none(metrics.get(key))
    return value if value is not None else -1.0


def _structure_stem(
    task: ProtenixTaskInput,
    structure: ProtenixStructureResult,
) -> str:
    seed_part = (
        ""
        if structure.seed == DEFAULT_PROTENIX_SEEDS[0]
        else f"__seed{structure.seed}"
    )
    config_part = (
        f"__cfg-{_safe_identifier(task.validation_config_hash, max_len=12)}"
        if task.needs_config_suffix
        else ""
    )
    return (
        f"{_safe_identifier(task.candidate_id, max_len=80)}__"
        f"{validator_slug(task.model_name)}"
        f"{config_part}"
        f"{seed_part}__"
        f"sample{structure.sample_rank:02d}"
    )


def _status_dir(status: str) -> str:
    if status == "passing":
        return VALIDATION_PASSING_DIR
    if status == "rejected":
        return VALIDATION_REJECTED_DIR
    if status == "pending":
        return VALIDATION_STAGING_DIR
    raise ValueError(f"unsupported validation structure status: {status}")


def _convert_chain_pair_metric_to_matrix(value: Any) -> list[list[float | None]] | None:
    if value is None:
        return None
    if isinstance(value, list):
        matrix: list[list[float | None]] = []
        for row in value:
            if not isinstance(row, list):
                return None
            matrix.append([_float_or_none(item) for item in row])
        return matrix
    if isinstance(value, dict):
        idxs = sorted(int(key) for key in value if str(key).isdigit())
        if not idxs:
            return None
        n = max(idxs) + 1
        matrix = [[None for _col in range(n)] for _row in range(n)]
        for row_key, row_value in value.items():
            if not str(row_key).isdigit():
                continue
            row_idx = int(row_key)
            if isinstance(row_value, dict):
                for col_key, item in row_value.items():
                    if str(col_key).isdigit():
                        col_idx = int(col_key)
                        if 0 <= col_idx < n:
                            matrix[row_idx][col_idx] = _float_or_none(item)
            elif isinstance(row_value, list):
                for col_idx, item in enumerate(row_value[:n]):
                    matrix[row_idx][col_idx] = _float_or_none(item)
        return matrix
    return None


def _pair_values(
    matrix: Sequence[Sequence[float | None]],
    chain_a: str,
    chain_b: str,
) -> list[float]:
    values: list[float] = []
    for row_idx, col_idx in (
        (_chain_index(chain_a), _chain_index(chain_b)),
        (_chain_index(chain_b), _chain_index(chain_a)),
    ):
        if row_idx < 0 or col_idx < 0:
            continue
        if row_idx >= len(matrix) or col_idx >= len(matrix[row_idx]):
            continue
        value = matrix[row_idx][col_idx]
        if value is not None:
            values.append(float(value))
    return values


def _sample_rank(path: Path) -> int | None:
    match = re.search(r"_sample_(\d+)\.json$", path.name)
    if match:
        return int(match.group(1))
    return None


def _chain_id_for_index(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return alphabet[index]
    raise ValueError("Protenix validation currently supports at most 26 chains")


def _chain_index(chain_id: str) -> int:
    if len(chain_id) != 1:
        return -1
    char = chain_id.upper()
    if not ("A" <= char <= "Z"):
        return -1
    return ord(char) - ord("A")


def _normalize_sequence(sequence: str) -> str:
    value = "".join(str(sequence or "").split()).upper()
    if not value:
        raise ValueError("sequence cannot be empty")
    if "|" in value:
        raise ValueError("Protenix validation task sequences must be individual chains")
    return value


def _safe_identifier(value: str, *, max_len: int) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value.strip()]
    text = "".join(chars).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    if not text:
        text = "item"
    return text[:max_len].strip("_") or "item"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _checkpoint_dir_from_env() -> Path | None:
    value = os.environ.get("PROTENIX_CHECKPOINT_DIR")
    if not value:
        return None
    return Path(value).expanduser()


def _tail_text(value: str, *, limit: int = 1200) -> str:
    text = " ".join((value or "").split())
    return text[-limit:]


def _prepend_executable_dir_to_path(env: dict[str, str], executable: str) -> None:
    path = Path(executable).expanduser()
    if path.parent == Path(".") or not path.parent.exists():
        return
    existing = env.get("PATH")
    env["PATH"] = str(path.parent) if not existing else f"{path.parent}{os.pathsep}{existing}"
