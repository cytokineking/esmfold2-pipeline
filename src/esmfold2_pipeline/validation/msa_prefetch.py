from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
from typing import Any, Callable

from esmfold2_pipeline.db import CampaignStore, MsaJobClaim, initialize_database
from esmfold2_pipeline.frameworks import (
    get_scfv_framework_template_cif,
    get_vhh_framework_template_cif,
)
from esmfold2_pipeline.validation.msa import (
    MsaPair,
    ProtenixMsaConfig,
    analyze_vhh_sequence,
    binder_msa_cache_dir,
    msa_context_hash,
    normalize_sequence,
    read_cached_msa_pair,
    resolve_binder_msa_pair,
    resolve_binder_msa_pairs,
    resolve_target_msa_pairs,
    sha256_text,
    target_msa_cache_dir,
    vhh_template_group_cache_dir,
)

DEFAULT_MSA_MAX_REQUESTS_PER_MINUTE = 5.0
MSA_RATE_LIMIT_NAME = "colabfold_msa"


@dataclass(frozen=True)
class MsaPrefetchEnqueueResult:
    queued_jobs: int
    existing_jobs: int
    skipped: bool
    decisions: tuple[str, ...]


@dataclass(frozen=True)
class MsaPrefetchPlanResult:
    candidate_count: int
    queued_jobs: int
    existing_jobs: int
    skipped_candidates: int


@dataclass(frozen=True)
class MsaPrefetchRunResult:
    completed_jobs: int
    failed_jobs: int
    skipped_jobs: int
    no_pending: bool
    recovered_stale_jobs: int = 0


@dataclass(frozen=True)
class MsaJobSpec:
    scope: str
    cache_key: str
    context_hash: str
    representative_sequence: str | None
    member_sequences: tuple[str, ...]
    metadata: dict[str, Any]
    reason: str

    @property
    def msa_job_id(self) -> str:
        digest = hashlib.sha256(
            f"{self.scope}|{self.cache_key}|{self.context_hash}".encode("utf-8")
        ).hexdigest()[:16]
        return f"msa_{_safe_identifier(self.scope, max_len=32)}_{digest}"


def enqueue_msa_prefetch_for_candidate(
    campaign_dir: str | Path,
    *,
    store: CampaignStore | None = None,
    candidate_id: str,
    critic_metrics: dict[str, Any] | None = None,
    validation_config_hash: str | None = None,
    validation_config_override: dict[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> MsaPrefetchEnqueueResult:
    root = Path(campaign_dir)
    own_conn = None
    if store is None:
        own_conn = initialize_database(root / "campaign.sqlite")
        store = CampaignStore(own_conn)
    try:
        result = _enqueue_for_candidate(
            root,
            store=store,
            candidate_id=candidate_id,
            critic_metrics=critic_metrics,
            validation_config_hash=validation_config_hash,
            validation_config_override=validation_config_override,
            log=log,
        )
    finally:
        if own_conn is not None:
            own_conn.close()
    return result


def plan_msa_prefetch(
    campaign_dir: str | Path,
    *,
    validation_config_hash: str | None = None,
    validation_config_override: dict[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> MsaPrefetchPlanResult:
    root = Path(campaign_dir)
    conn = initialize_database(root / "campaign.sqlite")
    store = CampaignStore(conn)
    try:
        rows = conn.execute(
            """
            SELECT
                c.candidate_id,
                cm.metrics_json
            FROM candidates AS c
            JOIN critic_metrics AS cm
              ON cm.candidate_id = c.candidate_id
            WHERE c.status = 'completed'
              AND cm.status = 'completed'
            ORDER BY cm.completed_at, c.candidate_id
            """
        ).fetchall()
        queued = 0
        existing = 0
        skipped = 0
        for row in rows:
            metrics = _json_dict(row["metrics_json"])
            result = _enqueue_for_candidate(
                root,
                store=store,
                candidate_id=row["candidate_id"],
                critic_metrics=metrics,
                validation_config_hash=validation_config_hash,
                validation_config_override=validation_config_override,
                log=log,
            )
            queued += result.queued_jobs
            existing += result.existing_jobs
            if result.skipped:
                skipped += 1
        return MsaPrefetchPlanResult(
            candidate_count=len(rows),
            queued_jobs=queued,
            existing_jobs=existing,
            skipped_candidates=skipped,
        )
    finally:
        conn.close()


def run_msa_prefetch_worker(
    campaign_dir: str | Path,
    *,
    worker_id: str = "msa-prefetch-worker-0",
    max_jobs: int | None = None,
    max_requests_per_minute: float = DEFAULT_MSA_MAX_REQUESTS_PER_MINUTE,
    stale_timeout_seconds: float | None = None,
    log: Callable[[str], None] | None = None,
) -> MsaPrefetchRunResult:
    if max_jobs is not None and max_jobs < 0:
        raise ValueError("max_jobs must be non-negative when provided")
    if max_requests_per_minute <= 0:
        raise ValueError("max_requests_per_minute must be positive")
    if stale_timeout_seconds is not None and stale_timeout_seconds <= 0:
        raise ValueError("stale_timeout_seconds must be positive when provided")
    root = Path(campaign_dir)
    conn = initialize_database(root / "campaign.sqlite")
    store = CampaignStore(conn)
    completed = 0
    failed = 0
    skipped = 0
    recovered = 0
    try:
        if stale_timeout_seconds is not None:
            recovered = store.recover_stale_msa_jobs(
                stale_before=_stale_before_timestamp(stale_timeout_seconds),
                error_message=(
                    "running MSA job heartbeat exceeded "
                    f"{stale_timeout_seconds:g}s; recovering for resume"
                ),
            )
            if recovered:
                _log(log, f"recovered stale MSA jobs: {recovered}")
        if max_jobs != 0:
            reopened = store.reopen_ready_msa_jobs_with_missing_cache(base_dir=root)
            if reopened:
                _log(log, f"reopened ready MSA jobs with missing cache: {reopened}")
        while max_jobs is None or completed + failed + skipped < max_jobs:
            claim = store.claim_next_pending_msa_job(
                worker_id=worker_id,
                hostname=socket.gethostname(),
                pid=_pid(),
            )
            if claim is None:
                break
            try:
                outcome = _run_msa_job(
                    root,
                    store=store,
                    claim=claim,
                    max_requests_per_minute=max_requests_per_minute,
                    log=log,
                )
                if outcome == "skipped":
                    skipped += 1
                else:
                    completed += 1
            except Exception as exc:
                _log(log, f"MSA job {claim.msa_job_id}: failed: {exc}")
                store.fail_msa_job(
                    msa_job_id=claim.msa_job_id,
                    error_message=str(exc),
                    retry_after_seconds=30.0,
                )
                failed += 1
    finally:
        conn.close()
    total = completed + failed + skipped
    return MsaPrefetchRunResult(
        completed_jobs=completed,
        failed_jobs=failed,
        skipped_jobs=skipped,
        no_pending=total == 0,
        recovered_stale_jobs=recovered,
    )


def _stale_before_timestamp(stale_timeout_seconds: float) -> str:
    stale_before = datetime.now(timezone.utc) - timedelta(seconds=stale_timeout_seconds)
    return stale_before.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _enqueue_for_candidate(
    root: Path,
    *,
    store: CampaignStore,
    candidate_id: str,
    critic_metrics: dict[str, Any] | None,
    validation_config_hash: str | None,
    validation_config_override: dict[str, Any] | None,
    log: Callable[[str], None] | None,
) -> MsaPrefetchEnqueueResult:
    row = store.conn.execute(
        """
        SELECT
            c.candidate_id,
            c.designed_sequence,
            c.design_metrics_json,
            cm.metrics_json
        FROM candidates AS c
        LEFT JOIN critic_metrics AS cm
          ON cm.candidate_id = c.candidate_id
         AND cm.status = 'completed'
        WHERE c.candidate_id = ?
        """,
        (candidate_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"unknown candidate: {candidate_id}")

    resolved_config = _resolved_config(store)
    validation = validation_config_override or _validation_config(resolved_config)
    if not validation:
        return MsaPrefetchEnqueueResult(
            queued_jobs=0,
            existing_jobs=0,
            skipped=True,
            decisions=("no validation config; no MSA prefetch configured",),
        )

    metrics = critic_metrics or _json_dict(row["metrics_json"])
    threshold = _prefetch_min_esm_iptm(validation)
    iptm = _float_or_none(metrics.get("iptm"))
    if threshold is not None and (iptm is None or iptm < threshold):
        return _skip(
            candidate_id,
            f"design below prefetch threshold {threshold:g}; no MSA queued",
            log,
        )
    require_hotspot = str(validation.get("require_hotspot_contact") or "auto").lower()
    if require_hotspot == "always" and not _has_hotspot_contact(metrics):
        return _skip(
            candidate_id,
            "design failed hotspot prefetch filter; no MSA queued",
            log,
        )

    design_metrics = _json_dict(row["design_metrics_json"])
    binder_scaffold = str(design_metrics.get("binder_scaffold") or "miniprotein").lower()
    template_mode = _template_mode_from_validation(validation)
    msa_config = _msa_config_from_validation(root, validation)
    if (
        binder_scaffold in {"scfv", "sc_fv"}
        and not _framework_template_available(
            design_metrics,
            binder_scaffold=binder_scaffold,
            template_mode=template_mode,
        )
    ):
        return _skip(
            candidate_id,
            (
                "scFv Protenix validation requires a bundled framework "
                "structural template; no MSA queued"
            ),
            log,
        )
    if (
        binder_scaffold in {"scfv", "sc_fv"}
        and _requests_unsupported_scfv_binder_msa(
            explicit_use_msa=_explicit_use_msa(validation),
            binder_mode=msa_config.binder_mode,
        )
    ):
        return _skip(
            candidate_id,
            (
                "scFv binder MSA support is not implemented; set "
                "validation.msa.binder: none to run target/template MSA only"
            ),
            log,
        )
    sequence = normalize_sequence(row["designed_sequence"])
    specs, decisions = _msa_job_specs_for_candidate(
        root,
        candidate_id=candidate_id,
        designed_sequence=sequence,
        binder_scaffold=binder_scaffold,
        design_metrics=design_metrics,
        resolved_config=resolved_config,
        validation=validation,
        msa_config=msa_config,
    )

    queued = 0
    existing = 0
    for spec in specs:
        created = store.create_or_update_msa_job(
            msa_job_id=spec.msa_job_id,
            scope=spec.scope,
            cache_key=spec.cache_key,
            msa_context_hash=spec.context_hash,
            candidate_id=candidate_id,
            reason=spec.reason,
            representative_sequence=spec.representative_sequence,
            member_sequences=spec.member_sequences,
            metadata=spec.metadata,
            validation_config_hash=validation_config_hash,
        )
        if created:
            queued += 1
        else:
            existing += 1
    for decision in decisions:
        _log(log, f"candidate {candidate_id}: {decision}")
    return MsaPrefetchEnqueueResult(
        queued_jobs=queued,
        existing_jobs=existing,
        skipped=False,
        decisions=tuple(decisions),
    )


def _msa_job_specs_for_candidate(
    root: Path,
    *,
    candidate_id: str,
    designed_sequence: str,
    binder_scaffold: str,
    design_metrics: dict[str, Any],
    resolved_config: dict[str, Any],
    validation: dict[str, Any],
    msa_config: ProtenixMsaConfig,
) -> tuple[list[MsaJobSpec], list[str]]:
    specs: list[MsaJobSpec] = []
    decisions: list[str] = []
    context_hash = msa_context_hash(msa_config)

    target_mode = msa_config.target_mode
    target_sequences, target_labels = _target_sequences_from_config(
        root,
        resolved_config,
    )
    template_mode = _template_mode_from_validation(validation)
    explicit_use_msa = _explicit_use_msa(validation)
    target_template_available = (
        template_mode != "false"
        and _target_template_available(root, target_sequences=target_sequences)
    )
    if target_mode != "none":
        if target_template_available and not explicit_use_msa:
            decisions.append(
                "target structural template available; target MSA prefetch suppressed"
            )
        else:
            for sequence, label in zip(target_sequences, target_labels):
                normalized = normalize_sequence(sequence)
                seq_hash = sha256_text(normalized)
                specs.append(
                    MsaJobSpec(
                        scope="target",
                        cache_key=f"target:{seq_hash}",
                        context_hash=context_hash,
                        representative_sequence=normalized,
                        member_sequences=(normalized,),
                        reason=f"target:{label}",
                        metadata={
                            "msa_config": _msa_config_payload(msa_config, root=root),
                            "target_sequence": normalized,
                            "target_label": label,
                            "target_name": _target_name(resolved_config),
                        },
                    )
                )
                decisions.append(f"target {label}: MSA prefetch queued")

    target_msa_active = any(spec.scope == "target" for spec in specs)
    use_msa = bool(explicit_use_msa or target_msa_active)
    binder_mode = msa_config.binder_mode
    framework_template_available = _framework_template_available(
        design_metrics,
        binder_scaffold=binder_scaffold,
        template_mode=template_mode,
    )
    if binder_mode == "none":
        decisions.append("binder MSA mode none; no binder MSA queued")
    elif binder_scaffold in {"miniprotein", "minibinder", ""}:
        if use_msa or binder_mode == "single_sequence":
            seq_hash = sha256_text(designed_sequence)
            specs.append(
                MsaJobSpec(
                    scope="miniprotein_single_sequence",
                    cache_key=f"miniprotein:{seq_hash}",
                    context_hash=context_hash,
                    representative_sequence=designed_sequence,
                    member_sequences=(designed_sequence,),
                    reason="binder:miniprotein_single_sequence",
                    metadata={
                        "msa_config": _msa_config_payload(msa_config, root=root),
                        "binder_sequence": designed_sequence,
                        "binder_scaffold": binder_scaffold or "miniprotein",
                    },
                )
            )
            decisions.append("miniprotein binder; single-sequence MSA queued")
        else:
            decisions.append("miniprotein binder; no binder MSA server query required")
    elif binder_scaffold == "vhh":
        if framework_template_available and not explicit_use_msa:
            decisions.append(
                "VHH framework structural template available; "
                "binder MSA prefetch suppressed"
            )
        else:
            analysis = analyze_vhh_sequence(designed_sequence)
            template_hash = str(analysis["lengths_only_template_key_hash"])
            specs.append(
                MsaJobSpec(
                    scope="vhh_binder_group",
                    cache_key=f"vhh:{template_hash}",
                    context_hash=context_hash,
                    representative_sequence=designed_sequence,
                    member_sequences=(designed_sequence,),
                    reason="binder:vhh_template",
                    metadata={
                        "msa_config": _msa_config_payload(msa_config, root=root),
                        "binder_sequence": designed_sequence,
                        "binder_scaffold": "vhh",
                        "vhh_analysis": analysis,
                        "template_key_hash": template_hash,
                        "candidate_id": candidate_id,
                    },
                )
            )
            decisions.append(f"VHH template {template_hash}: MSA prefetch queued")
    elif binder_scaffold in {"scfv", "sc_fv"}:
        if framework_template_available and not explicit_use_msa:
            decisions.append(
                "scFv framework structural template available; "
                "binder MSA prefetch suppressed"
            )
        else:
            decisions.append("scFv binder MSA grouping pending; no binder MSA queued")
    else:
        decisions.append(
            f"unsupported binder scaffold {binder_scaffold}; no binder MSA queued"
        )
    return specs, decisions


def _run_msa_job(
    root: Path,
    *,
    store: CampaignStore,
    claim: MsaJobClaim,
    max_requests_per_minute: float,
    log: Callable[[str], None] | None,
) -> str:
    if claim.scope == "target":
        return _run_target_msa_job(
            root,
            store=store,
            claim=claim,
            max_requests_per_minute=max_requests_per_minute,
            log=log,
        )
    if claim.scope == "vhh_binder_group":
        return _run_vhh_msa_job(
            root,
            store=store,
            claim=claim,
            max_requests_per_minute=max_requests_per_minute,
            log=log,
        )
    if claim.scope == "miniprotein_single_sequence":
        return _run_miniprotein_msa_job(root, store=store, claim=claim, log=log)
    if claim.scope == "scfv_binder_group":
        store.skip_msa_job(
            msa_job_id=claim.msa_job_id,
            error_message="scFv binder MSA grouping is not implemented yet",
        )
        _log(log, f"MSA job {claim.msa_job_id}: scFv support pending; skipped")
        return "skipped"
    raise ValueError(f"unsupported MSA job scope: {claim.scope}")


def _run_target_msa_job(
    root: Path,
    *,
    store: CampaignStore,
    claim: MsaJobClaim,
    max_requests_per_minute: float,
    log: Callable[[str], None] | None,
) -> str:
    config = _msa_config_from_payload(root, claim.metadata.get("msa_config"))
    sequence = normalize_sequence(
        claim.metadata.get("target_sequence") or claim.representative_sequence or ""
    )
    label = str(claim.metadata.get("target_label") or "B")
    target_name = claim.metadata.get("target_name")
    cache_dir = target_msa_cache_dir(root, sequence=sequence, config=config)
    cached = read_cached_msa_pair(cache_dir, sequence=sequence, config=config)
    if cached is not None:
        _log(log, f"target {target_name or label}: target MSA cache hit")
        store.complete_msa_job(
            msa_job_id=claim.msa_job_id,
            cache_paths=_cache_paths(cache_dir),
            metadata={"decision": "cache_hit"},
        )
        return "completed"
    if config.target_mode == "server":
        _log(
            log,
            f"target {target_name or label}: target MSA missing; acquiring MMseqs rate slot",
        )
        _acquire_rate_slot(
            store,
            max_requests_per_minute=max_requests_per_minute,
            log=log,
        )
        _log(log, f"target {target_name or label}: submitting target MSA to MMseqs")
    elif config.target_mode == "provided":
        _log(log, f"target {target_name or label}: provided target MSA cache missing; importing")
    else:
        store.skip_msa_job(
            msa_job_id=claim.msa_job_id,
            error_message="target MSA mode none",
        )
        return "skipped"
    pairs = resolve_target_msa_pairs(
        root,
        target_sequences=(sequence,),
        target_labels=(label,),
        target_name=str(target_name) if target_name else None,
        config=config,
    )
    pair = pairs[0]
    if pair is None or pair.cache_dir is None:
        raise RuntimeError("target MSA resolver did not produce a cached MSA")
    store.complete_msa_job(
        msa_job_id=claim.msa_job_id,
        cache_paths=_cache_paths(pair.cache_dir),
        metadata={"decision": "resolved"},
    )
    return "completed"


def _run_vhh_msa_job(
    root: Path,
    *,
    store: CampaignStore,
    claim: MsaJobClaim,
    max_requests_per_minute: float,
    log: Callable[[str], None] | None,
) -> str:
    config = _msa_config_from_payload(root, claim.metadata.get("msa_config"))
    latest = store.conn.execute(
        """
        SELECT member_sequences_json
        FROM validation_msa_jobs
        WHERE msa_job_id = ?
        """,
        (claim.msa_job_id,),
    ).fetchone()
    latest_members = _json_list(latest["member_sequences_json"] if latest else None)
    members = tuple(
        normalize_sequence(seq)
        for seq in (latest_members or list(claim.member_sequences))
        if seq
    )
    if not members:
        raise ValueError("VHH MSA job has no member sequences")
    representative = normalize_sequence(claim.representative_sequence or members[0])
    analysis = claim.metadata.get("vhh_analysis")
    if isinstance(analysis, dict):
        template_hash = str(analysis.get("lengths_only_template_key_hash") or "")
    else:
        template_hash = str(claim.metadata.get("template_key_hash") or "")
    if not template_hash:
        analysis = analyze_vhh_sequence(representative)
        template_hash = str(analysis["lengths_only_template_key_hash"])

    representative_cache = (
        vhh_template_group_cache_dir(
            root,
            template_key_hash=template_hash,
            config=config,
        )
        / "representative"
    )
    cached = read_cached_msa_pair(
        representative_cache,
        sequence=representative,
        config=config,
    )
    if cached is not None:
        _log(log, f"VHH template {template_hash}: already cached; materializing members")
    else:
        if not config.server_url:
            raise ValueError(
                f"VHH template {template_hash} has no cached representative and no MSA server"
            )
        _log(
            log,
            f"VHH template {template_hash}: missing; acquiring MMseqs rate slot",
        )
        _acquire_rate_slot(
            store,
            max_requests_per_minute=max_requests_per_minute,
            log=log,
        )
        _log(log, f"VHH template {template_hash}: submitting representative to MMseqs")
    pairs = resolve_binder_msa_pairs(
        root,
        binders=tuple((seq, "vhh") for seq in members),
        config=config,
    )
    cache_paths = {
        seq: _cache_paths(pair.cache_dir)
        for seq, pair in zip(members, pairs)
        if pair is not None and pair.cache_dir is not None
    }
    if len(cache_paths) != len(members):
        raise RuntimeError("VHH MSA resolver did not cache every member sequence")
    store.complete_msa_job(
        msa_job_id=claim.msa_job_id,
        cache_paths={"members": cache_paths},
        metadata={"decision": "resolved", "template_key_hash": template_hash},
    )
    return "completed"


def _run_miniprotein_msa_job(
    root: Path,
    *,
    store: CampaignStore,
    claim: MsaJobClaim,
    log: Callable[[str], None] | None,
) -> str:
    config = _msa_config_from_payload(root, claim.metadata.get("msa_config"))
    sequence = normalize_sequence(
        claim.metadata.get("binder_sequence") or claim.representative_sequence or ""
    )
    scaffold = str(claim.metadata.get("binder_scaffold") or "miniprotein")
    cache_dir = binder_msa_cache_dir(
        root,
        sequence=sequence,
        scaffold=scaffold,
        config=config,
    )
    cached = read_cached_msa_pair(cache_dir, sequence=sequence, config=config)
    if cached is not None:
        _log(log, "miniprotein binder: single-sequence MSA cache hit")
        store.complete_msa_job(
            msa_job_id=claim.msa_job_id,
            cache_paths=_cache_paths(cache_dir),
            metadata={"decision": "cache_hit"},
        )
        return "completed"
    _log(log, "miniprotein binder: no MSA server query required; writing single-sequence MSA")
    pair = resolve_binder_msa_pair(
        root,
        binder_sequence=sequence,
        binder_scaffold=scaffold,
        config=config,
    )
    if pair is None or pair.cache_dir is None:
        raise RuntimeError("miniprotein MSA resolver did not produce a cached MSA")
    store.complete_msa_job(
        msa_job_id=claim.msa_job_id,
        cache_paths=_cache_paths(pair.cache_dir),
        metadata={"decision": "single_sequence"},
    )
    return "completed"


def _acquire_rate_slot(
    store: CampaignStore,
    *,
    max_requests_per_minute: float,
    log: Callable[[str], None] | None,
) -> None:
    interval = 60.0 / float(max_requests_per_minute)
    while True:
        wait_seconds = store.try_acquire_msa_rate_slot(
            name=MSA_RATE_LIMIT_NAME,
            min_interval_seconds=interval,
        )
        if wait_seconds <= 0:
            return
        _log(log, f"MSA server throttle: waiting {wait_seconds:.1f}s")
        time.sleep(wait_seconds)


def _skip(
    candidate_id: str,
    reason: str,
    log: Callable[[str], None] | None,
) -> MsaPrefetchEnqueueResult:
    _log(log, f"candidate {candidate_id}: {reason}")
    return MsaPrefetchEnqueueResult(
        queued_jobs=0,
        existing_jobs=0,
        skipped=True,
        decisions=(reason,),
    )


def _validation_config(resolved_config: dict[str, Any]) -> dict[str, Any]:
    value = resolved_config.get("validation")
    return value if isinstance(value, dict) else {}


def _resolved_config(store: CampaignStore) -> dict[str, Any]:
    row = store.conn.execute(
        "SELECT resolved_config_json FROM campaign WHERE id = 1"
    ).fetchone()
    if row is None:
        return {}
    return _json_dict(row["resolved_config_json"])


def _msa_config_from_validation(root: Path, validation: dict[str, Any]) -> ProtenixMsaConfig:
    msa = validation.get("msa") if isinstance(validation.get("msa"), dict) else {}
    protenix = (
        validation.get("protenix")
        if isinstance(validation.get("protenix"), dict)
        else {}
    )
    runtime = {**validation, **msa, **protenix}
    server_url = _first_str(runtime, "msa_server", "msa_server_url", "server_url")
    target_msa_dir = _optional_path(runtime.get("target_msa_dir"), base=root)
    target_msa_map_csv = _optional_path(runtime.get("target_msa_map_csv"), base=root)
    target_mode = _first_str(runtime, "target_msa_mode", "target")
    if target_mode is None:
        if server_url:
            target_mode = "server"
        elif target_msa_dir is not None or target_msa_map_csv is not None:
            target_mode = "provided"
        else:
            target_mode = "none"
    binder_mode = _first_str(runtime, "binder_msa_mode", "binder_msa", "binder") or "auto"
    pairing_strategy = _first_str(runtime, "msa_pairing_strategy", "pairing_strategy") or "greedy"
    return ProtenixMsaConfig(
        target_mode=target_mode,  # type: ignore[arg-type]
        binder_mode=binder_mode,  # type: ignore[arg-type]
        target_msa_dir=target_msa_dir,
        target_msa_map_csv=target_msa_map_csv,
        server_url=server_url,
        cache_root=_optional_path(runtime.get("msa_cache_root"), base=root),
        pairing_strategy=pairing_strategy,  # type: ignore[arg-type]
    )


def _msa_config_payload(config: ProtenixMsaConfig, *, root: Path) -> dict[str, Any]:
    return {
        "target_mode": config.target_mode,
        "binder_mode": config.binder_mode,
        "target_msa_dir": _rel_or_abs(config.target_msa_dir, root=root),
        "target_msa_map_csv": _rel_or_abs(config.target_msa_map_csv, root=root),
        "server_url": config.server_url,
        "cache_root": _rel_or_abs(config.cache_root, root=root),
        "pairing_strategy": config.pairing_strategy,
    }


def _msa_config_from_payload(root: Path, payload: Any) -> ProtenixMsaConfig:
    data = payload if isinstance(payload, dict) else {}
    return ProtenixMsaConfig(
        target_mode=str(data.get("target_mode") or "none"),  # type: ignore[arg-type]
        binder_mode=str(data.get("binder_mode") or "auto"),  # type: ignore[arg-type]
        target_msa_dir=_optional_path(data.get("target_msa_dir"), base=root),
        target_msa_map_csv=_optional_path(data.get("target_msa_map_csv"), base=root),
        server_url=data.get("server_url"),
        cache_root=_optional_path(data.get("cache_root"), base=root),
        pairing_strategy=str(data.get("pairing_strategy") or "greedy"),  # type: ignore[arg-type]
    )


def _target_sequences_from_config(
    root: Path,
    config: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    target = config.get("target")
    if not isinstance(target, dict):
        return (), ()
    direct = target.get("sequence")
    if isinstance(direct, str) and direct.strip():
        return (normalize_sequence(direct),), ("B",)
    sequences = target.get("sequences")
    chains = target.get("chains")
    if isinstance(sequences, dict) and isinstance(chains, list):
        out: list[str] = []
        labels: list[str] = []
        for chain in chains:
            seq = sequences.get(chain)
            if isinstance(seq, str) and seq.strip():
                out.append(normalize_sequence(seq))
                labels.append(str(chain))
        if out:
            return tuple(out), tuple(labels)
    summary_path = root / "target" / "chain_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            return (), ()
        chains_summary = summary.get("chains") if isinstance(summary, dict) else None
        if isinstance(chains_summary, list):
            out = []
            labels = []
            for chain in chains_summary:
                if not isinstance(chain, dict):
                    continue
                sequence = chain.get("sequence")
                if not isinstance(sequence, str) or not sequence.strip():
                    continue
                label = (
                    chain.get("canonical_chain_id")
                    or chain.get("auth_asym_id")
                    or chain.get("label_asym_id")
                    or f"target{len(labels)}"
                )
                out.append(normalize_sequence(sequence))
                labels.append(str(label))
            if out:
                return tuple(out), tuple(labels)
    return (), ()


def _target_name(config: dict[str, Any]) -> str | None:
    target = config.get("target")
    if isinstance(target, dict) and target.get("name"):
        return str(target["name"])
    return None


def _prefetch_min_esm_iptm(validation: dict[str, Any]) -> float | None:
    msa = validation.get("msa") if isinstance(validation.get("msa"), dict) else {}
    raw = (
        msa.get("prefetch_min_esm_iptm")
        if "prefetch_min_esm_iptm" in msa
        else validation.get("min_esm_iptm")
    )
    return _float_or_none(raw)


def _has_hotspot_contact(metrics: dict[str, Any]) -> bool:
    for key in ("hotspot_contact_pass", "hotspot_pass", "esm_hotspot_pass"):
        if key in metrics:
            return bool(metrics[key])
    value = _float_or_none(metrics.get("hotspot_satisfaction"))
    return value is not None and value > 0


def _template_mode_from_validation(validation: dict[str, Any]) -> str:
    runtime = _validation_runtime(validation)
    value = runtime.get("use_template")
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"auto", "true", "false"}:
        return text
    if text in {"1", "yes", "on"}:
        return "true"
    if text in {"0", "no", "off"}:
        return "false"
    return "auto"


def _explicit_use_msa(validation: dict[str, Any]) -> bool:
    runtime = _validation_runtime(validation)
    return _coerce_bool(runtime.get("use_msa")) if "use_msa" in runtime else False


def _validation_runtime(validation: dict[str, Any]) -> dict[str, Any]:
    msa = validation.get("msa") if isinstance(validation.get("msa"), dict) else {}
    protenix = (
        validation.get("protenix")
        if isinstance(validation.get("protenix"), dict)
        else {}
    )
    return {**validation, **msa, **protenix}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _requests_unsupported_scfv_binder_msa(
    *,
    explicit_use_msa: bool,
    binder_mode: str,
) -> bool:
    if binder_mode == "none":
        return False
    return bool(explicit_use_msa or binder_mode == "single_sequence")


def _target_template_available(
    root: Path,
    *,
    target_sequences: tuple[str, ...],
) -> bool:
    if not (root / "target" / "normalized_target.cif").exists():
        return False
    summary_path = root / "target" / "chain_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    chains = summary.get("chains") if isinstance(summary, dict) else None
    if not isinstance(chains, list) or len(chains) != len(target_sequences):
        return False
    for chain, sequence in zip(chains, target_sequences):
        if not isinstance(chain, dict):
            return False
        template_sequence = chain.get("sequence")
        if isinstance(template_sequence, str) and template_sequence.strip():
            if normalize_sequence(template_sequence) != normalize_sequence(sequence):
                return False
    return True


def _framework_template_available(
    design_metrics: dict[str, Any],
    *,
    binder_scaffold: str,
    template_mode: str,
) -> bool:
    if template_mode == "false":
        return False
    if str(design_metrics.get("framework_source") or "").strip().lower() != "builtin":
        return False
    framework = str(
        design_metrics.get("framework")
        or design_metrics.get("framework_name")
        or ""
    ).strip()
    if not framework:
        return False
    try:
        if binder_scaffold in {"scfv", "sc_fv"}:
            return get_scfv_framework_template_cif(framework) is not None
        if binder_scaffold == "vhh":
            return get_vhh_framework_template_cif(framework) is not None
    except KeyError:
        return False
    return False


def _cache_paths(cache_dir: Path) -> dict[str, str]:
    return {
        "cache_dir": str(cache_dir),
        "pairing_path": str(cache_dir / "pairing.a3m"),
        "non_pairing_path": str(cache_dir / "non_pairing.a3m"),
        "metadata_path": str(cache_dir / "metadata.json"),
    }


def _first_str(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _optional_path(value: Any, *, base: Path) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _rel_or_abs(path: Path | None, *, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_dict(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: str | None) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item)]


def _safe_identifier(value: str, *, max_len: int) -> str:
    text = "".join(char.lower() if char.isalnum() else "_" for char in value)
    while "__" in text:
        text = text.replace("__", "_")
    return (text.strip("_") or "job")[:max_len].strip("_") or "job"


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
    else:
        print(message, file=sys.stderr, flush=True)


def _pid() -> int:
    try:
        import os

        return os.getpid()
    except Exception:
        return 0
