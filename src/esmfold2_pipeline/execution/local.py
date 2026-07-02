from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import os
from pathlib import Path
import socket
import threading
from types import TracebackType

from esmfold2_pipeline.config import load_campaign_config
from esmfold2_pipeline.db import CampaignStore, connect_database, initialize_database
from esmfold2_pipeline.esm_adapter import run_binder_design_artifact
from esmfold2_pipeline.planning import semantic_candidate_id
from esmfold2_pipeline.structure import parse_structure_target
from esmfold2_pipeline.execution.recovery import (
    resolve_stale_after_seconds,
    stale_before_timestamp,
)
from esmfold2_pipeline.validation import enqueue_msa_prefetch_for_candidate


@dataclass(frozen=True)
class RunCampaignResult:
    completed_shards: int
    skipped_no_pending: bool
    recovered_stale_shards: int = 0


def run_campaign(
    campaign_dir: str | Path,
    *,
    esm_repo: str | Path | None = None,
    worker_id: str = "local-worker-0",
    gpu_id: str | None = None,
    max_shards: int | None = None,
    heartbeat_interval_seconds: float = 30.0,
    stale_after_seconds: float | None = None,
    disable_hf_xet: bool = True,
) -> RunCampaignResult:
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be positive")
    resolved_stale_after_seconds = resolve_stale_after_seconds(
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        stale_after_seconds=stale_after_seconds,
    )

    root = Path(campaign_dir)
    campaign_config = load_campaign_config(root / "resolved_config.yaml")
    structure_target = (
        parse_structure_target(campaign_config.target_structure)
        if campaign_config.target_structure is not None
        else None
    )
    target_name = campaign_config.target_name
    steps = campaign_config.steps
    inversion_model_name = campaign_config.inversion_model_name
    critic_name = campaign_config.critic_name

    conn = initialize_database(root / "campaign.sqlite")
    store = CampaignStore(conn)
    recovered_stale = store.recover_stale_shards(
        stale_before=stale_before_timestamp(resolved_stale_after_seconds),
        error_message=(
            "running shard heartbeat exceeded "
            f"{resolved_stale_after_seconds:g}s; recovering for resume"
        ),
    )
    completed = 0
    try:
        while max_shards is None or completed < max_shards:
            claim = store.claim_next_pending_shard(
                worker_id=worker_id,
                hostname=socket.gethostname(),
                pid=os.getpid(),
                gpu_id=gpu_id,
            )
            if claim is None:
                break
            try:
                framework = campaign_config.binder_framework_for_design_index(
                    claim.batch_index
                )
                binder_name = campaign_config.binder_name_for_design_index(
                    claim.batch_index
                )
                cid = semantic_candidate_id(
                    target_name=campaign_config.target_name,
                    binder_scaffold=campaign_config.binder.scaffold,
                    seed=claim.seed,
                    candidate_index=0,
                )
                with _ShardHeartbeat(
                    db_path=root / "campaign.sqlite",
                    shard_id=claim.shard_id,
                    attempt_id=claim.attempt_id,
                    interval_seconds=heartbeat_interval_seconds,
                ):
                    artifact = run_binder_design_artifact(
                        campaign_dir=root,
                        candidate_id=cid,
                        shard_id=claim.shard_id,
                        seed=claim.seed,
                        esm_repo=esm_repo,
                        gpu_id=gpu_id,
                        steps=steps,
                        target_name=target_name,
                        target_sequence=campaign_config.target_sequence,
                        binder_name=binder_name,
                        binder_scaffold=campaign_config.binder.scaffold,
                        binder_framework_name=(
                            framework.name if framework is not None else None
                        ),
                        binder_framework_source=(
                            framework.source if framework is not None else None
                        ),
                        binder_framework_template=(
                            framework.template if framework is not None else None
                        ),
                        binder_framework_cdr_lengths=(
                            framework.cdr_lengths if framework is not None else None
                        ),
                        binder_framework_sequence=(
                            framework.sequence if framework is not None else None
                        ),
                        binder_framework_cdr_indices=(
                            framework.cdr_indices if framework is not None else None
                        ),
                        binder_length_range=campaign_config.binder.length_range,
                        critic_name=critic_name,
                        inversion_model_name=inversion_model_name,
                        structure_target=structure_target,
                        target_structure_indexing=(
                            campaign_config.target_structure.structure_indexing
                            if campaign_config.target_structure is not None
                            else "auto"
                        ),
                        conditioning_mode=(
                            campaign_config.target_structure.conditioning_mode
                            if campaign_config.target_structure is not None
                            else "none"
                        ),
                        conditioning_assembly=(
                            campaign_config.target_structure.conditioning_assembly
                            if campaign_config.target_structure is not None
                            else False
                        ),
                        conditioning_chain_pairs=(
                            campaign_config.target_structure.conditioning_chain_pairs
                            if campaign_config.target_structure is not None
                            else None
                        ),
                        hotspot_contact_weight=campaign_config.hotspot_contact_weight,
                        hotspot_distogram_contact_cutoff_angstrom=(
                            campaign_config.hotspot_distogram_contact_cutoff_angstrom
                        ),
                        hotspot_critic_contact_cutoff_angstrom=(
                            campaign_config.hotspot_critic_contact_cutoff_angstrom
                        ),
                        hotspot_num_contacts=campaign_config.hotspot_num_contacts,
                        hotspot_contact_probability_target=(
                            campaign_config.hotspot_contact_probability_target
                        ),
                        hotspot_loss_mode=campaign_config.hotspot_loss_mode,
                        binder_target_contact_mode=(
                            campaign_config.binder_target_contact_mode
                        ),
                        mosaic_cdr_contact_weight=(
                            campaign_config.mosaic_cdr_contact_weight
                        ),
                        mosaic_cdr_contact_cutoff_angstrom=(
                            campaign_config.mosaic_cdr_contact_cutoff_angstrom
                        ),
                        mosaic_cdr_num_target_contacts=(
                            campaign_config.mosaic_cdr_num_target_contacts
                        ),
                        mosaic_framework_contact_penalty_weight=(
                            campaign_config.mosaic_framework_contact_penalty_weight
                        ),
                        mosaic_framework_contact_penalty_cutoff_angstrom=(
                            campaign_config.mosaic_framework_contact_penalty_cutoff_angstrom
                        ),
                        mosaic_framework_contact_probability_threshold=(
                            campaign_config.mosaic_framework_contact_probability_threshold
                        ),
                        mosaic_framework_contact_penalty_scope=(
                            campaign_config.mosaic_framework_contact_penalty_scope
                        ),
                        target_geometry_drift=campaign_config.target_geometry_drift,
                        disable_hf_xet=disable_hf_xet,
                        artifact_stem=cid,
                    )
                store.record_completed_candidate(
                    candidate_id=artifact.candidate_id,
                    shard_id=claim.shard_id,
                    candidate_index=0,
                    designed_sequence=artifact.designed_sequence,
                    binder_chain_id=artifact.design_metrics.get("binder_chain_id"),
                    sequence_path=artifact.sequence_path,
                    design_metrics=artifact.design_metrics,
                )
                store.record_completed_critic(
                    candidate_id=artifact.candidate_id,
                    critic_name=artifact.critic_name,
                    structure_path=artifact.structure_path,
                    metrics=artifact.critic_metrics,
                )
                enqueue_msa_prefetch_for_candidate(
                    root,
                    store=store,
                    candidate_id=artifact.candidate_id,
                    critic_metrics=artifact.critic_metrics,
                    log=lambda message: print(
                        f"[esmfold2-pipeline] {message}",
                        flush=True,
                    ),
                )
                store.complete_shard(shard_id=claim.shard_id, attempt_id=claim.attempt_id)
                completed += 1
            except Exception as exc:
                store.fail_shard(
                    shard_id=claim.shard_id,
                    attempt_id=claim.attempt_id,
                    error_message=str(exc),
                )
                raise
    finally:
        conn.close()

    return RunCampaignResult(
        completed_shards=completed,
        skipped_no_pending=completed == 0,
        recovered_stale_shards=recovered_stale,
    )


class _ShardHeartbeat(AbstractContextManager["_ShardHeartbeat"]):
    def __init__(
        self,
        *,
        db_path: Path,
        shard_id: str,
        attempt_id: int,
        interval_seconds: float,
    ):
        self.db_path = db_path
        self.shard_id = shard_id
        self.attempt_id = attempt_id
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"heartbeat-{shard_id}-{attempt_id}",
            daemon=True,
        )
        self._error: BaseException | None = None

    def __enter__(self) -> _ShardHeartbeat:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        if exc_type is None and self._error is not None:
            raise RuntimeError("shard heartbeat failed") from self._error
        return None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            conn = connect_database(self.db_path)
            try:
                CampaignStore(conn).heartbeat_shard(
                    shard_id=self.shard_id,
                    attempt_id=self.attempt_id,
                )
            except BaseException as exc:
                self._error = exc
                self._stop.set()
            finally:
                conn.close()
