from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket

from esmfold2_pipeline.artifact_layout import artifact_stem
from esmfold2_pipeline.config import (
    DEFAULT_ESMFOLD2_CRITIC_MODEL,
    DEFAULT_ESMFOLD2_INVERSION_MODEL,
    DEFAULT_MINIPROTEIN_LENGTH_RANGE,
)
from esmfold2_pipeline.db import CampaignStore, initialize_database
from esmfold2_pipeline.esm_adapter import run_binder_design_artifact
from esmfold2_pipeline.planning import candidate_id, shard_id

DEFAULT_TARGET_NAME = "ctla4"
DEFAULT_BINDER_NAME = "minibinder"
DEFAULT_BINDER_SCAFFOLD = "miniprotein"
DEFAULT_CRITIC_NAME = DEFAULT_ESMFOLD2_CRITIC_MODEL
DEFAULT_INVERSION_MODEL_NAME = DEFAULT_ESMFOLD2_INVERSION_MODEL
DEFAULT_GPU_SMOKE_STEPS = 2


@dataclass(frozen=True)
class GPUSmokeResult:
    shard_id: str
    candidate_id: str
    critic_name: str
    sequence_path: str | None
    structure_path: str
    metrics: dict[str, float | int | str | None]


def plan_one_gpu_smoke_shard(
    campaign_dir: str | Path,
    *,
    target_name: str = DEFAULT_TARGET_NAME,
    binder_name: str = DEFAULT_BINDER_NAME,
    inversion_model_name: str = DEFAULT_INVERSION_MODEL_NAME,
    critic_name: str = DEFAULT_CRITIC_NAME,
) -> str:
    """Initialize a one-shard campaign for the real ESMFold2 GPU smoke path."""

    root = Path(campaign_dir)
    conn = initialize_database(
        root / "campaign.sqlite",
        config_hash="gpu-smoke",
        resolved_config={
            "target": {"name": target_name},
            "binder": {
                "scaffold": DEFAULT_BINDER_SCAFFOLD,
                "length": {
                    "min": DEFAULT_MINIPROTEIN_LENGTH_RANGE[0],
                    "max": DEFAULT_MINIPROTEIN_LENGTH_RANGE[1],
                },
            },
            "campaign": {
                "num_designs": 1,
                "inversion_model": inversion_model_name,
                "critics": [critic_name],
            },
        },
        software_versions={"esmfold2_pipeline": "0.1.0"},
    )
    store = CampaignStore(conn)
    sid = shard_id(0)
    store.create_shard(
        shard_id=sid,
        seed=0,
        batch_index=0,
        target_key=f"target:name:{target_name}",
        binder_key=(
            f"binder:{DEFAULT_BINDER_SCAFFOLD}:"
            f"length={DEFAULT_MINIPROTEIN_LENGTH_RANGE[0]}-"
            f"{DEFAULT_MINIPROTEIN_LENGTH_RANGE[1]}"
        ),
        critic_set=[critic_name],
    )
    conn.close()
    return sid


def run_one_gpu_smoke_shard(
    campaign_dir: str | Path,
    *,
    esm_repo: str | Path | None = None,
    worker_id: str = "gpu-smoke-worker-0",
    gpu_id: str | None = None,
    steps: int = DEFAULT_GPU_SMOKE_STEPS,
    target_name: str = DEFAULT_TARGET_NAME,
    binder_name: str = DEFAULT_BINDER_NAME,
    inversion_model_name: str = DEFAULT_INVERSION_MODEL_NAME,
    critic_name: str = DEFAULT_CRITIC_NAME,
    disable_hf_xet: bool = True,
) -> GPUSmokeResult | None:
    """Run one real ESMFold2 design shard and return only scalar/path data."""

    root = Path(campaign_dir)
    conn = initialize_database(root / "campaign.sqlite")
    store = CampaignStore(conn)
    claim = store.claim_next_pending_shard(
        worker_id=worker_id,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        gpu_id=gpu_id,
    )
    if claim is None:
        conn.close()
        return None

    try:
        result = _run_claimed_gpu_smoke(
            root=root,
            claim=claim,
            store=store,
            esm_repo=esm_repo,
            gpu_id=gpu_id,
            steps=steps,
            target_name=target_name,
            binder_name=binder_name,
            inversion_model_name=inversion_model_name,
            critic_name=critic_name,
            disable_hf_xet=disable_hf_xet,
        )
        store.complete_shard(shard_id=claim.shard_id, attempt_id=claim.attempt_id)
        return result
    except Exception as exc:
        store.fail_shard(
            shard_id=claim.shard_id,
            attempt_id=claim.attempt_id,
            error_message=str(exc),
        )
        raise
    finally:
        conn.close()


def _run_claimed_gpu_smoke(
    *,
    root: Path,
    claim,
    store: CampaignStore,
    esm_repo: str | Path | None,
    gpu_id: str | None,
    steps: int,
    target_name: str,
    binder_name: str,
    inversion_model_name: str,
    critic_name: str,
    disable_hf_xet: bool,
) -> GPUSmokeResult:
    cid = candidate_id(claim.batch_index, 0)
    stem = artifact_stem(
        batch_index=claim.batch_index,
        seed=claim.seed,
        candidate_index=0,
    )
    artifact = run_binder_design_artifact(
        campaign_dir=root,
        candidate_id=cid,
        shard_id=claim.shard_id,
        seed=claim.seed,
        esm_repo=esm_repo,
        gpu_id=gpu_id,
        steps=steps,
        target_name=target_name,
        binder_name=binder_name,
        critic_name=critic_name,
        inversion_model_name=inversion_model_name,
        disable_hf_xet=disable_hf_xet,
        artifact_stem=stem,
    )
    store.record_completed_candidate(
        candidate_id=artifact.candidate_id,
        shard_id=claim.shard_id,
        candidate_index=0,
        designed_sequence=artifact.designed_sequence,
        sequence_path=artifact.sequence_path,
        binder_chain_id=artifact.design_metrics.get("binder_chain_id"),
        design_metrics=artifact.design_metrics,
    )
    store.record_completed_critic(
        candidate_id=artifact.candidate_id,
        critic_name=artifact.critic_name,
        structure_path=artifact.structure_path,
        metrics=artifact.critic_metrics,
    )

    return GPUSmokeResult(
        shard_id=claim.shard_id,
        candidate_id=artifact.candidate_id,
        critic_name=artifact.critic_name,
        sequence_path=artifact.sequence_path,
        structure_path=artifact.structure_path,
        metrics=artifact.critic_metrics,
    )
