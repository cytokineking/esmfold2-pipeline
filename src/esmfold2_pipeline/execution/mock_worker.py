from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket

from esmfold2_pipeline.artifact_layout import artifact_stem, structure_relpath
from esmfold2_pipeline.artifacts import write_bytes_atomic
from esmfold2_pipeline.db import CampaignStore, initialize_database
from esmfold2_pipeline.planning import candidate_id, shard_id
from esmfold2_pipeline.validation import enqueue_msa_prefetch_for_candidate

MOCK_CRITIC_NAME = "MockCritic"
MOCK_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
MOCK_PDB = (
    b"HEADER    MOCK ESMFOLD2 COMPLEX\n"
    b"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N\n"
    b"ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 90.00           C\n"
    b"TER\n"
    b"END\n"
)
MOCK_METRICS = {
    "iptm": 0.72,
    "ptm": 0.61,
    "plddt": 89.4,
    "distogram_iptm_proxy": 0.68,
}


@dataclass(frozen=True)
class MockWorkerResult:
    shard_id: str
    candidate_id: str
    critic_name: str
    sequence_path: str | None
    structure_path: str
    metrics: dict[str, float]


def plan_one_mock_shard(campaign_dir: str | Path) -> str:
    """Initialize a campaign DB and insert the deterministic one-shard plan."""

    root = Path(campaign_dir)
    conn = initialize_database(
        root / "campaign.sqlite",
        config_hash="mock-vertical-slice",
        resolved_config={
            "target": {"sequence": "mock"},
            "binder": {"scaffold": "miniprotein", "length": {"min": 60, "max": 200}},
            "campaign": {"num_designs": 1, "critics": [MOCK_CRITIC_NAME]},
        },
        software_versions={"esmfold2_pipeline": "0.1.0"},
    )
    store = CampaignStore(conn)
    sid = shard_id(0)
    store.create_shard(
        shard_id=sid,
        seed=0,
        batch_index=0,
        target_key="target:sequence:mock",
        binder_key="binder:minibinder",
        critic_set=[MOCK_CRITIC_NAME],
    )
    conn.close()
    return sid


def run_one_mock_shard(
    campaign_dir: str | Path,
    *,
    worker_id: str = "mock-worker-0",
    gpu_id: str | None = None,
) -> MockWorkerResult | None:
    """Claim one pending shard, write mock artifacts, and commit paths/metrics."""

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

    cid = candidate_id(claim.batch_index, 0)
    stem = artifact_stem(
        batch_index=claim.batch_index,
        seed=claim.seed,
        candidate_index=0,
    )
    structure_path = structure_relpath(stem)
    store.record_completed_candidate(
        candidate_id=cid,
        shard_id=claim.shard_id,
        candidate_index=0,
        designed_sequence=MOCK_SEQUENCE,
        sequence_path=None,
        design_metrics={"mock": True, "seed": claim.seed},
    )

    write_bytes_atomic(root / structure_path, MOCK_PDB)
    store.record_completed_critic(
        candidate_id=cid,
        critic_name=MOCK_CRITIC_NAME,
        structure_path=structure_path.as_posix(),
        metrics=MOCK_METRICS,
        runtime_seconds=0.0,
    )
    enqueue_msa_prefetch_for_candidate(
        root,
        store=store,
        candidate_id=cid,
        critic_metrics=MOCK_METRICS,
        log=lambda _message: None,
    )
    store.complete_shard(shard_id=claim.shard_id, attempt_id=claim.attempt_id)
    conn.close()

    return MockWorkerResult(
        shard_id=claim.shard_id,
        candidate_id=cid,
        critic_name=MOCK_CRITIC_NAME,
        sequence_path=None,
        structure_path=structure_path.as_posix(),
        metrics=dict(MOCK_METRICS),
    )
