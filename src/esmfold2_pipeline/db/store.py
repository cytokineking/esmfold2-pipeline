from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sqlite3
from importlib import resources
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 5
DEFAULT_BUSY_TIMEOUT_MS = 30_000


@dataclass(frozen=True)
class ShardClaim:
    shard_id: str
    attempt_id: int
    seed: int
    batch_index: int
    target_key: str
    binder_key: str
    critic_set: list[str]


@dataclass(frozen=True)
class ValidationClaim:
    validation_id: str
    attempt_id: int
    candidate_id: str
    model_name: str
    validation_config_hash: str
    selection_rank: int | None


@dataclass(frozen=True)
class MsaJobClaim:
    msa_job_id: str
    scope: str
    cache_key: str
    msa_context_hash: str
    attempt_count: int
    representative_sequence: str | None
    member_sequences: tuple[str, ...]
    metadata: dict[str, Any]


class CampaignStore:
    """Small SQLite store for the milestone-0 worker contract."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    @classmethod
    def open(cls, db_path: str | Path) -> CampaignStore:
        return cls(connect_database(db_path))

    def create_shard(
        self,
        *,
        shard_id: str,
        seed: int,
        batch_index: int,
        target_key: str,
        binder_key: str,
        critic_set: list[str],
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO shards (
                shard_id,
                seed,
                batch_index,
                target_key,
                binder_key,
                critic_set_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(shard_id) DO NOTHING
            """,
            (
                shard_id,
                seed,
                batch_index,
                target_key,
                binder_key,
                _json_text_list(critic_set),
            ),
        )
        self.conn.commit()

    def claim_next_pending_shard(
        self,
        *,
        worker_id: str,
        hostname: str | None = None,
        pid: int | None = None,
        gpu_id: str | None = None,
    ) -> ShardClaim | None:
        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                """
                SELECT *
                FROM shards
                WHERE status = 'pending'
                  AND attempt_count < max_attempts
                ORDER BY shard_id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                self.conn.commit()
                return None

            self.conn.execute(
                """
                UPDATE shards
                SET status = 'running',
                    claim_worker_id = ?,
                    claim_hostname = ?,
                    claim_pid = ?,
                    claim_gpu_id = ?,
                    claimed_at = ?,
                    heartbeat_at = ?,
                    started_at = COALESCE(started_at, ?),
                    attempt_count = attempt_count + 1,
                    error_message = NULL
                WHERE shard_id = ?
                  AND status = 'pending'
                """,
                (
                    worker_id,
                    hostname,
                    pid,
                    gpu_id,
                    now,
                    now,
                    now,
                    row["shard_id"],
                ),
            )
            cursor = self.conn.execute(
                """
                INSERT INTO attempts (
                    shard_id,
                    stage,
                    status,
                    worker_id,
                    hostname,
                    pid,
                    gpu_id,
                    started_at
                )
                VALUES (?, 'shard', 'running', ?, ?, ?, ?, ?)
                """,
                (row["shard_id"], worker_id, hostname, pid, gpu_id, now),
            )
            self.conn.commit()
            return ShardClaim(
                shard_id=row["shard_id"],
                attempt_id=int(cursor.lastrowid),
                seed=int(row["seed"]),
                batch_index=int(row["batch_index"]),
                target_key=row["target_key"],
                binder_key=row["binder_key"],
                critic_set=json.loads(row["critic_set_json"]),
            )
        except Exception:
            self.conn.rollback()
            raise

    def heartbeat_shard(self, *, shard_id: str, attempt_id: int) -> None:
        now = _utc_now()
        self.conn.execute(
            """
            UPDATE shards
            SET heartbeat_at = ?
            WHERE shard_id = ?
              AND status = 'running'
            """,
            (now, shard_id),
        )
        self.conn.execute(
            """
            UPDATE attempts
            SET ended_at = NULL
            WHERE attempt_id = ?
              AND status = 'running'
            """,
            (attempt_id,),
        )
        self.conn.commit()

    def recover_stale_shards(
        self,
        *,
        stale_before: str,
        error_message: str = "stale shard claim",
    ) -> int:
        """Release stale running shards back to pending when retry budget remains."""

        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                """
                SELECT *
                FROM shards
                WHERE status = 'running'
                  AND heartbeat_at IS NOT NULL
                  AND heartbeat_at < ?
                ORDER BY shard_id
                """,
                (stale_before,),
            ).fetchall()
            for row in rows:
                next_status = (
                    "pending"
                    if int(row["attempt_count"]) < int(row["max_attempts"])
                    else "failed"
                )
                self.conn.execute(
                    """
                    UPDATE attempts
                    SET status = 'stale',
                        ended_at = ?,
                        error_message = COALESCE(error_message, ?)
                    WHERE shard_id = ?
                      AND stage = 'shard'
                      AND status = 'running'
                    """,
                    (now, error_message, row["shard_id"]),
                )
                self.conn.execute(
                    """
                    UPDATE shards
                    SET status = ?,
                        claim_worker_id = NULL,
                        claim_hostname = NULL,
                        claim_pid = NULL,
                        claim_gpu_id = NULL,
                        claimed_at = NULL,
                        heartbeat_at = NULL,
                        completed_at = CASE WHEN ? = 'failed' THEN ? ELSE completed_at END,
                        error_message = CASE WHEN ? = 'failed' THEN ? ELSE NULL END
                    WHERE shard_id = ?
                    """,
                    (
                        next_status,
                        next_status,
                        now,
                        next_status,
                        error_message,
                        row["shard_id"],
                    ),
                )
            self.conn.commit()
            return len(rows)
        except Exception:
            self.conn.rollback()
            raise

    def recover_failed_worker_shards(
        self,
        *,
        worker_id: str,
        pid: int | None = None,
        exit_code: int | None = None,
        error_message: str = "worker process exited before completing shard",
    ) -> int:
        """Release running shards owned by a failed worker process."""

        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            query = """
                SELECT *
                FROM shards
                WHERE status = 'running'
                  AND claim_worker_id = ?
            """
            params: list[Any] = [worker_id]
            if pid is not None:
                query += " AND claim_pid = ?"
                params.append(pid)
            query += " ORDER BY shard_id"

            rows = self.conn.execute(query, tuple(params)).fetchall()
            for row in rows:
                next_status = (
                    "pending"
                    if int(row["attempt_count"]) < int(row["max_attempts"])
                    else "failed"
                )
                attempt_params: list[Any] = [
                    now,
                    exit_code,
                    error_message,
                    row["shard_id"],
                    worker_id,
                ]
                attempt_filter = ""
                if pid is not None:
                    attempt_filter = "AND pid = ?"
                    attempt_params.append(pid)

                self.conn.execute(
                    f"""
                    UPDATE attempts
                    SET status = 'failed',
                        ended_at = ?,
                        exit_code = ?,
                        error_message = ?
                    WHERE shard_id = ?
                      AND worker_id = ?
                      {attempt_filter}
                      AND stage = 'shard'
                      AND status = 'running'
                    """,
                    tuple(attempt_params),
                )
                self.conn.execute(
                    """
                    UPDATE shards
                    SET status = ?,
                        claim_worker_id = NULL,
                        claim_hostname = NULL,
                        claim_pid = NULL,
                        claim_gpu_id = NULL,
                        claimed_at = NULL,
                        heartbeat_at = NULL,
                        completed_at = CASE WHEN ? = 'failed' THEN ? ELSE completed_at END,
                        error_message = CASE WHEN ? = 'failed' THEN ? ELSE NULL END
                    WHERE shard_id = ?
                    """,
                    (
                        next_status,
                        next_status,
                        now,
                        next_status,
                        error_message,
                        row["shard_id"],
                    ),
                )
            self.conn.commit()
            return len(rows)
        except Exception:
            self.conn.rollback()
            raise

    def fail_shard(
        self,
        *,
        shard_id: str,
        attempt_id: int,
        error_message: str,
        exit_code: int | None = 1,
    ) -> str:
        """Record a failed attempt and return the shard's next status."""

        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                """
                SELECT attempt_count, max_attempts
                FROM shards
                WHERE shard_id = ?
                """,
                (shard_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"unknown shard: {shard_id}")

            next_status = (
                "pending"
                if int(row["attempt_count"]) < int(row["max_attempts"])
                else "failed"
            )
            self.conn.execute(
                """
                UPDATE attempts
                SET status = 'failed',
                    ended_at = ?,
                    exit_code = ?,
                    error_message = ?
                WHERE attempt_id = ?
                """,
                (now, exit_code, error_message, attempt_id),
            )
            self.conn.execute(
                """
                UPDATE shards
                SET status = ?,
                    claim_worker_id = NULL,
                    claim_hostname = NULL,
                    claim_pid = NULL,
                    claim_gpu_id = NULL,
                    claimed_at = NULL,
                    heartbeat_at = NULL,
                    completed_at = CASE WHEN ? = 'failed' THEN ? ELSE completed_at END,
                    error_message = CASE WHEN ? = 'failed' THEN ? ELSE NULL END
                WHERE shard_id = ?
                """,
                (
                    next_status,
                    next_status,
                    now,
                    next_status,
                    error_message,
                    shard_id,
                ),
            )
            self.conn.commit()
            return next_status
        except Exception:
            self.conn.rollback()
            raise

    def create_or_update_msa_job(
        self,
        *,
        msa_job_id: str,
        scope: str,
        cache_key: str,
        msa_context_hash: str,
        candidate_id: str | None = None,
        reason: str = "",
        representative_sequence: str | None = None,
        member_sequences: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
        validation_config_hash: str | None = None,
        max_attempts: int = 3,
    ) -> bool:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        members = _unique_texts(member_sequences or [])
        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            existing = self.conn.execute(
                """
                SELECT msa_job_id,
                       status,
                       representative_sequence,
                       member_sequences_json,
                       metadata_json
                FROM validation_msa_jobs
                WHERE scope = ?
                  AND cache_key = ?
                  AND msa_context_hash = ?
                """,
                (scope, cache_key, msa_context_hash),
            ).fetchone()
            created = existing is None
            if existing is None:
                self.conn.execute(
                    """
                    INSERT INTO validation_msa_jobs (
                        msa_job_id,
                        scope,
                        cache_key,
                        msa_context_hash,
                        representative_sequence,
                        member_sequences_json,
                        metadata_json,
                        max_attempts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        msa_job_id,
                        scope,
                        cache_key,
                        msa_context_hash,
                        representative_sequence,
                        _json_text(members),
                        _json_text(metadata or {}),
                        max_attempts,
                    ),
                )
                job_id = msa_job_id
            else:
                job_id = existing["msa_job_id"]
                existing_members = _json_list(existing["member_sequences_json"])
                existing_metadata = _json_dict(existing["metadata_json"])
                merged_metadata = {**(metadata or {}), **existing_metadata}
                merged_members = _unique_texts([*existing_members, *members])
                has_new_members = set(merged_members) != set(existing_members)
                reopens_ready_job = has_new_members and existing["status"] == "ready"
                next_status = "pending" if reopens_ready_job else existing["status"]
                representative_sequence = existing[
                    "representative_sequence"
                ] or representative_sequence
                self.conn.execute(
                    """
                    UPDATE validation_msa_jobs
                    SET representative_sequence = COALESCE(?, representative_sequence),
                        member_sequences_json = ?,
                        metadata_json = ?,
                        max_attempts = ?,
                        status = ?,
                        attempt_count = CASE WHEN ? THEN 0 ELSE attempt_count END,
                        claim_worker_id = CASE WHEN ? THEN NULL ELSE claim_worker_id END,
                        claim_hostname = CASE WHEN ? THEN NULL ELSE claim_hostname END,
                        claim_pid = CASE WHEN ? THEN NULL ELSE claim_pid END,
                        claimed_at = CASE WHEN ? THEN NULL ELSE claimed_at END,
                        heartbeat_at = CASE WHEN ? THEN NULL ELSE heartbeat_at END,
                        completed_at = CASE WHEN ? THEN NULL ELSE completed_at END,
                        next_eligible_at = CASE WHEN ? THEN NULL ELSE next_eligible_at END,
                        error_message = CASE WHEN ? THEN NULL ELSE error_message END
                    WHERE msa_job_id = ?
                    """,
                    (
                        representative_sequence,
                        _json_text(merged_members),
                        _json_text(merged_metadata),
                        max_attempts,
                        next_status,
                        1 if reopens_ready_job else 0,
                        1 if reopens_ready_job else 0,
                        1 if reopens_ready_job else 0,
                        1 if reopens_ready_job else 0,
                        1 if reopens_ready_job else 0,
                        1 if reopens_ready_job else 0,
                        1 if reopens_ready_job else 0,
                        1 if reopens_ready_job else 0,
                        1 if reopens_ready_job else 0,
                        job_id,
                    ),
                )
            if candidate_id is not None:
                dependency_hash = str(validation_config_hash or "")
                self.conn.execute(
                    """
                    INSERT INTO validation_msa_job_candidates (
                        candidate_id,
                        msa_job_id,
                        validation_config_hash,
                        reason,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(candidate_id, msa_job_id, validation_config_hash) DO UPDATE SET
                        reason = excluded.reason
                    """,
                    (candidate_id, job_id, dependency_hash, reason, now),
                )
            self.conn.commit()
            return created
        except Exception:
            self.conn.rollback()
            raise

    def claim_next_pending_msa_job(
        self,
        *,
        worker_id: str,
        hostname: str | None = None,
        pid: int | None = None,
    ) -> MsaJobClaim | None:
        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                """
                SELECT *
                FROM validation_msa_jobs
                WHERE status = 'pending'
                  AND attempt_count < max_attempts
                  AND (next_eligible_at IS NULL OR next_eligible_at <= ?)
                ORDER BY created_at, msa_job_id
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                self.conn.commit()
                return None
            self.conn.execute(
                """
                UPDATE validation_msa_jobs
                SET status = 'running',
                    claim_worker_id = ?,
                    claim_hostname = ?,
                    claim_pid = ?,
                    claimed_at = ?,
                    heartbeat_at = ?,
                    started_at = COALESCE(started_at, ?),
                    attempt_count = attempt_count + 1,
                    error_message = NULL
                WHERE msa_job_id = ?
                  AND status = 'pending'
                """,
                (
                    worker_id,
                    hostname,
                    pid,
                    now,
                    now,
                    now,
                    row["msa_job_id"],
                ),
            )
            self.conn.commit()
            return MsaJobClaim(
                msa_job_id=row["msa_job_id"],
                scope=row["scope"],
                cache_key=row["cache_key"],
                msa_context_hash=row["msa_context_hash"],
                attempt_count=int(row["attempt_count"]) + 1,
                representative_sequence=row["representative_sequence"],
                member_sequences=tuple(_json_list(row["member_sequences_json"])),
                metadata=_json_dict(row["metadata_json"]),
            )
        except Exception:
            self.conn.rollback()
            raise

    def recover_stale_msa_jobs(
        self,
        *,
        stale_before: str,
        error_message: str = "stale MSA claim",
    ) -> int:
        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                """
                SELECT msa_job_id, attempt_count, max_attempts
                FROM validation_msa_jobs
                WHERE status = 'running'
                  AND heartbeat_at IS NOT NULL
                  AND heartbeat_at < ?
                ORDER BY msa_job_id
                """,
                (stale_before,),
            ).fetchall()
            for row in rows:
                next_status = (
                    "pending"
                    if int(row["attempt_count"]) < int(row["max_attempts"])
                    else "failed"
                )
                self.conn.execute(
                    """
                    UPDATE validation_msa_jobs
                    SET status = ?,
                        claim_worker_id = NULL,
                        claim_hostname = NULL,
                        claim_pid = NULL,
                        claimed_at = NULL,
                        heartbeat_at = NULL,
                        completed_at = CASE WHEN ? = 'failed' THEN ? ELSE completed_at END,
                        next_eligible_at = NULL,
                        error_message = CASE WHEN ? = 'failed' THEN ? ELSE NULL END
                    WHERE msa_job_id = ?
                    """,
                    (
                        next_status,
                        next_status,
                        now,
                        next_status,
                        error_message,
                        row["msa_job_id"],
                    ),
                )
            self.conn.commit()
            return len(rows)
        except Exception:
            self.conn.rollback()
            raise

    def complete_msa_job(
        self,
        *,
        msa_job_id: str,
        cache_paths: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _utc_now()
        row = self.conn.execute(
            "SELECT metadata_json FROM validation_msa_jobs WHERE msa_job_id = ?",
            (msa_job_id,),
        ).fetchone()
        existing_metadata = _json_dict(row["metadata_json"]) if row else {}
        merged_metadata = {**existing_metadata, **(metadata or {})}
        self.conn.execute(
            """
            UPDATE validation_msa_jobs
            SET status = 'ready',
                claim_worker_id = NULL,
                claim_hostname = NULL,
                claim_pid = NULL,
                claimed_at = NULL,
                heartbeat_at = ?,
                completed_at = ?,
                cache_paths_json = ?,
                metadata_json = ?,
                error_message = NULL
            WHERE msa_job_id = ?
            """,
            (
                now,
                now,
                _json_text(cache_paths or {}),
                _json_text(merged_metadata),
                msa_job_id,
            ),
        )
        self.conn.commit()

    def reopen_ready_msa_jobs_with_missing_cache(
        self,
        *,
        base_dir: str | Path | None = None,
    ) -> int:
        """Move ready MSA jobs back to pending when required cache files vanished."""

        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                """
                SELECT msa_job_id, cache_paths_json
                FROM validation_msa_jobs
                WHERE status = 'ready'
                ORDER BY msa_job_id
                """
            ).fetchall()
            reopened = 0
            for row in rows:
                missing_path = _first_missing_msa_cache_path(
                    _json_dict(row["cache_paths_json"]),
                    base_dir=base_dir,
                )
                if missing_path is None:
                    continue
                self.conn.execute(
                    """
                    UPDATE validation_msa_jobs
                    SET status = 'pending',
                        attempt_count = 0,
                        claim_worker_id = NULL,
                        claim_hostname = NULL,
                        claim_pid = NULL,
                        claimed_at = NULL,
                        heartbeat_at = NULL,
                        completed_at = NULL,
                        next_eligible_at = NULL,
                        error_message = ?
                    WHERE msa_job_id = ?
                      AND status = 'ready'
                    """,
                    (
                        f"ready MSA cache artifact missing: {missing_path}",
                        row["msa_job_id"],
                    ),
                )
                reopened += 1
            self.conn.commit()
            return reopened
        except Exception:
            self.conn.rollback()
            raise

    def skip_msa_job(
        self,
        *,
        msa_job_id: str,
        error_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _utc_now()
        row = self.conn.execute(
            "SELECT metadata_json FROM validation_msa_jobs WHERE msa_job_id = ?",
            (msa_job_id,),
        ).fetchone()
        existing_metadata = _json_dict(row["metadata_json"]) if row else {}
        merged_metadata = {**existing_metadata, **(metadata or {})}
        self.conn.execute(
            """
            UPDATE validation_msa_jobs
            SET status = 'skipped',
                claim_worker_id = NULL,
                claim_hostname = NULL,
                claim_pid = NULL,
                claimed_at = NULL,
                heartbeat_at = ?,
                completed_at = ?,
                metadata_json = ?,
                error_message = ?
            WHERE msa_job_id = ?
            """,
            (now, now, _json_text(merged_metadata), error_message, msa_job_id),
        )
        self.conn.commit()

    def fail_msa_job(
        self,
        *,
        msa_job_id: str,
        error_message: str,
        retry_after_seconds: float | None = None,
    ) -> str:
        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                """
                SELECT attempt_count, max_attempts
                FROM validation_msa_jobs
                WHERE msa_job_id = ?
                """,
                (msa_job_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"unknown MSA job: {msa_job_id}")
            next_status = (
                "pending"
                if int(row["attempt_count"]) < int(row["max_attempts"])
                else "failed"
            )
            next_eligible_at = (
                _utc_offset_seconds(retry_after_seconds)
                if retry_after_seconds is not None and retry_after_seconds > 0
                else None
            )
            self.conn.execute(
                """
                UPDATE validation_msa_jobs
                SET status = ?,
                    claim_worker_id = NULL,
                    claim_hostname = NULL,
                    claim_pid = NULL,
                    claimed_at = NULL,
                    heartbeat_at = NULL,
                    completed_at = CASE WHEN ? = 'failed' THEN ? ELSE completed_at END,
                    next_eligible_at = ?,
                    error_message = ?
                WHERE msa_job_id = ?
                """,
                (
                    next_status,
                    next_status,
                    now,
                    next_eligible_at,
                    error_message,
                    msa_job_id,
                ),
            )
            self.conn.commit()
            return next_status
        except Exception:
            self.conn.rollback()
            raise

    def retry_failed_msa_jobs(
        self,
        *,
        msa_job_ids: list[str] | tuple[str, ...] | None = None,
        candidate_ids: list[str] | tuple[str, ...] | None = None,
        reset_attempt_count: bool = True,
    ) -> int:
        job_ids = _unique_texts(msa_job_ids or [])
        candidates = _unique_texts(candidate_ids or [])
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            query = """
                SELECT DISTINCT job.msa_job_id
                FROM validation_msa_jobs AS job
            """
            params: list[Any] = []
            if candidates:
                query += """
                    JOIN validation_msa_job_candidates AS dep
                      ON dep.msa_job_id = job.msa_job_id
                """
            filters = ["job.status = 'failed'"]
            if job_ids:
                filters.append(
                    "job.msa_job_id IN ("
                    + ",".join("?" for _value in job_ids)
                    + ")"
                )
                params.extend(job_ids)
            if candidates:
                filters.append(
                    "dep.candidate_id IN ("
                    + ",".join("?" for _value in candidates)
                    + ")"
                )
                params.extend(candidates)
            query += " WHERE " + " AND ".join(filters)
            rows = self.conn.execute(query, tuple(params)).fetchall()
            retry_ids = [row["msa_job_id"] for row in rows]
            if not retry_ids:
                self.conn.commit()
                return 0

            update_params: list[Any] = [1 if reset_attempt_count else 0, *retry_ids]
            self.conn.execute(
                f"""
                UPDATE validation_msa_jobs
                SET status = 'pending',
                    claim_worker_id = NULL,
                    claim_hostname = NULL,
                    claim_pid = NULL,
                    claimed_at = NULL,
                    heartbeat_at = NULL,
                    completed_at = NULL,
                    next_eligible_at = NULL,
                    attempt_count = CASE WHEN ? THEN 0 ELSE attempt_count END,
                    error_message = NULL
                WHERE msa_job_id IN ({",".join("?" for _value in retry_ids)})
                """,
                tuple(update_params),
            )
            self.conn.commit()
            return len(retry_ids)
        except Exception:
            self.conn.rollback()
            raise

    def try_acquire_msa_rate_slot(
        self,
        *,
        name: str,
        min_interval_seconds: float,
    ) -> float:
        now_dt = datetime.now(timezone.utc)
        now = _utc_now_from_datetime(now_dt)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT last_submit_at FROM msa_rate_limits WHERE name = ?",
                (name,),
            ).fetchone()
            if row is not None and row["last_submit_at"]:
                last = _parse_utc(row["last_submit_at"])
                elapsed = (now_dt - last).total_seconds()
                wait_seconds = float(min_interval_seconds) - elapsed
                if wait_seconds > 0:
                    self.conn.rollback()
                    return wait_seconds
            self.conn.execute(
                """
                INSERT INTO msa_rate_limits (name, last_submit_at)
                VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    last_submit_at = excluded.last_submit_at
                """,
                (name, now),
            )
            self.conn.commit()
            return 0.0
        except Exception:
            self.conn.rollback()
            raise

    def create_validation_task(
        self,
        *,
        validation_id: str,
        candidate_id: str,
        model_name: str,
        validation_config_hash: str,
        selection_rank: int | None = None,
        max_attempts: int = 3,
    ) -> bool:
        """Create or refresh a validation task without resetting completed work."""

        if selection_rank is not None and selection_rank <= 0:
            raise ValueError("selection_rank must be positive when provided")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            existing = self.conn.execute(
                """
                SELECT validation_id
                FROM validation_tasks
                WHERE candidate_id = ?
                  AND model_name = ?
                  AND validation_config_hash = ?
                """,
                (candidate_id, model_name, validation_config_hash),
            ).fetchone()
            if existing is not None:
                self.conn.execute(
                    """
                    UPDATE validation_tasks
                    SET selection_rank = ?,
                        max_attempts = ?
                    WHERE validation_id = ?
                    """,
                    (selection_rank, max_attempts, existing["validation_id"]),
                )
                self.conn.commit()
                return False

            self.conn.execute(
                """
                INSERT INTO validation_tasks (
                    validation_id,
                    candidate_id,
                    model_name,
                    validation_config_hash,
                    selection_rank,
                    max_attempts
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    validation_id,
                    candidate_id,
                    model_name,
                    validation_config_hash,
                    selection_rank,
                    max_attempts,
                ),
            )
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def claim_next_pending_validation_tasks(
        self,
        *,
        worker_id: str,
        batch_size: int = 1,
        hostname: str | None = None,
        pid: int | None = None,
        gpu_id: str | None = None,
    ) -> list[ValidationClaim]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                """
                SELECT *
                FROM validation_tasks
                WHERE validation_tasks.status = 'pending'
                  AND validation_tasks.attempt_count < validation_tasks.max_attempts
                  AND NOT EXISTS (
                      SELECT 1
                      FROM validation_msa_job_candidates AS dep
                      JOIN validation_msa_jobs AS job
                        ON job.msa_job_id = dep.msa_job_id
                      WHERE dep.candidate_id = validation_tasks.candidate_id
                        AND dep.validation_config_hash = validation_tasks.validation_config_hash
                        AND job.status NOT IN ('ready', 'skipped')
                  )
                ORDER BY
                    selection_rank IS NULL,
                    selection_rank,
                    validation_id
                LIMIT ?
                """,
                (batch_size,),
            ).fetchall()
            claims: list[ValidationClaim] = []
            for row in rows:
                self.conn.execute(
                    """
                    UPDATE validation_tasks
                    SET status = 'running',
                        claim_worker_id = ?,
                        claim_hostname = ?,
                        claim_pid = ?,
                        claim_gpu_id = ?,
                        claimed_at = ?,
                        heartbeat_at = ?,
                        started_at = COALESCE(started_at, ?),
                        attempt_count = attempt_count + 1,
                        error_message = NULL
                    WHERE validation_id = ?
                      AND status = 'pending'
                    """,
                    (
                        worker_id,
                        hostname,
                        pid,
                        gpu_id,
                        now,
                        now,
                        now,
                        row["validation_id"],
                    ),
                )
                cursor = self.conn.execute(
                    """
                    INSERT INTO attempts (
                        candidate_id,
                        validation_id,
                        critic_name,
                        stage,
                        status,
                        worker_id,
                        hostname,
                        pid,
                        gpu_id,
                        started_at
                    )
                    VALUES (?, ?, ?, 'validation', 'running', ?, ?, ?, ?, ?)
                    """,
                    (
                        row["candidate_id"],
                        row["validation_id"],
                        row["model_name"],
                        worker_id,
                        hostname,
                        pid,
                        gpu_id,
                        now,
                    ),
                )
                claims.append(
                    ValidationClaim(
                        validation_id=row["validation_id"],
                        attempt_id=int(cursor.lastrowid),
                        candidate_id=row["candidate_id"],
                        model_name=row["model_name"],
                        validation_config_hash=row["validation_config_hash"],
                        selection_rank=(
                            int(row["selection_rank"])
                            if row["selection_rank"] is not None
                            else None
                        ),
                    )
                )
            self.conn.commit()
            return claims
        except Exception:
            self.conn.rollback()
            raise

    def heartbeat_validation_task(
        self,
        *,
        validation_id: str,
        attempt_id: int,
    ) -> None:
        now = _utc_now()
        self.conn.execute(
            """
            UPDATE validation_tasks
            SET heartbeat_at = ?
            WHERE validation_id = ?
              AND status = 'running'
            """,
            (now, validation_id),
        )
        self.conn.execute(
            """
            UPDATE attempts
            SET ended_at = NULL
            WHERE attempt_id = ?
              AND status = 'running'
            """,
            (attempt_id,),
        )
        self.conn.commit()

    def recover_stale_validation_tasks(
        self,
        *,
        stale_before: str,
        error_message: str = "stale validation claim",
    ) -> int:
        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                """
                SELECT *
                FROM validation_tasks
                WHERE status = 'running'
                  AND heartbeat_at IS NOT NULL
                  AND heartbeat_at < ?
                ORDER BY validation_id
                """,
                (stale_before,),
            ).fetchall()
            for row in rows:
                next_status = (
                    "pending"
                    if int(row["attempt_count"]) < int(row["max_attempts"])
                    else "failed"
                )
                self.conn.execute(
                    """
                    UPDATE attempts
                    SET status = 'stale',
                        ended_at = ?,
                        error_message = COALESCE(error_message, ?)
                    WHERE validation_id = ?
                      AND stage = 'validation'
                      AND status = 'running'
                    """,
                    (now, error_message, row["validation_id"]),
                )
                self.conn.execute(
                    """
                    UPDATE validation_tasks
                    SET status = ?,
                        claim_worker_id = NULL,
                        claim_hostname = NULL,
                        claim_pid = NULL,
                        claim_gpu_id = NULL,
                        claimed_at = NULL,
                        heartbeat_at = NULL,
                        completed_at = CASE WHEN ? = 'failed' THEN ? ELSE completed_at END,
                        error_message = CASE WHEN ? = 'failed' THEN ? ELSE NULL END
                    WHERE validation_id = ?
                    """,
                    (
                        next_status,
                        next_status,
                        now,
                        next_status,
                        error_message,
                        row["validation_id"],
                    ),
                )
            self.conn.commit()
            return len(rows)
        except Exception:
            self.conn.rollback()
            raise

    def recover_failed_worker_validation_tasks(
        self,
        *,
        worker_id: str,
        pid: int | None = None,
        exit_code: int | None = None,
        error_message: str = "worker process exited before completing validation",
    ) -> int:
        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            query = """
                SELECT *
                FROM validation_tasks
                WHERE status = 'running'
                  AND claim_worker_id = ?
            """
            params: list[Any] = [worker_id]
            if pid is not None:
                query += " AND claim_pid = ?"
                params.append(pid)
            query += " ORDER BY validation_id"

            rows = self.conn.execute(query, tuple(params)).fetchall()
            for row in rows:
                next_status = (
                    "pending"
                    if int(row["attempt_count"]) < int(row["max_attempts"])
                    else "failed"
                )
                attempt_params: list[Any] = [
                    now,
                    exit_code,
                    error_message,
                    row["validation_id"],
                    worker_id,
                ]
                attempt_filter = ""
                if pid is not None:
                    attempt_filter = "AND pid = ?"
                    attempt_params.append(pid)
                self.conn.execute(
                    f"""
                    UPDATE attempts
                    SET status = 'failed',
                        ended_at = ?,
                        exit_code = ?,
                        error_message = ?
                    WHERE validation_id = ?
                      AND worker_id = ?
                      {attempt_filter}
                      AND stage = 'validation'
                      AND status = 'running'
                    """,
                    tuple(attempt_params),
                )
                self.conn.execute(
                    """
                    UPDATE validation_tasks
                    SET status = ?,
                        claim_worker_id = NULL,
                        claim_hostname = NULL,
                        claim_pid = NULL,
                        claim_gpu_id = NULL,
                        claimed_at = NULL,
                        heartbeat_at = NULL,
                        completed_at = CASE WHEN ? = 'failed' THEN ? ELSE completed_at END,
                        error_message = CASE WHEN ? = 'failed' THEN ? ELSE NULL END
                    WHERE validation_id = ?
                    """,
                    (
                        next_status,
                        next_status,
                        now,
                        next_status,
                        error_message,
                        row["validation_id"],
                    ),
                )
            self.conn.commit()
            return len(rows)
        except Exception:
            self.conn.rollback()
            raise

    def fail_validation_task(
        self,
        *,
        validation_id: str,
        attempt_id: int,
        error_message: str,
        exit_code: int | None = 1,
    ) -> str:
        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                """
                SELECT attempt_count, max_attempts
                FROM validation_tasks
                WHERE validation_id = ?
                """,
                (validation_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"unknown validation task: {validation_id}")
            next_status = (
                "pending"
                if int(row["attempt_count"]) < int(row["max_attempts"])
                else "failed"
            )
            self.conn.execute(
                """
                UPDATE attempts
                SET status = 'failed',
                    ended_at = ?,
                    exit_code = ?,
                    error_message = ?
                WHERE attempt_id = ?
                """,
                (now, exit_code, error_message, attempt_id),
            )
            self.conn.execute(
                """
                UPDATE validation_tasks
                SET status = ?,
                    claim_worker_id = NULL,
                    claim_hostname = NULL,
                    claim_pid = NULL,
                    claim_gpu_id = NULL,
                    claimed_at = NULL,
                    heartbeat_at = NULL,
                    completed_at = CASE WHEN ? = 'failed' THEN ? ELSE completed_at END,
                    error_message = CASE WHEN ? = 'failed' THEN ? ELSE NULL END
                WHERE validation_id = ?
                """,
                (
                    next_status,
                    next_status,
                    now,
                    next_status,
                    error_message,
                    validation_id,
                ),
            )
            self.conn.commit()
            return next_status
        except Exception:
            self.conn.rollback()
            raise

    def skip_validation_task(
        self,
        *,
        validation_id: str,
        error_message: str,
        attempt_id: int | None = None,
    ) -> None:
        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                """
                UPDATE validation_tasks
                SET status = 'skipped',
                    claim_worker_id = NULL,
                    claim_hostname = NULL,
                    claim_pid = NULL,
                    claim_gpu_id = NULL,
                    claimed_at = NULL,
                    heartbeat_at = NULL,
                    completed_at = ?,
                    error_message = ?
                WHERE validation_id = ?
                """,
                (now, error_message, validation_id),
            )
            if attempt_id is not None:
                self.conn.execute(
                    """
                    UPDATE attempts
                    SET status = 'completed',
                        ended_at = ?,
                        exit_code = 0,
                        error_message = ?
                    WHERE attempt_id = ?
                    """,
                    (now, error_message, attempt_id),
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def record_validation_structure(
        self,
        *,
        validation_id: str,
        structure_id: str,
        candidate_id: str,
        model_name: str,
        seed: int,
        sample_rank: int,
        status: str,
        structure_path: str,
        metrics: dict[str, Any],
    ) -> None:
        if status not in {"pending", "passing", "rejected"}:
            raise ValueError("validation structure status must be pending, passing, or rejected")
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO validation_structures (
                validation_id,
                structure_id,
                candidate_id,
                model_name,
                seed,
                sample_rank,
                status,
                structure_path,
                metrics_json,
                scoped_iptm,
                scoped_ipsae,
                ptm,
                ranking_score,
                hotspot_satisfaction,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(validation_id, structure_id) DO UPDATE SET
                status = excluded.status,
                structure_path = excluded.structure_path,
                metrics_json = excluded.metrics_json,
                scoped_iptm = excluded.scoped_iptm,
                scoped_ipsae = excluded.scoped_ipsae,
                ptm = excluded.ptm,
                ranking_score = excluded.ranking_score,
                hotspot_satisfaction = excluded.hotspot_satisfaction
            """,
            (
                validation_id,
                structure_id,
                candidate_id,
                model_name,
                seed,
                sample_rank,
                status,
                structure_path,
                _json_text(metrics),
                _first_metric(metrics, "validation_iptm", "scoped_iptm", "iptm"),
                _first_metric(
                    metrics,
                    "validation_ipSAE",
                    "validation_ipsae",
                    "scoped_ipsae",
                    "ipsae",
                ),
                metrics.get("ptm"),
                metrics.get("ranking_score"),
                _first_metric(
                    metrics,
                    "hotspot_satisfaction",
                    "validation_hotspot_satisfaction",
                ),
                now,
            ),
        )
        self.conn.commit()

    def record_completed_validation(
        self,
        *,
        validation_id: str,
        output_structure_path: str,
        metrics: dict[str, Any],
        runtime_seconds: float | None = None,
    ) -> None:
        now = _utc_now()
        self._require_promoted_validation_structure(
            validation_id=validation_id,
            output_structure_path=output_structure_path,
        )
        self.conn.execute(
            """
            UPDATE validation_tasks
            SET status = 'completed',
                completed_at = ?,
                heartbeat_at = ?,
                output_structure_path = ?,
                metrics_json = ?,
                iptm = ?,
                ipsae = ?,
                ptm = ?,
                ranking_score = ?,
                hotspot_satisfaction = ?,
                runtime_seconds = ?,
                error_message = NULL
            WHERE validation_id = ?
            """,
            (
                now,
                now,
                output_structure_path,
                _json_text(metrics),
                _first_metric(metrics, "validation_iptm", "scoped_iptm", "iptm"),
                _first_metric(
                    metrics,
                    "validation_ipSAE",
                    "validation_ipsae",
                    "scoped_ipsae",
                    "ipsae",
                ),
                metrics.get("ptm"),
                metrics.get("ranking_score"),
                _first_metric(
                    metrics,
                    "hotspot_satisfaction",
                    "validation_hotspot_satisfaction",
                ),
                runtime_seconds,
                validation_id,
            ),
        )
        self.conn.commit()

    def complete_validation_task(
        self,
        *,
        validation_id: str,
        attempt_id: int,
        output_structure_path: str,
        metrics: dict[str, Any],
        runtime_seconds: float | None = None,
    ) -> None:
        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self._require_promoted_validation_structure(
                validation_id=validation_id,
                output_structure_path=output_structure_path,
            )
            self.conn.execute(
                """
                UPDATE validation_tasks
                SET status = 'completed',
                    completed_at = ?,
                    heartbeat_at = ?,
                    output_structure_path = ?,
                    metrics_json = ?,
                    iptm = ?,
                    ipsae = ?,
                    ptm = ?,
                    ranking_score = ?,
                    hotspot_satisfaction = ?,
                    runtime_seconds = ?,
                    error_message = NULL
                WHERE validation_id = ?
                """,
                (
                    now,
                    now,
                    output_structure_path,
                    _json_text(metrics),
                    _first_metric(metrics, "validation_iptm", "scoped_iptm", "iptm"),
                    _first_metric(
                        metrics,
                        "validation_ipSAE",
                        "validation_ipsae",
                        "scoped_ipsae",
                        "ipsae",
                    ),
                    metrics.get("ptm"),
                    metrics.get("ranking_score"),
                    _first_metric(
                        metrics,
                        "hotspot_satisfaction",
                        "validation_hotspot_satisfaction",
                    ),
                    runtime_seconds,
                    validation_id,
                ),
            )
            self.conn.execute(
                """
                UPDATE attempts
                SET status = 'completed',
                    ended_at = ?,
                    exit_code = 0,
                    error_message = NULL
                WHERE attempt_id = ?
                """,
                (now, attempt_id),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _require_promoted_validation_structure(
        self,
        *,
        validation_id: str,
        output_structure_path: str,
    ) -> None:
        row = self.conn.execute(
            """
            SELECT metrics_json
            FROM validation_structures
            WHERE validation_id = ?
              AND structure_path = ?
              AND status IN ('passing', 'rejected')
            """,
            (validation_id, output_structure_path),
        ).fetchone()
        if row is None:
            raise ValueError(
                "cannot complete validation task before the selected CIF is "
                "promoted and recorded in validation_structures"
            )
        try:
            metrics = json.loads(row["metrics_json"] or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(
                "cannot complete validation task with invalid selected CIF metrics"
            ) from exc
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(
                "cannot complete validation task before selected CIF metrics are "
                "recorded in validation_structures"
            )

    def record_completed_candidate(
        self,
        *,
        candidate_id: str,
        shard_id: str,
        candidate_index: int,
        designed_sequence: str,
        sequence_path: str | None,
        binder_chain_id: str | None = None,
        design_metrics: dict[str, Any] | None = None,
    ) -> None:
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO candidates (
                candidate_id,
                shard_id,
                candidate_index,
                designed_sequence,
                binder_chain_id,
                status,
                sequence_path,
                design_metrics_json,
                started_at,
                completed_at
            )
            VALUES (?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                designed_sequence = excluded.designed_sequence,
                binder_chain_id = excluded.binder_chain_id,
                status = 'completed',
                sequence_path = excluded.sequence_path,
                design_metrics_json = excluded.design_metrics_json,
                completed_at = excluded.completed_at,
                error_message = NULL
            """,
            (
                candidate_id,
                shard_id,
                candidate_index,
                designed_sequence,
                binder_chain_id,
                sequence_path,
                _json_text(design_metrics or {}),
                now,
                now,
            ),
        )
        self.conn.commit()

    def record_completed_critic(
        self,
        *,
        candidate_id: str,
        critic_name: str,
        structure_path: str,
        metrics: dict[str, Any],
        runtime_seconds: float | None = None,
    ) -> None:
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO critic_metrics (
                candidate_id,
                critic_name,
                status,
                structure_path,
                metrics_json,
                iptm,
                ptm,
                plddt,
                distogram_iptm_proxy,
                hotspot_satisfaction,
                runtime_seconds,
                attempt_count,
                started_at,
                completed_at
            )
            VALUES (
                ?, ?, 'completed', ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE(
                    (
                        SELECT attempt_count + 1
                        FROM critic_metrics
                        WHERE candidate_id = ?
                          AND critic_name = ?
                    ),
                    1
                ),
                ?, ?
            )
            ON CONFLICT(candidate_id, critic_name) DO UPDATE SET
                status = 'completed',
                structure_path = excluded.structure_path,
                metrics_json = excluded.metrics_json,
                iptm = excluded.iptm,
                ptm = excluded.ptm,
                plddt = excluded.plddt,
                distogram_iptm_proxy = excluded.distogram_iptm_proxy,
                hotspot_satisfaction = excluded.hotspot_satisfaction,
                runtime_seconds = excluded.runtime_seconds,
                attempt_count = excluded.attempt_count,
                completed_at = excluded.completed_at,
                error_message = NULL
            """,
            (
                candidate_id,
                critic_name,
                structure_path,
                _json_text(metrics),
                metrics.get("iptm"),
                metrics.get("ptm"),
                metrics.get("plddt"),
                metrics.get("distogram_iptm_proxy"),
                metrics.get("hotspot_satisfaction"),
                runtime_seconds,
                candidate_id,
                critic_name,
                now,
                now,
            ),
        )
        self.conn.commit()

    def complete_shard(
        self,
        *,
        shard_id: str,
        attempt_id: int,
    ) -> None:
        now = _utc_now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                """
                UPDATE shards
                SET status = 'completed',
                    completed_at = ?,
                    heartbeat_at = ?,
                    error_message = NULL
                WHERE shard_id = ?
                """,
                (now, now, shard_id),
            )
            self.conn.execute(
                """
                UPDATE attempts
                SET status = 'completed',
                    ended_at = ?,
                    exit_code = 0,
                    error_message = NULL
                WHERE attempt_id = ?
                """,
                (now, attempt_id),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row:
        row = self.conn.execute(query, params).fetchone()
        if row is None:
            raise LookupError("query returned no rows")
        return row


def connect_database(
    db_path: str | Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    """Open a campaign database with the runtime pragmas the workers need."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, timeout=busy_timeout_ms / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    return conn


def initialize_database(
    db_path: str | Path,
    *,
    config_hash: str = "uninitialized",
    resolved_config: dict[str, Any] | None = None,
    software_versions: dict[str, Any] | None = None,
) -> sqlite3.Connection:
    """Create the minimal campaign ledger and ensure the singleton campaign row."""

    conn = connect_database(db_path)
    conn.executescript(_schema_sql())
    _migrate_attempts_for_validation(conn)
    _migrate_validation_msa_job_candidates(conn)
    conn.execute(
        """
        INSERT INTO campaign (
            id,
            schema_version,
            config_hash,
            resolved_config_json,
            software_versions_json
        )
        VALUES (1, ?, ?, ?, ?)
        ON CONFLICT(id) DO NOTHING
        """,
        (
            SCHEMA_VERSION,
            config_hash,
            _json_text(resolved_config or {}),
            _json_text(software_versions or {}),
        ),
    )
    conn.execute(
        """
        UPDATE campaign
        SET schema_version = ?
        WHERE id = 1
          AND schema_version < ?
        """,
        (SCHEMA_VERSION, SCHEMA_VERSION),
    )
    conn.commit()
    return conn


def _schema_sql() -> str:
    return resources.files("esmfold2_pipeline.db").joinpath("schema.sql").read_text()


def _migrate_attempts_for_validation(conn: sqlite3.Connection) -> None:
    """Rebuild pre-v3 attempts tables so validation attempts can be recorded."""

    row = conn.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'attempts'
        """
    ).fetchone()
    if row is None:
        return
    sql = str(row["sql"] or "")
    if "'validation'" in sql and "validation_id" in sql:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attempts_validation ON attempts(validation_id)"
        )
        return

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("ALTER TABLE attempts RENAME TO attempts_old")
        conn.execute(
            """
            CREATE TABLE attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                shard_id TEXT REFERENCES shards(shard_id) ON DELETE CASCADE,
                candidate_id TEXT REFERENCES candidates(candidate_id) ON DELETE CASCADE,
                validation_id TEXT REFERENCES validation_tasks(validation_id) ON DELETE CASCADE,
                critic_name TEXT,
                stage TEXT NOT NULL CHECK (
                    stage IN ('shard', 'design', 'critic', 'worker', 'validation')
                ),
                status TEXT NOT NULL CHECK (
                    status IN ('running', 'completed', 'failed', 'stale', 'cancelled')
                ),
                worker_id TEXT NOT NULL,
                hostname TEXT,
                pid INTEGER,
                gpu_id TEXT,
                started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                ended_at TEXT,
                exit_code INTEGER,
                log_path TEXT,
                traceback_path TEXT,
                error_message TEXT,
                CHECK (
                    shard_id IS NOT NULL
                    OR candidate_id IS NOT NULL
                    OR validation_id IS NOT NULL
                )
            )
            """
        )
        conn.execute(
            """
            INSERT INTO attempts (
                attempt_id,
                shard_id,
                candidate_id,
                critic_name,
                stage,
                status,
                worker_id,
                hostname,
                pid,
                gpu_id,
                started_at,
                ended_at,
                exit_code,
                log_path,
                traceback_path,
                error_message
            )
            SELECT
                attempt_id,
                shard_id,
                candidate_id,
                critic_name,
                stage,
                status,
                worker_id,
                hostname,
                pid,
                gpu_id,
                started_at,
                ended_at,
                exit_code,
                log_path,
                traceback_path,
                error_message
            FROM attempts_old
            """
        )
        conn.execute("DROP TABLE attempts_old")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attempts_shard ON attempts(shard_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attempts_candidate ON attempts(candidate_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attempts_status ON attempts(status, started_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attempts_worker ON attempts(worker_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attempts_validation ON attempts(validation_id)"
        )
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _migrate_validation_msa_job_candidates(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'validation_msa_job_candidates'
        """
    ).fetchone()
    if row is None:
        return
    table_info = conn.execute(
        "PRAGMA table_info(validation_msa_job_candidates)"
    ).fetchall()
    pk_columns = [
        column["name"]
        for column in sorted(
            (column for column in table_info if int(column["pk"]) > 0),
            key=lambda column: int(column["pk"]),
        )
    ]
    if pk_columns == ["candidate_id", "msa_job_id", "validation_config_hash"]:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_validation_msa_job_candidates_job
                ON validation_msa_job_candidates(msa_job_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_validation_msa_job_candidates_candidate_hash
                ON validation_msa_job_candidates(candidate_id, validation_config_hash)
            """
        )
        return

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute(
            "ALTER TABLE validation_msa_job_candidates RENAME TO validation_msa_job_candidates_old"
        )
        conn.execute(
            """
            CREATE TABLE validation_msa_job_candidates (
                candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
                msa_job_id TEXT NOT NULL REFERENCES validation_msa_jobs(msa_job_id) ON DELETE CASCADE,
                validation_config_hash TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                PRIMARY KEY(candidate_id, msa_job_id, validation_config_hash)
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO validation_msa_job_candidates (
                candidate_id,
                msa_job_id,
                validation_config_hash,
                reason,
                created_at
            )
            SELECT
                candidate_id,
                msa_job_id,
                '',
                reason,
                created_at
            FROM validation_msa_job_candidates_old
            """
        )
        conn.execute("DROP TABLE validation_msa_job_candidates_old")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_validation_msa_job_candidates_job
                ON validation_msa_job_candidates(msa_job_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_validation_msa_job_candidates_candidate_hash
                ON validation_msa_job_candidates(candidate_id, validation_config_hash)
            """
        )
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _first_metric(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_text_list(value: list[str]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_list(value: str | None) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item)]


def _json_dict(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _first_missing_msa_cache_path(
    value: Any,
    *,
    base_dir: str | Path | None = None,
) -> str | None:
    if not value:
        return None

    base = Path(base_dir) if base_dir is not None else None

    def resolve_candidates(raw: str) -> list[Path]:
        path = Path(raw).expanduser()
        candidates = [path]
        if base is not None and not path.is_absolute():
            based = base / path
            if based != path:
                candidates.append(based)
        return candidates

    def walk(item: Any) -> str | None:
        if isinstance(item, dict):
            for key in ("non_pairing_path", "metadata_path"):
                raw = item.get(key)
                if not isinstance(raw, str) or not raw.strip():
                    continue
                candidates = resolve_candidates(raw.strip())
                if not any(path.exists() for path in candidates):
                    return str(candidates[-1])
            for nested in item.values():
                missing = walk(nested)
                if missing is not None:
                    return missing
        elif isinstance(item, list):
            for nested in item:
                missing = walk(nested)
                if missing is not None:
                    return missing
        return None

    return walk(value)


def _unique_texts(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _utc_now() -> str:
    return _utc_now_from_datetime(datetime.now(timezone.utc))


def _utc_now_from_datetime(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _utc_offset_seconds(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return _utc_now_from_datetime(
        datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + float(seconds),
            timezone.utc,
        )
    )


def _parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
