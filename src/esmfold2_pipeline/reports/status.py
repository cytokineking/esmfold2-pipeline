from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from esmfold2_pipeline.db import connect_database


@dataclass(frozen=True)
class ReconciliationIssue:
    kind: str
    path: str
    message: str
    table: str | None = None
    row_id: str | None = None


@dataclass(frozen=True)
class MsaFailureSummary:
    msa_job_id: str
    scope: str
    candidate_ids: tuple[str, ...]
    error_message: str


@dataclass(frozen=True)
class CampaignStatus:
    schema_version: int
    campaign_db_schema_version: int | None
    config_hash: str | None
    state: str
    latest_activity_at: str | None
    validation_configured: bool
    terminal_failure_count: int
    expected_artifacts: dict[str, bool]
    table_counts: dict[str, int]
    shard_status_counts: dict[str, int]
    candidate_status_counts: dict[str, int]
    critic_status_counts: dict[str, int]
    validation_status_counts: dict[str, int]
    validation_structure_status_counts: dict[str, int]
    validation_msa_status_counts: dict[str, int]
    validation_msa_blocked_counts: dict[str, int]
    validation_msa_failures: list[MsaFailureSummary]
    attempt_status_counts: dict[str, int]
    issues: list[ReconciliationIssue]

    @property
    def missing_artifact_count(self) -> int:
        return sum(1 for issue in self.issues if issue.kind.startswith("missing_"))

    @property
    def untracked_artifact_count(self) -> int:
        return sum(1 for issue in self.issues if issue.kind == "untracked_artifact")

    @property
    def terminal(self) -> bool:
        return self.state in {"complete", "failed", "inconsistent"}

    @property
    def successful(self) -> bool:
        return self.state == "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign": {
                "db_schema_version": self.campaign_db_schema_version,
                "config_hash": self.config_hash,
                "state": self.state,
                "terminal": self.terminal,
                "successful": self.successful,
                "latest_activity_at": self.latest_activity_at,
                "validation_configured": self.validation_configured,
                "terminal_failure_count": self.terminal_failure_count,
            },
            "counts": {
                "tables": self.table_counts,
                "shards": self.shard_status_counts,
                "candidates": self.candidate_status_counts,
                "critics": self.critic_status_counts,
                "validation_tasks": self.validation_status_counts,
                "validation_structures": self.validation_structure_status_counts,
                "validation_msa_jobs": self.validation_msa_status_counts,
                "validation_tasks_blocked_by_msa": (
                    self.validation_msa_blocked_counts
                ),
                "attempts": self.attempt_status_counts,
            },
            "expected_artifacts": self.expected_artifacts,
            "reconciliation": {
                "missing_artifacts": self.missing_artifact_count,
                "untracked_artifacts": self.untracked_artifact_count,
                "issues": [
                    {
                        "kind": issue.kind,
                        "path": issue.path,
                        "message": issue.message,
                        "table": issue.table,
                        "row_id": issue.row_id,
                    }
                    for issue in self.issues
                ],
            },
            "validation_msa_failures": [
                {
                    "msa_job_id": failure.msa_job_id,
                    "scope": failure.scope,
                    "candidate_ids": list(failure.candidate_ids),
                    "error_message": failure.error_message,
                }
                for failure in self.validation_msa_failures
            ],
        }


def inspect_campaign(campaign_dir: str | Path) -> CampaignStatus:
    """Read SQLite and artifact paths, then report obvious inconsistencies."""

    root = Path(campaign_dir)
    conn = connect_database(root / "campaign.sqlite")
    try:
        referenced_paths: set[str] = set()
        issues: list[ReconciliationIssue] = []

        for row in conn.execute(
            """
            SELECT candidate_id, sequence_path
            FROM candidates
            WHERE sequence_path IS NOT NULL
            """
        ):
            relpath = row["sequence_path"]
            referenced_paths.add(relpath)
            if not (root / relpath).exists():
                issues.append(
                    ReconciliationIssue(
                        kind="missing_sequence_artifact",
                        path=relpath,
                        table="candidates",
                        row_id=row["candidate_id"],
                        message="candidate row references a missing sequence file",
                    )
                )

        for row in conn.execute(
            """
            SELECT candidate_id, critic_name, structure_path
            FROM critic_metrics
            WHERE structure_path IS NOT NULL
            """
        ):
            relpath = row["structure_path"]
            referenced_paths.add(relpath)
            if not (root / relpath).exists():
                issues.append(
                    ReconciliationIssue(
                        kind="missing_structure_artifact",
                        path=relpath,
                        table="critic_metrics",
                        row_id=f"{row['candidate_id']}:{row['critic_name']}",
                        message="critic row references a missing structure file",
                    )
                )

        if _table_exists(conn, "validation_tasks"):
            for row in conn.execute(
                """
                SELECT validation_id, output_structure_path
                FROM validation_tasks
                WHERE output_structure_path IS NOT NULL
                """
            ):
                relpath = row["output_structure_path"]
                referenced_paths.add(relpath)
                if not (root / relpath).exists():
                    issues.append(
                        ReconciliationIssue(
                            kind="missing_validation_artifact",
                            path=relpath,
                            table="validation_tasks",
                            row_id=row["validation_id"],
                            message=(
                                "validation task row references a missing "
                                "structure file"
                            ),
                        )
                    )

        if _table_exists(conn, "validation_structures"):
            for row in conn.execute(
                """
                SELECT validation_id, structure_id, structure_path
                FROM validation_structures
                """
            ):
                relpath = row["structure_path"]
                referenced_paths.add(relpath)
                if not (root / relpath).exists():
                    issues.append(
                        ReconciliationIssue(
                            kind="missing_validation_artifact",
                            path=relpath,
                            table="validation_structures",
                            row_id=f"{row['validation_id']}:{row['structure_id']}",
                            message=(
                                "validation structure row references a missing "
                                "structure file"
                            ),
                        )
                    )

        for relpath in _iter_shard_artifacts(root):
            if relpath not in referenced_paths:
                issues.append(
                    ReconciliationIssue(
                        kind="untracked_artifact",
                        path=relpath,
                        message="artifact exists under shards/ but is not referenced by SQLite",
                    )
                )

        campaign_row = conn.execute(
            """
            SELECT schema_version, config_hash, resolved_config_json, updated_at
            FROM campaign
            WHERE id = 1
            """
        ).fetchone()
        resolved_config = _json_object(
            campaign_row["resolved_config_json"] if campaign_row else None
        )
        validation_configured = isinstance(resolved_config.get("validation"), dict)
        shard_counts = _status_counts(conn, "shards")
        candidate_counts = _status_counts(conn, "candidates")
        critic_counts = _status_counts(conn, "critic_metrics")
        validation_counts = _status_counts(conn, "validation_tasks")
        validation_structure_counts = _status_counts(conn, "validation_structures")
        validation_msa_counts = _status_counts(conn, "validation_msa_jobs")
        attempt_counts = _status_counts(conn, "attempts")
        expected_artifacts = _expected_artifacts(root, validation_configured)
        terminal_failure_count = _terminal_failure_count(
            shard_counts=shard_counts,
            candidate_counts=candidate_counts,
            critic_counts=critic_counts,
            validation_counts=validation_counts,
            validation_msa_counts=validation_msa_counts,
        )

        return CampaignStatus(
            schema_version=1,
            campaign_db_schema_version=(
                int(campaign_row["schema_version"]) if campaign_row else None
            ),
            config_hash=(str(campaign_row["config_hash"]) if campaign_row else None),
            state=_campaign_state(
                shard_counts=shard_counts,
                candidate_counts=candidate_counts,
                critic_counts=critic_counts,
                validation_counts=validation_counts,
                validation_structure_counts=validation_structure_counts,
                validation_msa_counts=validation_msa_counts,
                validation_configured=validation_configured,
                terminal_failure_count=terminal_failure_count,
                expected_artifacts=expected_artifacts,
                issues=issues,
            ),
            latest_activity_at=_latest_activity_at(conn),
            validation_configured=validation_configured,
            terminal_failure_count=terminal_failure_count,
            expected_artifacts=expected_artifacts,
            table_counts=_table_counts(conn),
            shard_status_counts=shard_counts,
            candidate_status_counts=candidate_counts,
            critic_status_counts=critic_counts,
            validation_status_counts=validation_counts,
            validation_structure_status_counts=validation_structure_counts,
            validation_msa_status_counts=validation_msa_counts,
            validation_msa_blocked_counts=_validation_msa_blocked_counts(conn),
            validation_msa_failures=_validation_msa_failures(conn),
            attempt_status_counts=attempt_counts,
            issues=sorted(issues, key=lambda issue: (issue.kind, issue.path)),
        )
    finally:
        conn.close()


def _table_counts(conn) -> dict[str, int]:
    tables = [
        "shards",
        "candidates",
        "critic_metrics",
        "validation_tasks",
        "validation_structures",
        "attempts",
    ]
    return {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in tables
        if _table_exists(conn, table)
    }


def _status_counts(conn, table: str) -> dict[str, int]:
    if not _table_exists(conn, table):
        return {}
    return {
        row["status"]: int(row["count"])
        for row in conn.execute(
            f"""
            SELECT status, COUNT(*) AS count
            FROM {table}
            GROUP BY status
            ORDER BY status
            """
        )
    }


def _validation_msa_blocked_counts(conn) -> dict[str, int]:
    if not (
        _table_exists(conn, "validation_tasks")
        and _table_exists(conn, "validation_msa_jobs")
        and _table_exists(conn, "validation_msa_job_candidates")
    ):
        return {}
    return {
        row["status"]: int(row["count"])
        for row in conn.execute(
            """
            SELECT job.status, COUNT(DISTINCT task.validation_id) AS count
            FROM validation_tasks AS task
            JOIN validation_msa_job_candidates AS dep
              ON dep.candidate_id = task.candidate_id
            JOIN validation_msa_jobs AS job
              ON job.msa_job_id = dep.msa_job_id
            WHERE task.status = 'pending'
              AND dep.validation_config_hash = task.validation_config_hash
              AND job.status NOT IN ('ready', 'skipped')
            GROUP BY job.status
            ORDER BY job.status
            """
        )
    }


def _validation_msa_failures(conn) -> list[MsaFailureSummary]:
    if not (
        _table_exists(conn, "validation_msa_jobs")
        and _table_exists(conn, "validation_msa_job_candidates")
    ):
        return []
    rows = conn.execute(
        """
        SELECT
            job.msa_job_id,
            job.scope,
            job.error_message,
            GROUP_CONCAT(dep.candidate_id, ',') AS candidate_ids
        FROM validation_msa_jobs AS job
        LEFT JOIN validation_msa_job_candidates AS dep
          ON dep.msa_job_id = job.msa_job_id
        WHERE job.status = 'failed'
        GROUP BY job.msa_job_id, job.scope, job.error_message
        ORDER BY job.completed_at DESC, job.msa_job_id
        """
    ).fetchall()
    return [
        MsaFailureSummary(
            msa_job_id=row["msa_job_id"],
            scope=row["scope"],
            candidate_ids=tuple(
                cid
                for cid in str(row["candidate_ids"] or "").split(",")
                if cid
            ),
            error_message=row["error_message"] or "",
        )
        for row in rows
    ]


def _table_exists(conn, table: str) -> bool:
    return (
        conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            """,
            (table,),
        ).fetchone()
        is not None
    )


def _json_object(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _expected_artifacts(root: Path, validation_configured: bool) -> dict[str, bool]:
    artifacts = {
        "config": (root / "config.yaml").is_file(),
        "resolved_config": (root / "resolved_config.yaml").is_file(),
        "campaign_database": (root / "campaign.sqlite").is_file(),
        "esmfold2_metrics": (root / "esmfold2" / "metrics_all.csv").is_file(),
        "esmfold2_selection": (
            root / "esmfold2" / "selected_designs.csv"
        ).is_file(),
    }
    if validation_configured:
        artifacts.update(
            {
                "validation_results": (
                    root
                    / "validation"
                    / "protenix_v2"
                    / "validation_results.csv"
                ).is_file(),
                "combined_ranking": (
                    root / "ranked_results" / "combined_ranking.csv"
                ).is_file(),
                "ranking_summary": (
                    root / "ranked_results" / "ranking_summary.json"
                ).is_file(),
            }
        )
    return artifacts


def _terminal_failure_count(
    *,
    shard_counts: dict[str, int],
    candidate_counts: dict[str, int],
    critic_counts: dict[str, int],
    validation_counts: dict[str, int],
    validation_msa_counts: dict[str, int],
) -> int:
    return sum(
        (
            shard_counts.get("failed", 0),
            shard_counts.get("cancelled", 0),
            candidate_counts.get("failed", 0),
            critic_counts.get("failed", 0),
            validation_counts.get("failed", 0),
            validation_msa_counts.get("failed", 0),
        )
    )


def _campaign_state(
    *,
    shard_counts: dict[str, int],
    candidate_counts: dict[str, int],
    critic_counts: dict[str, int],
    validation_counts: dict[str, int],
    validation_structure_counts: dict[str, int],
    validation_msa_counts: dict[str, int],
    validation_configured: bool,
    terminal_failure_count: int,
    expected_artifacts: dict[str, bool],
    issues: list[ReconciliationIssue],
) -> str:
    if terminal_failure_count:
        return "failed"

    status_groups = (
        shard_counts,
        candidate_counts,
        critic_counts,
        validation_counts,
        validation_structure_counts,
        validation_msa_counts,
    )
    if any(group.get("running", 0) for group in status_groups):
        return "running"
    if any(group.get("pending", 0) for group in status_groups):
        return "pending"
    if not sum(shard_counts.values()):
        return "unplanned"

    if issues:
        return "inconsistent"

    core_complete = (
        shard_counts.get("completed", 0) == sum(shard_counts.values())
        and not candidate_counts.get("failed", 0)
        and not critic_counts.get("failed", 0)
    )
    if not core_complete:
        return "finalizing"

    if validation_configured:
        validation_terminal = not any(
            validation_counts.get(status, 0) for status in ("pending", "running")
        )
        msa_terminal = not any(
            validation_msa_counts.get(status, 0)
            for status in ("pending", "running", "failed")
        )
        required_keys = (
            "validation_results",
            "combined_ranking",
            "ranking_summary",
        )
        if (
            validation_terminal
            and msa_terminal
            and all(expected_artifacts.get(key, False) for key in required_keys)
        ):
            return "complete"
        return "finalizing"

    if all(
        expected_artifacts.get(key, False)
        for key in ("esmfold2_metrics", "esmfold2_selection")
    ):
        return "complete"
    return "finalizing"


def _latest_activity_at(conn) -> str | None:
    queries = (
        "SELECT MAX(updated_at) FROM campaign",
        "SELECT MAX(COALESCE(completed_at, heartbeat_at, started_at, created_at)) FROM shards",
        "SELECT MAX(COALESCE(completed_at, started_at, created_at)) FROM candidates",
        "SELECT MAX(COALESCE(completed_at, started_at, created_at)) FROM critic_metrics",
        "SELECT MAX(COALESCE(completed_at, heartbeat_at, started_at, created_at)) FROM validation_tasks",
        "SELECT MAX(created_at) FROM validation_structures",
        "SELECT MAX(COALESCE(completed_at, heartbeat_at, started_at, created_at)) FROM validation_msa_jobs",
        "SELECT MAX(COALESCE(ended_at, started_at)) FROM attempts",
    )
    values = []
    for query in queries:
        try:
            row = conn.execute(query).fetchone()
        except Exception:
            continue
        if row and row[0]:
            values.append(str(row[0]))
    return max(values) if values else None


def _iter_shard_artifacts(root: Path) -> list[str]:
    shard_root = root / "shards"
    if not shard_root.exists():
        return []

    relpaths: list[str] = []
    for path in shard_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.endswith(".tmp"):
            continue
        relpaths.append(path.relative_to(root).as_posix())
    return sorted(relpaths)
