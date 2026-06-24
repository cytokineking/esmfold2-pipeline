from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

        return CampaignStatus(
            table_counts=_table_counts(conn),
            shard_status_counts=_status_counts(conn, "shards"),
            candidate_status_counts=_status_counts(conn, "candidates"),
            critic_status_counts=_status_counts(conn, "critic_metrics"),
            validation_status_counts=_status_counts(conn, "validation_tasks"),
            validation_structure_status_counts=_status_counts(
                conn,
                "validation_structures",
            ),
            validation_msa_status_counts=_status_counts(conn, "validation_msa_jobs"),
            validation_msa_blocked_counts=_validation_msa_blocked_counts(conn),
            validation_msa_failures=_validation_msa_failures(conn),
            attempt_status_counts=_status_counts(conn, "attempts"),
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
