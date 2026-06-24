from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from esmfold2_pipeline.db.store import SCHEMA_VERSION, initialize_database


class CampaignSchemaTest(unittest.TestCase):
    def test_initialize_database_creates_minimal_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "campaign.sqlite"
            conn = initialize_database(
                db_path,
                config_hash="test-config",
                resolved_config={"campaign": {"num_designs": 1}},
                software_versions={"esmfold2_pipeline": "0.1.0"},
            )

            tables = {
                row["name"]
                for row in conn.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                    """
                )
            }
            self.assertTrue(
                {
                    "campaign",
                    "shards",
                    "candidates",
                    "critic_metrics",
                    "validation_tasks",
                    "validation_structures",
                    "validation_msa_jobs",
                    "validation_msa_job_candidates",
                    "msa_rate_limits",
                    "attempts",
                }.issubset(tables)
            )

            campaign = conn.execute("SELECT * FROM campaign WHERE id = 1").fetchone()
            self.assertEqual(campaign["schema_version"], SCHEMA_VERSION)
            self.assertEqual(campaign["config_hash"], "test-config")
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 5)
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            dep_columns = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(validation_msa_job_candidates)"
                )
            }
            self.assertIn("validation_config_hash", dep_columns)

    def test_core_relationships_and_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = initialize_database(Path(tmpdir) / "campaign.sqlite")

            conn.execute(
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
                """,
                (
                    "shard_000000",
                    0,
                    0,
                    "target:sequence",
                    "binder:miniprotein:length=60-200",
                    '["ESMFold2-Experimental-Fast"]',
                ),
            )
            conn.execute(
                """
                INSERT INTO candidates (
                    candidate_id,
                    shard_id,
                    candidate_index,
                    designed_sequence,
                    binder_chain_id,
                    sequence_path
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "cand_00000000",
                    "shard_000000",
                    0,
                    "ACDEFGHIK",
                    "B",
                    "shards/shard_000000/candidates/cand_00000000/sequence.fasta",
                ),
            )
            candidate = conn.execute(
                "SELECT binder_chain_id FROM candidates WHERE candidate_id = ?",
                ("cand_00000000",),
            ).fetchone()
            self.assertEqual(candidate["binder_chain_id"], "B")
            conn.execute(
                """
                INSERT INTO critic_metrics (
                    candidate_id,
                    critic_name,
                    status,
                    structure_path,
                    iptm,
                    distogram_iptm_proxy
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "cand_00000000",
                    "ESMFold2-Experimental-Fast",
                    "completed",
                    (
                        "shards/shard_000000/candidates/cand_00000000/"
                        "critics/ESMFold2-Experimental-Fast/complex.pdb"
                    ),
                    0.72,
                    0.68,
                ),
            )
            conn.execute(
                """
                INSERT INTO validation_tasks (
                    validation_id,
                    candidate_id,
                    model_name,
                    validation_config_hash,
                    selection_rank
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "val_cand_00000000_test",
                    "cand_00000000",
                    "protenix-v2",
                    "hash",
                    1,
                ),
            )
            conn.execute(
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
                    scoped_iptm,
                    scoped_ipsae
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "val_cand_00000000_test",
                    "seed101_sample0",
                    "cand_00000000",
                    "protenix-v2",
                    101,
                    0,
                    "passing",
                    "validation/protenix_v2/structures/passing/cand_00000000.cif",
                    0.81,
                    0.62,
                ),
            )
            conn.execute(
                """
                INSERT INTO attempts (
                    shard_id,
                    candidate_id,
                    critic_name,
                    stage,
                    status,
                    worker_id,
                    hostname,
                    pid,
                    gpu_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "shard_000000",
                    "cand_00000000",
                    "ESMFold2-Experimental-Fast",
                    "critic",
                    "completed",
                    "worker-1",
                    "localhost",
                    1234,
                    "0",
                ),
            )
            conn.commit()

            critic = conn.execute(
                """
                SELECT *
                FROM critic_metrics
                WHERE candidate_id = ? AND critic_name = ?
                """,
                ("cand_00000000", "ESMFold2-Experimental-Fast"),
            ).fetchone()
            self.assertEqual(critic["status"], "completed")
            self.assertEqual(critic["iptm"], 0.72)

            validation = conn.execute(
                """
                SELECT *
                FROM validation_structures
                WHERE validation_id = ? AND structure_id = ?
                """,
                ("val_cand_00000000_test", "seed101_sample0"),
            ).fetchone()
            self.assertEqual(validation["status"], "passing")
            self.assertEqual(validation["scoped_iptm"], 0.81)

            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO candidates (
                        candidate_id,
                        shard_id,
                        candidate_index,
                        designed_sequence
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    ("cand_missing_parent", "missing_shard", 0, "ACD"),
                )

            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO shards (shard_id, seed, batch_index, status)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("shard_bad_status", 1, 0, "unknown"),
                )

    def test_initialize_database_migrates_unscoped_msa_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "campaign.sqlite"
            conn = initialize_database(db_path)
            conn.execute(
                """
                INSERT INTO shards (
                    shard_id,
                    seed,
                    batch_index,
                    target_key,
                    binder_key,
                    critic_set_json
                )
                VALUES ('shard_000000', 0, 0, 'target', 'binder', '[]')
                """
            )
            conn.execute(
                """
                INSERT INTO candidates (
                    candidate_id,
                    shard_id,
                    candidate_index,
                    designed_sequence
                )
                VALUES ('cand_000000', 'shard_000000', 0, 'ACD')
                """
            )
            conn.execute(
                """
                INSERT INTO validation_msa_jobs (
                    msa_job_id,
                    scope,
                    cache_key,
                    msa_context_hash
                )
                VALUES ('msa_old', 'target', 'target:old', 'ctx')
                """
            )
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("DROP TABLE validation_msa_job_candidates")
            conn.execute(
                """
                CREATE TABLE validation_msa_job_candidates (
                    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
                    msa_job_id TEXT NOT NULL REFERENCES validation_msa_jobs(msa_job_id) ON DELETE CASCADE,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    PRIMARY KEY(candidate_id, msa_job_id)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO validation_msa_job_candidates (
                    candidate_id,
                    msa_job_id,
                    reason
                )
                VALUES ('cand_000000', 'msa_old', 'target:A')
                """
            )
            conn.commit()
            conn.close()

            migrated = initialize_database(db_path)
            try:
                columns = {
                    row["name"]
                    for row in migrated.execute(
                        "PRAGMA table_info(validation_msa_job_candidates)"
                    )
                }
                self.assertIn("validation_config_hash", columns)
                dep = migrated.execute(
                    """
                    SELECT validation_config_hash, reason
                    FROM validation_msa_job_candidates
                    WHERE candidate_id = 'cand_000000'
                      AND msa_job_id = 'msa_old'
                    """
                ).fetchone()
                self.assertEqual(dep["validation_config_hash"], "")
                self.assertEqual(dep["reason"], "target:A")
            finally:
                migrated.close()


if __name__ == "__main__":
    unittest.main()
