PRAGMA user_version = 5;

CREATE TABLE IF NOT EXISTS campaign (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    config_hash TEXT NOT NULL,
    resolved_config_json TEXT NOT NULL DEFAULT '{}',
    software_versions_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS shards (
    shard_id TEXT PRIMARY KEY,
    campaign_id INTEGER NOT NULL DEFAULT 1 REFERENCES campaign(id) ON DELETE CASCADE,
    seed INTEGER NOT NULL,
    batch_index INTEGER NOT NULL CHECK (batch_index >= 0),
    target_key TEXT NOT NULL DEFAULT '',
    binder_key TEXT NOT NULL DEFAULT '',
    critic_set_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'cancelled')
    ),
    claim_worker_id TEXT,
    claim_hostname TEXT,
    claim_pid INTEGER,
    claim_gpu_id TEXT,
    claimed_at TEXT,
    heartbeat_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    UNIQUE (campaign_id, seed, batch_index, target_key, binder_key)
);

CREATE INDEX IF NOT EXISTS idx_shards_status
    ON shards(status, heartbeat_at);

CREATE INDEX IF NOT EXISTS idx_shards_claim_worker
    ON shards(claim_worker_id);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL REFERENCES shards(shard_id) ON DELETE CASCADE,
    candidate_index INTEGER NOT NULL CHECK (candidate_index >= 0),
    designed_sequence TEXT NOT NULL,
    binder_chain_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'skipped')
    ),
    sequence_path TEXT,
    design_metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    UNIQUE (shard_id, candidate_index)
);

CREATE INDEX IF NOT EXISTS idx_candidates_shard
    ON candidates(shard_id);

CREATE INDEX IF NOT EXISTS idx_candidates_status
    ON candidates(status);

CREATE INDEX IF NOT EXISTS idx_candidates_sequence
    ON candidates(designed_sequence);

CREATE TABLE IF NOT EXISTS critic_metrics (
    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    critic_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'skipped')
    ),
    structure_path TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    iptm REAL,
    ptm REAL,
    plddt REAL,
    distogram_iptm_proxy REAL,
    hotspot_satisfaction REAL,
    runtime_seconds REAL CHECK (runtime_seconds IS NULL OR runtime_seconds >= 0),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    PRIMARY KEY (candidate_id, critic_name)
);

CREATE INDEX IF NOT EXISTS idx_critic_metrics_status
    ON critic_metrics(status);

CREATE INDEX IF NOT EXISTS idx_critic_metrics_critic
    ON critic_metrics(critic_name);

CREATE INDEX IF NOT EXISTS idx_critic_metrics_rank
    ON critic_metrics(iptm DESC, distogram_iptm_proxy DESC);

CREATE TABLE IF NOT EXISTS validation_tasks (
    validation_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    validation_config_hash TEXT NOT NULL,
    selection_rank INTEGER CHECK (selection_rank IS NULL OR selection_rank > 0),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'skipped')
    ),
    claim_worker_id TEXT,
    claim_hostname TEXT,
    claim_pid INTEGER,
    claim_gpu_id TEXT,
    claimed_at TEXT,
    heartbeat_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    started_at TEXT,
    completed_at TEXT,
    output_structure_path TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    iptm REAL,
    ipsae REAL,
    ptm REAL,
    ranking_score REAL,
    hotspot_satisfaction REAL,
    runtime_seconds REAL CHECK (runtime_seconds IS NULL OR runtime_seconds >= 0),
    error_message TEXT,
    UNIQUE(candidate_id, model_name, validation_config_hash)
);

CREATE INDEX IF NOT EXISTS idx_validation_tasks_status
    ON validation_tasks(status, heartbeat_at);

CREATE INDEX IF NOT EXISTS idx_validation_tasks_candidate
    ON validation_tasks(candidate_id);

CREATE INDEX IF NOT EXISTS idx_validation_tasks_rank
    ON validation_tasks(selection_rank, iptm DESC, ipsae DESC);

CREATE TABLE IF NOT EXISTS validation_structures (
    validation_id TEXT NOT NULL REFERENCES validation_tasks(validation_id) ON DELETE CASCADE,
    structure_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    sample_rank INTEGER NOT NULL CHECK (sample_rank >= 0),
    status TEXT NOT NULL CHECK (status IN ('pending', 'passing', 'rejected')),
    structure_path TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    scoped_iptm REAL,
    scoped_ipsae REAL,
    ptm REAL,
    ranking_score REAL,
    hotspot_satisfaction REAL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (validation_id, structure_id)
);

CREATE INDEX IF NOT EXISTS idx_validation_structures_candidate
    ON validation_structures(candidate_id);

CREATE INDEX IF NOT EXISTS idx_validation_structures_status
    ON validation_structures(status);

CREATE TABLE IF NOT EXISTS validation_msa_jobs (
    msa_job_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL CHECK (
        scope IN (
            'target',
            'vhh_binder_group',
            'scfv_binder_group',
            'miniprotein_single_sequence'
        )
    ),
    cache_key TEXT NOT NULL,
    msa_context_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'running', 'ready', 'failed', 'skipped')
    ),
    claim_worker_id TEXT,
    claim_hostname TEXT,
    claim_pid INTEGER,
    claimed_at TEXT,
    heartbeat_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    next_eligible_at TEXT,
    representative_sequence TEXT,
    member_sequences_json TEXT NOT NULL DEFAULT '[]',
    cache_paths_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    UNIQUE(scope, cache_key, msa_context_hash)
);

CREATE INDEX IF NOT EXISTS idx_validation_msa_jobs_status
    ON validation_msa_jobs(status, next_eligible_at, heartbeat_at);

CREATE TABLE IF NOT EXISTS validation_msa_job_candidates (
    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
    msa_job_id TEXT NOT NULL REFERENCES validation_msa_jobs(msa_job_id) ON DELETE CASCADE,
    validation_config_hash TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY(candidate_id, msa_job_id, validation_config_hash)
);

CREATE INDEX IF NOT EXISTS idx_validation_msa_job_candidates_job
    ON validation_msa_job_candidates(msa_job_id);

CREATE TABLE IF NOT EXISTS msa_rate_limits (
    name TEXT PRIMARY KEY,
    last_submit_at TEXT
);

CREATE TABLE IF NOT EXISTS attempts (
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
    CHECK (shard_id IS NOT NULL OR candidate_id IS NOT NULL OR validation_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_attempts_shard
    ON attempts(shard_id);

CREATE INDEX IF NOT EXISTS idx_attempts_candidate
    ON attempts(candidate_id);

CREATE INDEX IF NOT EXISTS idx_attempts_status
    ON attempts(status, started_at);

CREATE INDEX IF NOT EXISTS idx_attempts_worker
    ON attempts(worker_id);
