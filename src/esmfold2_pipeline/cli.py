from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, contextmanager, nullcontext
import os
import shlex
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml

from esmfold2_pipeline.artifacts import write_text_atomic
from esmfold2_pipeline.config import (
    BINDER_TARGET_CONTACT_MODES,
    DEFAULT_ESMFOLD2_CRITIC_MODEL,
    DEFAULT_ESMFOLD2_INVERSION_MODEL,
    ESMFOLD2_MODEL_ALIASES,
    MOSAIC_FRAMEWORK_CONTACT_PENALTY_SCOPES,
    ConfigCheckResult,
    check_campaign_config,
    resolve_esmfold2_model_name,
)
from esmfold2_pipeline.db import CampaignStore, connect_database
from esmfold2_pipeline.esm_adapter import check_environment, preflight_models
from esmfold2_pipeline.execution import (
    plan_one_gpu_smoke_shard,
    plan_one_mock_shard,
    run_campaign,
    run_multi_campaign,
    run_one_gpu_smoke_shard,
    run_one_mock_shard,
)
from esmfold2_pipeline.frameworks import (
    all_scfv_framework_names,
    all_vhh_framework_names,
    get_scfv_framework_template_cif,
    get_vhh_framework_template_cif,
    resolve_scfv_framework_name,
    resolve_vhh_framework_name,
    scfv_framework_alias_choices,
    vhh_framework_alias_choices,
)
from esmfold2_pipeline.planning import plan_campaign
from esmfold2_pipeline.reports import (
    CampaignStatus,
    aggregate_campaign,
    analyze_campaign,
    export_campaign,
    inspect_campaign,
    report_validation,
    select_campaign,
)
from esmfold2_pipeline.validation import (
    DEFAULT_MSA_MAX_REQUESTS_PER_MINUTE,
    DEFAULT_VALIDATE_MODEL,
    DEFAULT_PROTENIX_TOKEN_LIMIT,
    DEFAULT_VALIDATION_BINDER_LENGTH,
    ProtenixRunnerConfig,
    ValidationPlanConfig,
    default_validation_binder_sequence,
    check_protenix_environment,
    plan_msa_prefetch,
    plan_validation_tasks,
    run_local_protenix_validation,
    run_msa_prefetch_worker,
    run_mock_validation,
    run_multi_validation,
    validate_conditioning_config,
)
from esmfold2_pipeline.validation.workers import (
    _normalize_gpu_ids as _normalize_validation_gpu_ids,
)


_LOCAL_RUNTIME_CACHE_DISABLE_ENV = "ESMFOLD2_PIPELINE_DISABLE_LOCAL_RUNTIME_CACHE"
DEFAULT_MIN_IPTM = 0.6
DEFAULT_MAX_DESIGNS = 100


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="esmfold2-pipeline",
        description="Campaign orchestration for ESMFold2-native binder design.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_mock = subparsers.add_parser(
        "plan-mock",
        help="create the deterministic one-shard mock campaign",
    )
    plan_mock.add_argument("campaign_dir", type=Path)
    plan_mock.set_defaults(func=_plan_mock)

    plan = subparsers.add_parser(
        "plan",
        help="create a campaign from a minimal YAML config",
    )
    plan.add_argument("config", type=Path)
    plan.add_argument("--out", type=Path, default=None)
    plan.set_defaults(func=_plan)

    check = subparsers.add_parser(
        "check",
        help="validate a campaign config without planning it",
    )
    check.add_argument("config", type=Path)
    check.add_argument("--out", type=Path, default=None)
    check.add_argument("--esm-repo", type=Path, default=None)
    check.add_argument(
        "--env",
        action="store_true",
        help="also check optional ESM/Torch/CUDA dependencies",
    )
    check.add_argument(
        "--models",
        action="store_true",
        help="also load the configured model set",
    )
    check.add_argument("--gpu-id", default=None)
    check.add_argument(
        "--enable-hf-xet",
        action="store_true",
        help="do not force HF_HUB_DISABLE_XET=1 during model preflight",
    )
    check.set_defaults(func=_check_config)

    check_env = subparsers.add_parser(
        "check-env",
        help="check optional ESM/Torch/CUDA dependencies",
    )
    check_env.add_argument("--esm-repo", type=Path, default=None)
    check_env.add_argument(
        "--no-tutorial",
        action="store_true",
        help="do not require cookbook/tutorials/binder_design.py",
    )
    check_env.add_argument(
        "--local-runtime",
        action="store_true",
        help="require local ESMFold2 runtime APIs explicitly",
    )
    check_env.add_argument("--json", action="store_true", help="print JSON output")
    check_env.set_defaults(func=_check_env)

    check_models = subparsers.add_parser(
        "check-models",
        help="load the model set used by the GPU smoke command",
        description="load the model set used by the GPU smoke command",
    )
    check_models.add_argument("--esm-repo", type=Path, default=None)
    check_models.add_argument("--gpu-id", default=None)
    check_models.add_argument(
        "--model",
        default=None,
        help=_model_help(),
    )
    check_models.add_argument(
        "--inversion-model-name", default=DEFAULT_ESMFOLD2_INVERSION_MODEL
    )
    check_models.add_argument("--critic-name", default=DEFAULT_ESMFOLD2_CRITIC_MODEL)
    check_models.add_argument("--steps", type=int, default=1)
    check_models.add_argument(
        "--enable-hf-xet",
        action="store_true",
        help="do not force HF_HUB_DISABLE_XET=1 during model preflight",
    )
    check_models.set_defaults(func=_check_models)

    check_protenix = subparsers.add_parser(
        "check-protenix",
        help="check optional Protenix validation dependencies",
    )
    _add_check_protenix_arguments(check_protenix)
    check_protenix.set_defaults(func=_check_protenix)

    validate_check_env = subparsers.add_parser(
        "validate-check-env",
        help="alias for check-protenix",
    )
    _add_check_protenix_arguments(validate_check_env)
    validate_check_env.set_defaults(func=_check_protenix)

    launch = subparsers.add_parser(
        "launch",
        help="run or resume a campaign end to end",
        description=(
            "validate, plan, run, aggregate, select, export, and optionally "
            "validate/analyze a campaign from config.yaml; resume an existing "
            "campaign directory; or generate a minimal campaign from explicit "
            "target input flags"
        ),
    )
    launch.add_argument("config", type=Path, nargs="?")
    launch.add_argument("--out", type=Path, default=None)
    launch.add_argument("--esm-repo", type=Path, default=None)
    launch.add_argument("--gpu-id", default=None)
    launch.add_argument(
        "--gpus",
        nargs="+",
        default=None,
        help="GPU ids as comma-separated values, ranges like 0-3, or all",
    )
    launch.add_argument("--worker-prefix", default="local-gpu")
    launch.add_argument("--max-shards", type=int, default=None)
    launch.add_argument("--max-shards-per-worker", type=int, default=None)
    launch.add_argument("--poll-interval", type=float, default=2.0)
    launch.add_argument("--heartbeat-interval", type=float, default=30.0)
    launch.add_argument(
        "--validation-msa-workers",
        type=int,
        default=None,
        help=(
            "background MSA workers to run during launch; default is 1 when "
            "validation config is present, otherwise 0"
        ),
    )
    launch.add_argument(
        "--validation-msa-poll-interval",
        type=float,
        default=5.0,
        help="seconds launch MSA workers wait before polling for newly queued jobs",
    )
    launch.add_argument(
        "--msa-max-requests-per-minute",
        type=float,
        default=DEFAULT_MSA_MAX_REQUESTS_PER_MINUTE,
        help="campaign-wide ColabFold/MMseqs submission rate limit",
    )
    launch.add_argument(
        "--stale-timeout",
        type=float,
        default=None,
        help=(
            "seconds after the last shard heartbeat before a running shard is "
            "automatically recovered for resume; default max(90, 3*heartbeat)"
        ),
    )
    launch.add_argument(
        "--enable-hf-xet",
        action="store_true",
        help="do not force HF_HUB_DISABLE_XET=1 during model loading",
    )
    launch.add_argument(
        "--disable-local-runtime-cache",
        action="store_true",
        default=False,
        help="reload local ESMFold2/ESMC runtime for each design shard",
    )
    launch.add_argument(
        "--target-name",
        default=None,
        help="value for generated YAML target.name",
    )
    launch.add_argument(
        "--target-sequence",
        default=None,
        help="single-chain target protein sequence for no-YAML launch",
    )
    launch.add_argument(
        "--target-structure",
        type=Path,
        default=None,
        help="PDB/mmCIF target structure path for no-YAML launch",
    )
    launch.add_argument(
        "--chains",
        nargs="+",
        default=None,
        help="target structure chains to include, e.g. A,C or A C",
    )
    launch.add_argument(
        "--hotspot",
        action="append",
        default=None,
        help="target structure hotspot selector, e.g. A:88,91; can repeat",
    )
    launch.add_argument(
        "--scaffold",
        choices=("miniprotein", "scfv", "vhh"),
        default=None,
        help="binder scaffold for no-YAML launch",
    )
    launch.add_argument(
        "--frameworks",
        nargs="+",
        default=None,
        help=(
            "bundled antibody framework aliases/names, comma/space separated, "
            "or all for the selected antibody scaffold"
        ),
    )
    launch.add_argument(
        "--length",
        default=None,
        help="miniprotein length or range for no-YAML launch, e.g. 80-140",
    )
    launch.add_argument(
        "--num-designs",
        type=int,
        default=None,
        help="number of designs for no-YAML launch",
    )
    launch.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="first deterministic seed for no-YAML launch",
    )
    launch.add_argument(
        "--model",
        default=None,
        help=_model_help(),
    )
    launch.add_argument(
        "--steps",
        type=int,
        default=None,
        help="gradient optimization steps for no-YAML launch",
    )
    launch.add_argument(
        "--binder-target-contact-mode",
        choices=tuple(sorted(BINDER_TARGET_CONTACT_MODES)),
        default=None,
        help="binder-target contact loss mode for no-YAML launch",
    )
    launch.add_argument(
        "--mosaic-cdr-contact-weight",
        type=float,
        default=None,
        help="weight for Mosaic-style CDR contact loss in no-YAML launch",
    )
    launch.add_argument(
        "--mosaic-cdr-contact-cutoff-angstrom",
        type=float,
        default=None,
        help="distogram contact cutoff for Mosaic-style CDR contact loss",
    )
    launch.add_argument(
        "--mosaic-cdr-num-target-contacts",
        type=int,
        default=None,
        help="target contacts averaged per CDR residue for Mosaic-style loss",
    )
    launch.add_argument(
        "--mosaic-framework-contact-penalty-weight",
        type=float,
        default=None,
        help="optional framework contact penalty weight for Mosaic-style loss",
    )
    launch.add_argument(
        "--mosaic-framework-contact-penalty-scope",
        choices=tuple(sorted(MOSAIC_FRAMEWORK_CONTACT_PENALTY_SCOPES)),
        default=None,
        help="target scope for optional Mosaic-style framework contact penalty",
    )
    launch.add_argument(
        "--max-designs",
        type=int,
        default=DEFAULT_MAX_DESIGNS,
        help="maximum ranked designs to select after launch",
    )
    launch.add_argument(
        "--min-iptm",
        type=float,
        default=None,
        help=(
            "drop designs below this ESMFold2 ipTM during launch selection and "
            "Protenix validation; default 0.6, set 0 to disable"
        ),
    )
    launch.add_argument(
        "--require-hotspot-contact",
        choices=("auto", "always", "never"),
        default=None,
        help="filter selected and validated designs by final hotspot contact",
    )
    launch.add_argument(
        "--skip-export",
        action="store_true",
        default=False,
        help="skip copying selected ESMFold2 structures after launch selection",
    )
    launch.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="skip validation even when the campaign config contains validation settings",
    )
    launch.add_argument(
        "--skip-analysis",
        action="store_true",
        default=False,
        help="skip validation analysis ranking after launch validation",
    )
    launch.add_argument(
        "--analysis-top-k",
        type=int,
        default=None,
        help="number of top validated paired structures to copy during launch analysis",
    )
    launch.add_argument(
        "--analysis-max-binder-rmsd-angstrom",
        type=float,
        default=None,
        help="override the final-ranking binder RMSD gate; default 2.5 angstrom",
    )
    launch.add_argument(
        "--analysis-rmsd-weight",
        type=float,
        default=None,
        help="override RMSD agreement weight in final ranking; default 0.10",
    )
    launch.set_defaults(func=_launch)

    validate_conditioning = subparsers.add_parser(
        "validate-conditioning",
        help="run fold-only GPU validation for target distogram conditioning",
        description="run fold-only GPU validation for target distogram conditioning",
    )
    validate_conditioning.add_argument("config", type=Path)
    validate_conditioning.add_argument("--out", type=Path, default=None)
    validate_conditioning.add_argument("--esm-repo", type=Path, default=None)
    validate_conditioning.add_argument("--gpu-id", default=None)
    validate_conditioning.add_argument("--critic-name", default=None)
    validate_conditioning.add_argument("--binder-sequence", default=None)
    validate_conditioning.add_argument(
        "--binder-length",
        type=int,
        default=DEFAULT_VALIDATION_BINDER_LENGTH,
        help=(
            "length for the deterministic validation binder when "
            "--binder-sequence is omitted"
        ),
    )
    validate_conditioning.add_argument("--num-sampling-steps", type=int, default=1)
    validate_conditioning.add_argument("--num-loops", type=int, default=0)
    validate_conditioning.add_argument("--seed", type=int, default=0)
    validate_conditioning.add_argument(
        "--calculate-confidence",
        action="store_true",
        help="ask ESMFold2 to calculate confidence heads during validation",
    )
    validate_conditioning.set_defaults(func=_validate_conditioning)

    run_mock = subparsers.add_parser(
        "run-mock",
        help="claim and run one pending mock shard",
    )
    run_mock.add_argument("campaign_dir", type=Path)
    run_mock.add_argument("--worker-id", default="mock-worker-0")
    run_mock.add_argument("--gpu-id", default=None)
    run_mock.set_defaults(func=_run_mock)

    plan_gpu = subparsers.add_parser(
        "plan-gpu-smoke",
        help="create the deterministic one-shard real GPU smoke campaign",
    )
    plan_gpu.add_argument("campaign_dir", type=Path)
    plan_gpu.add_argument("--target-name", default="ctla4")
    plan_gpu.add_argument(
        "--binder-scaffold",
        choices=("miniprotein",),
        default="miniprotein",
    )
    plan_gpu.add_argument(
        "--inversion-model-name", default=DEFAULT_ESMFOLD2_INVERSION_MODEL
    )
    plan_gpu.add_argument("--critic-name", default=DEFAULT_ESMFOLD2_CRITIC_MODEL)
    plan_gpu.set_defaults(func=_plan_gpu_smoke)

    run_gpu = subparsers.add_parser(
        "run-gpu-smoke",
        help="claim and run one pending real ESMFold2 GPU smoke shard",
    )
    run_gpu.add_argument("campaign_dir", type=Path)
    run_gpu.add_argument("--esm-repo", type=Path, default=None)
    run_gpu.add_argument("--worker-id", default="gpu-smoke-worker-0")
    run_gpu.add_argument("--gpu-id", default=None)
    run_gpu.add_argument("--steps", type=int, default=2)
    run_gpu.add_argument("--target-name", default="ctla4")
    run_gpu.add_argument(
        "--binder-scaffold",
        choices=("miniprotein",),
        default="miniprotein",
    )
    run_gpu.add_argument(
        "--inversion-model-name", default=DEFAULT_ESMFOLD2_INVERSION_MODEL
    )
    run_gpu.add_argument("--critic-name", default=DEFAULT_ESMFOLD2_CRITIC_MODEL)
    run_gpu.add_argument(
        "--enable-hf-xet",
        action="store_true",
        help="do not force HF_HUB_DISABLE_XET=1 during model loading",
    )
    run_gpu.set_defaults(func=_run_gpu_smoke)

    run = subparsers.add_parser(
        "run",
        help="run pending shards for a planned campaign on one local GPU",
    )
    run.add_argument("campaign_dir", type=Path)
    run.add_argument("--esm-repo", type=Path, default=None)
    run.add_argument("--worker-id", default="local-worker-0")
    run.add_argument("--gpu-id", default=None)
    run.add_argument("--max-shards", type=int, default=None)
    run.add_argument("--heartbeat-interval", type=float, default=30.0)
    run.add_argument(
        "--enable-hf-xet",
        action="store_true",
        help="do not force HF_HUB_DISABLE_XET=1 during model loading",
    )
    run.add_argument(
        "--disable-local-runtime-cache",
        action="store_true",
        default=False,
        help="reload local ESMFold2/ESMC runtime for each design shard",
    )
    run.add_argument(
        "--stale-timeout",
        type=float,
        default=None,
        help=(
            "seconds after the last shard heartbeat before a running shard is "
            "automatically recovered for resume; default max(90, 3*heartbeat)"
        ),
    )
    run.set_defaults(func=_run_campaign)

    run_multi = subparsers.add_parser(
        "run-multi",
        help="run one local worker process per GPU id",
        description="run one local worker process per GPU id",
    )
    run_multi.add_argument("campaign_dir", type=Path)
    run_multi.add_argument("--esm-repo", type=Path, default=None)
    run_multi.add_argument(
        "--gpus",
        nargs="+",
        required=True,
        help="GPU ids as comma-separated values, ranges like 0-3, or all",
    )
    run_multi.add_argument("--worker-prefix", default="local-gpu")
    run_multi.add_argument("--max-shards-per-worker", type=int, default=None)
    run_multi.add_argument("--poll-interval", type=float, default=2.0)
    run_multi.add_argument("--heartbeat-interval", type=float, default=30.0)
    run_multi.add_argument(
        "--validation-msa-workers",
        type=int,
        default=None,
        help=(
            "background MSA workers to run while design workers are active; "
            "default is 1, use 0 to disable"
        ),
    )
    run_multi.add_argument(
        "--validation-msa-poll-interval",
        type=float,
        default=5.0,
        help="seconds run-multi MSA workers wait before polling for newly queued jobs",
    )
    run_multi.add_argument(
        "--msa-max-requests-per-minute",
        type=float,
        default=DEFAULT_MSA_MAX_REQUESTS_PER_MINUTE,
        help="campaign-wide ColabFold/MMseqs submission rate limit",
    )
    run_multi.add_argument(
        "--stale-timeout",
        type=float,
        default=None,
        help=(
            "seconds after the last shard heartbeat before a running shard is "
            "automatically recovered for resume; default max(90, 3*heartbeat)"
        ),
    )
    run_multi.add_argument(
        "--enable-hf-xet",
        action="store_true",
        help="do not set HF_HUB_DISABLE_XET=1 in worker processes",
    )
    run_multi.add_argument(
        "--disable-local-runtime-cache",
        action="store_true",
        default=False,
        help="reload local ESMFold2/ESMC runtime for each design shard",
    )
    run_multi.set_defaults(func=_run_multi_campaign)

    status = subparsers.add_parser(
        "status",
        help="summarize SQLite state and artifact reconciliation issues",
    )
    status.add_argument("campaign_dir", type=Path)
    status.set_defaults(func=_status)

    aggregate = subparsers.add_parser(
        "aggregate",
        help="write completed critic metrics under esmfold2/",
    )
    aggregate.add_argument("campaign_dir", type=Path)
    aggregate.set_defaults(func=_aggregate)

    select = subparsers.add_parser(
        "select",
        help="rank completed designs and write esmfold2/selected_designs.csv",
    )
    select.add_argument("campaign_dir", type=Path)
    select.add_argument("--max-designs", type=int, default=DEFAULT_MAX_DESIGNS)
    select.add_argument(
        "--min-iptm",
        type=float,
        default=None,
        help="drop designs below this ipTM; default 0.6, set 0 to disable",
    )
    select.add_argument(
        "--require-hotspot-contact",
        choices=("auto", "always", "never"),
        default="auto",
        help="filter ranked designs by final heavy-atom hotspot contact",
    )
    select.set_defaults(func=_select)

    export = subparsers.add_parser(
        "export",
        help="copy selected ESMFold2 structures into esmfold2/selected_structures/",
    )
    export.add_argument("campaign_dir", type=Path)
    export.add_argument("--max-designs", type=int, default=None)
    export.set_defaults(func=_export)

    validate_plan = subparsers.add_parser(
        "validate-plan",
        help="plan optional Protenix validation tasks from completed critic rows",
    )
    validate_plan.add_argument("campaign_dir", type=Path)
    validate_plan.add_argument(
        "--validate-model",
        default=None,
        help="validation model name",
    )
    validate_plan.add_argument(
        "--validate-top-k",
        type=_parse_validate_top_k,
        default=None,
        help="number of ranked ESMFold2-passing candidates to validate, or all",
    )
    validate_plan.add_argument("--min-esm-iptm", type=float, default=None)
    validate_plan.add_argument("--min-validation-iptm", type=float, default=None)
    validate_plan.add_argument("--min-validation-ipsae", type=float, default=None)
    validate_plan.add_argument(
        "--require-hotspot-contact",
        choices=("auto", "always", "never"),
        default=None,
        help="filter validation planning by final ESMFold2 hotspot contact",
    )
    _add_validation_hash_arguments(validate_plan)
    validate_plan.add_argument("--max-attempts", type=int, default=None)
    validate_plan.set_defaults(func=_validate_plan)

    validate_msa_plan = subparsers.add_parser(
        "validate-msa-plan",
        help="enqueue Protenix MSA prefetch jobs from completed critic rows",
    )
    validate_msa_plan.add_argument("campaign_dir", type=Path)
    validate_msa_plan.set_defaults(func=_validate_msa_plan)

    validate_msa_run = subparsers.add_parser(
        "validate-msa-run",
        help="run rate-limited Protenix MSA prefetch jobs",
    )
    validate_msa_run.add_argument("campaign_dir", type=Path)
    validate_msa_run.add_argument("--worker-id", default="msa-prefetch-worker-0")
    validate_msa_run.add_argument("--max-jobs", type=int, default=None)
    validate_msa_run.add_argument(
        "--stale-timeout",
        type=float,
        default=None,
        help="recover running MSA jobs whose heartbeat is older than this many seconds",
    )
    validate_msa_run.add_argument(
        "--msa-max-requests-per-minute",
        type=float,
        default=DEFAULT_MSA_MAX_REQUESTS_PER_MINUTE,
        help="campaign-wide ColabFold/MMseqs submission rate limit",
    )
    validate_msa_run.set_defaults(func=_validate_msa_run)

    validate_msa_retry = subparsers.add_parser(
        "validate-msa-retry",
        help="reset failed Protenix MSA jobs so workers can retry them",
    )
    validate_msa_retry.add_argument("campaign_dir", type=Path)
    validate_msa_retry.add_argument(
        "--msa-job-id",
        action="append",
        default=None,
        help="failed MSA job ID to retry; can repeat",
    )
    validate_msa_retry.add_argument(
        "--candidate-id",
        action="append",
        default=None,
        help="retry failed MSA jobs associated with this candidate; can repeat",
    )
    validate_msa_retry.add_argument(
        "--keep-attempt-count",
        action="store_true",
        help="do not reset failed MSA job attempt counters",
    )
    validate_msa_retry.set_defaults(func=_validate_msa_retry)

    validate_run_mock = subparsers.add_parser(
        "validate-run-mock",
        help="run pending validation tasks with the durable mock validator",
    )
    validate_run_mock.add_argument("campaign_dir", type=Path)
    validate_run_mock.add_argument(
        "--worker-id",
        default="mock-validation-worker-0",
    )
    validate_run_mock.add_argument("--gpu-id", default=None)
    validate_run_mock.add_argument("--max-tasks", type=int, default=None)
    validate_run_mock.set_defaults(func=_validate_run_mock)

    validate_run = subparsers.add_parser(
        "validate-run",
        help="run pending validation tasks on one local GPU with Protenix",
    )
    _add_validate_run_arguments(validate_run)
    validate_run.set_defaults(func=_validate_run)

    validate_run_multi = subparsers.add_parser(
        "validate-run-multi",
        help="run one Protenix validation worker per GPU id",
        description="run one Protenix validation worker per GPU id",
    )
    validate_run_multi.add_argument("campaign_dir", type=Path)
    validate_run_multi.add_argument(
        "--gpus",
        nargs="+",
        required=True,
        help="GPU ids as comma-separated values, ranges like 0-3, or all",
    )
    validate_run_multi.add_argument("--worker-prefix", default="validation-gpu")
    validate_run_multi.add_argument("--max-tasks-per-worker", type=int, default=None)
    validate_run_multi.add_argument("--poll-interval", type=float, default=2.0)
    validate_run_multi.add_argument("--stale-timeout", type=float, default=None)
    _add_validate_run_arguments(
        validate_run_multi,
        include_campaign_dir=False,
        include_worker_identity=False,
        include_max_tasks=False,
    )
    validate_run_multi.set_defaults(func=_validate_run_multi)

    validate_report = subparsers.add_parser(
        "validate-report",
        help="write validation results and structure samples under validation/{model}/",
    )
    validate_report.add_argument("campaign_dir", type=Path)
    validate_report.set_defaults(func=_validate_report)

    analyze = subparsers.add_parser(
        "analyze",
        help="rank validated designs and copy top-k paired structures under ranked_results/",
    )
    analyze.add_argument("campaign_dir", type=Path)
    analyze.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="number of top-ranked paired structures to copy; default analysis.top_k",
    )
    analyze.add_argument(
        "--max-binder-rmsd-angstrom",
        "--analysis-max-binder-rmsd-angstrom",
        dest="analysis_max_binder_rmsd_angstrom",
        type=float,
        default=None,
        help="override the final-ranking binder RMSD gate; default 2.5 angstrom",
    )
    analyze.add_argument(
        "--rmsd-weight",
        "--analysis-rmsd-weight",
        dest="analysis_rmsd_weight",
        type=float,
        default=None,
        help="override RMSD agreement weight in final ranking; default 0.10",
    )
    analyze.set_defaults(func=_analyze)

    validate = subparsers.add_parser(
        "validate",
        help="run the optional Protenix validation lifecycle",
        description=(
            "thin wrapper over validate-msa-plan, validate-msa-run, "
            "validate-plan, validate-run/validate-run-multi, and validate-report"
        ),
    )
    validate.add_argument("campaign_dir", type=Path)
    validate.add_argument(
        "--validate-top-k",
        type=_parse_validate_top_k,
        default=None,
        help="number of ranked ESMFold2-passing candidates to validate, or all",
    )
    validate.add_argument("--min-esm-iptm", type=float, default=None)
    validate.add_argument(
        "--require-hotspot-contact",
        choices=("auto", "always", "never"),
        default=None,
        help="filter validation planning by final ESMFold2 hotspot contact",
    )
    validate.add_argument("--max-attempts", type=int, default=None)
    validate.add_argument(
        "--skip-msa-plan",
        action="store_true",
        default=False,
        help="do not enqueue MSA prefetch jobs before validation planning",
    )
    validate.add_argument(
        "--skip-msa-run",
        action="store_true",
        default=False,
        help="do not run MSA prefetch jobs before validation planning",
    )
    validate.add_argument(
        "--msa-max-jobs",
        type=int,
        default=None,
        help="maximum MSA jobs to run in this wrapper invocation",
    )
    validate.add_argument(
        "--msa-max-requests-per-minute",
        type=float,
        default=DEFAULT_MSA_MAX_REQUESTS_PER_MINUTE,
        help="campaign-wide ColabFold/MMseqs submission rate limit",
    )
    validate.add_argument(
        "--gpus",
        nargs="+",
        default=None,
        help="GPU ids as comma-separated values, ranges like 0-3, or all",
    )
    validate.add_argument("--worker-prefix", default="validation-gpu")
    validate.add_argument("--max-tasks-per-worker", type=int, default=None)
    validate.add_argument("--poll-interval", type=float, default=2.0)
    validate.add_argument("--stale-timeout", type=float, default=None)
    validate.add_argument(
        "--skip-report",
        action="store_true",
        default=False,
        help="do not regenerate validation reports at the end",
    )
    validate.add_argument(
        "--skip-analysis",
        action="store_true",
        default=False,
        help="do not regenerate analysis ranking outputs at the end",
    )
    validate.add_argument(
        "--analysis-top-k",
        type=int,
        default=None,
        help="number of top-ranked paired structures to copy; default analysis.top_k",
    )
    validate.add_argument(
        "--analysis-max-binder-rmsd-angstrom",
        type=float,
        default=None,
        help="override the final-ranking binder RMSD gate; default 2.5 angstrom",
    )
    validate.add_argument(
        "--analysis-rmsd-weight",
        type=float,
        default=None,
        help="override RMSD agreement weight in final ranking; default 0.10",
    )
    _add_validate_run_arguments(validate, include_campaign_dir=False)
    validate.set_defaults(func=_validate)

    return parser


def _add_validate_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_campaign_dir: bool = True,
    include_worker_identity: bool = True,
    include_max_tasks: bool = True,
) -> None:
    if include_campaign_dir:
        parser.add_argument("campaign_dir", type=Path)
    if include_worker_identity:
        parser.add_argument("--worker-id", default="protenix-validation-worker-0")
        parser.add_argument("--gpu-id", default=None)
    parser.add_argument(
        "--validate-model",
        default=None,
        help="validation model name",
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=None,
        help=(
            "maximum validation tasks to send through one Protenix invocation; "
            "multi-GPU validation may shrink this to spread small campaigns "
            "across workers"
        ),
    )
    if include_max_tasks:
        parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--min-validation-iptm", type=float, default=None)
    parser.add_argument("--min-validation-ipsae", type=float, default=None)
    parser.add_argument(
        "--ipsae-script",
        type=Path,
        default=None,
        help="path to ipsae.py for Protenix full_data ipSAE scoring",
    )
    parser.add_argument(
        "--ipsae-python",
        default=None,
        help="Python executable used to run ipsae.py; defaults to current Python",
    )
    parser.add_argument("--ipsae-pae-cutoff", type=float, default=None)
    parser.add_argument("--ipsae-dist-cutoff", type=float, default=None)
    parser.add_argument(
        "--validation-hotspot-cutoff-angstrom",
        type=float,
        default=None,
        help="override the campaign hotspot cutoff for Protenix validation",
    )
    parser.add_argument(
        "--protenix-command",
        default=None,
        help=(
            "explicit Protenix command prefix, parsed like a shell string; "
            "defaults to the current Python with -m runner.inference"
        ),
    )
    parser.add_argument(
        "--protenix-python",
        default=None,
        help="Python executable for a separate Protenix environment",
    )
    parser.add_argument("--protenix-root", type=Path, default=None)
    parser.add_argument("--protenix-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--keep-validation-debug", action="store_true", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--n-sample", type=int, default=None)
    parser.add_argument("--n-step", type=int, default=None)
    parser.add_argument("--n-cycle", type=int, default=None)
    parser.add_argument("--use-msa", action="store_true", default=None)
    parser.add_argument(
        "--use-template",
        choices=("auto", "true", "false"),
        default=None,
        help=(
            "structural template use for Protenix validation; auto uses available "
            "target and bundled framework CIF templates"
        ),
    )
    parser.add_argument(
        "--target-msa-mode",
        choices=("none", "provided", "server"),
        default=None,
        help="target MSA source; inferred from --msa-server or provided MSA paths when omitted",
    )
    parser.add_argument(
        "--binder-msa-mode",
        choices=("auto", "none", "single_sequence"),
        default=None,
        help=(
            "binder MSA source; auto uses single-sequence MSAs for miniproteins "
            "and grouped/template MSAs for VHH; scFv support is pending"
        ),
    )
    parser.add_argument(
        "--msa-server",
        default=None,
        help="ColabFold/MMseqs server URL for target MSA fetching",
    )
    parser.add_argument(
        "--target-msa-dir",
        type=Path,
        default=None,
        help="directory containing target pairing.a3m and non_pairing.a3m",
    )
    parser.add_argument(
        "--target-msa-map-csv",
        type=Path,
        default=None,
        help="CSV mapping target name/sequence/chain to MSA paths",
    )
    parser.add_argument(
        "--msa-cache-root",
        type=Path,
        default=None,
        help="cache root for fetched or provided target MSAs",
    )
    parser.add_argument(
        "--msa-pairing-strategy",
        choices=("greedy", "query_only", "copy_non_pairing"),
        default=None,
    )
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--heartbeat-interval", type=float, default=None)
    parser.add_argument("--token-limit", type=int, default=None)
    parser.add_argument(
        "--no-validation-preflight",
        action="store_true",
        default=None,
        help="skip the lightweight Protenix startup environment check",
    )


def _add_validation_hash_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--validation-hotspot-cutoff-angstrom",
        type=float,
        default=None,
        help="override the campaign hotspot cutoff for Protenix validation",
    )
    parser.add_argument(
        "--protenix-command",
        default=None,
        help=(
            "explicit Protenix command prefix, parsed like a shell string; "
            "included in the validation config hash"
        ),
    )
    parser.add_argument(
        "--protenix-python",
        default=None,
        help="Python executable for a separate Protenix environment",
    )
    parser.add_argument("--protenix-root", type=Path, default=None)
    parser.add_argument("--protenix-checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--ipsae-script",
        type=Path,
        default=None,
        help="path to ipsae.py for Protenix full_data ipSAE scoring",
    )
    parser.add_argument(
        "--ipsae-python",
        default=None,
        help="Python executable used to run ipsae.py; defaults to current Python",
    )
    parser.add_argument("--ipsae-pae-cutoff", type=float, default=None)
    parser.add_argument("--ipsae-dist-cutoff", type=float, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--n-sample", type=int, default=None)
    parser.add_argument("--n-step", type=int, default=None)
    parser.add_argument("--n-cycle", type=int, default=None)
    parser.add_argument("--use-msa", action="store_true", default=None)
    parser.add_argument(
        "--use-template",
        choices=("auto", "true", "false"),
        default=None,
        help=(
            "structural template use for Protenix validation; included in the "
            "validation config hash"
        ),
    )
    parser.add_argument(
        "--target-msa-mode",
        choices=("none", "provided", "server"),
        default=None,
        help="target MSA source; inferred from --msa-server or provided MSA paths when omitted",
    )
    parser.add_argument(
        "--binder-msa-mode",
        choices=("auto", "none", "single_sequence"),
        default=None,
        help=(
            "binder MSA source; auto uses single-sequence MSAs for miniproteins "
            "and grouped/template MSAs for VHH; scFv support is pending"
        ),
    )
    parser.add_argument(
        "--msa-server",
        default=None,
        help="ColabFold/MMseqs server URL for target MSA fetching",
    )
    parser.add_argument(
        "--target-msa-dir",
        type=Path,
        default=None,
        help="directory containing target pairing.a3m and non_pairing.a3m",
    )
    parser.add_argument(
        "--target-msa-map-csv",
        type=Path,
        default=None,
        help="CSV mapping target name/sequence/chain to MSA paths",
    )
    parser.add_argument(
        "--msa-cache-root",
        type=Path,
        default=None,
        help="cache root for fetched or provided target MSAs",
    )
    parser.add_argument(
        "--msa-pairing-strategy",
        choices=("greedy", "query_only", "copy_non_pairing"),
        default=None,
    )
    parser.add_argument("--token-limit", type=int, default=None)


def _add_check_protenix_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--protenix-command",
        default=None,
        help="explicit Protenix command prefix, parsed like a shell string",
    )
    parser.add_argument(
        "--protenix-python",
        default=None,
        help="Python executable for a separate Protenix environment",
    )
    parser.add_argument("--protenix-root", type=Path, default=None)
    parser.add_argument("--protenix-checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--ipsae-script",
        type=Path,
        default=None,
        help="path to ipsae.py for Protenix full_data ipSAE scoring",
    )
    parser.add_argument(
        "--gpu-id",
        default=None,
        help="GPU id expected to be visible through CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--json", action="store_true", help="print JSON output")


def _plan_mock(args: argparse.Namespace) -> int:
    shard_id = plan_one_mock_shard(args.campaign_dir)
    print(f"planned mock campaign: {args.campaign_dir}")
    print(f"shard: {shard_id}")
    return 0


def _plan(args: argparse.Namespace) -> int:
    result = plan_campaign(args.config, output_override=args.out)
    _print_plan_result(result, include_next=True)
    return 0


def _print_plan_result(result, *, include_next: bool) -> None:
    print(f"planned campaign: {result.campaign_dir}")
    print(f"shards: {result.shard_count}")
    print(f"designs: {len(result.config.seeds)}")
    print(f"target: {result.config.target_name}")
    print(f"binder_scaffold: {result.config.binder.scaffold}")
    if result.config.binder.framework_names:
        print(f"binder_frameworks: {', '.join(result.config.binder.framework_names)}")
    if result.config.binder.length_range is not None:
        low, high = result.config.binder.length_range
        print(f"binder_length: {low}-{high}")
    print(f"inversion_model: {result.config.inversion_model_name}")
    print(f"critic: {result.config.critic_name}")
    print(f"steps: {result.config.steps}")
    if include_next:
        _print_next_commands(result.campaign_dir)


def _check_config(args: argparse.Namespace) -> int:
    result = check_campaign_config(args.config, output_override=args.out)
    _print_config_check(result)
    if not result.ok or result.config is None:
        return 1
    try:
        _validation_yaml_defaults_for_config(args.config)
    except ValueError as exc:
        print("validation_config: invalid")
        print(f"  {exc}")
        return 1

    if args.env or args.models:
        require_local_runtime = _design_backend_from_env() == "local"
        env_result = check_environment(
            esm_repo=args.esm_repo,
            require_cuda=True,
            require_tutorial=not require_local_runtime,
            require_local_runtime=require_local_runtime,
        )
        _print_env_check(env_result)
        if not env_result.ok:
            return 1

    if args.models:
        model_result = preflight_models(
            esm_repo=args.esm_repo,
            gpu_id=args.gpu_id,
            inversion_model_name=result.config.inversion_model_name,
            critic_name=result.config.critic_name,
            steps=result.config.steps,
            disable_hf_xet=not args.enable_hf_xet,
        )
        print("model preflight: ok")
        print(f"inversion_model: {model_result.inversion_model_name}")
        print(f"critic: {model_result.critic_name}")
        print(f"inversion_models: {', '.join(model_result.loaded_inversion_models)}")
        print(f"critic_models: {', '.join(model_result.loaded_critic_models)}")
        print(f"esmc_loaded: {str(model_result.esmc_loaded).lower()}")

    return 0


def _design_backend_from_env() -> str:
    return os.environ.get("ESMFOLD2_PIPELINE_DESIGN_BACKEND", "local").strip().lower()


def _check_env(args: argparse.Namespace) -> int:
    require_local_runtime = args.local_runtime or _design_backend_from_env() == "local"
    result = check_environment(
        esm_repo=args.esm_repo,
        require_cuda=True,
        require_tutorial=not args.no_tutorial and not require_local_runtime,
        require_local_runtime=require_local_runtime,
    )
    if args.json:
        print(result.to_json())
    else:
        _print_env_check(result)
    return 0 if result.ok else 1


def _check_models(args: argparse.Namespace) -> int:
    inversion_model_name = args.inversion_model_name
    critic_name = args.critic_name
    if args.model is not None:
        if (
            args.inversion_model_name != DEFAULT_ESMFOLD2_INVERSION_MODEL
            or args.critic_name != DEFAULT_ESMFOLD2_CRITIC_MODEL
        ):
            print(
                "error: --model cannot be combined with "
                "--inversion-model-name or --critic-name"
            )
            return 2
        inversion_model_name = resolve_esmfold2_model_name(args.model)
        critic_name = inversion_model_name

    result = preflight_models(
        esm_repo=args.esm_repo,
        gpu_id=args.gpu_id,
        inversion_model_name=inversion_model_name,
        critic_name=critic_name,
        steps=args.steps,
        disable_hf_xet=not args.enable_hf_xet,
    )
    print("model preflight: ok")
    print(f"inversion_model: {result.inversion_model_name}")
    print(f"critic: {result.critic_name}")
    print(f"inversion_models: {', '.join(result.loaded_inversion_models)}")
    print(f"critic_models: {', '.join(result.loaded_critic_models)}")
    print(f"esmc_loaded: {str(result.esmc_loaded).lower()}")
    return 0


def _check_protenix(args: argparse.Namespace) -> int:
    if args.protenix_command is not None and args.protenix_python is not None:
        print("error: use --protenix-command or --protenix-python, not both")
        return 2
    command = None
    if args.protenix_command is not None:
        command = tuple(shlex.split(args.protenix_command))
        if not command:
            print("error: --protenix-command cannot be empty")
            return 2
    protenix_python = args.protenix_python
    if command is None and protenix_python is None:
        protenix_python = _default_protenix_python()
    checkpoint_dir = args.protenix_checkpoint_dir or _default_protenix_checkpoint_dir()
    result = check_protenix_environment(
        protenix_command=command,
        protenix_python=protenix_python,
        protenix_root=args.protenix_root,
        checkpoint_dir=checkpoint_dir,
        ipsae_script_path=args.ipsae_script,
        gpu_id=args.gpu_id,
    )
    if args.json:
        print(result.to_json())
    else:
        _print_protenix_check(result)
    return 0 if result.ok else 1


def _launch(args: argparse.Namespace) -> int:
    if args.gpu_id is not None and args.gpus is not None:
        print("error: use --gpu-id or --gpus, not both")
        return 2

    if args.config is not None:
        if _is_campaign_dir(args.config):
            resume_errors = _resume_launch_flag_errors(args)
            if resume_errors:
                print(
                    "error: launch CAMPAIGN_DIR cannot be combined with "
                    "config-generation flags:"
                )
                for error in resume_errors:
                    print(f"  {error}")
                return 2
            return _launch_campaign_dir(args.config, args)
        if args.config.is_dir():
            print(
                "error: launch CAMPAIGN_DIR requires an existing "
                f"campaign.sqlite: {args.config}"
            )
            return 2
        config_only_errors = _config_launch_flag_errors(args)
        if config_only_errors:
            print("error: launch CONFIG cannot be combined with no-YAML flags:")
            for error in config_only_errors:
                print(f"  {error}")
            return 2
        return _launch_config_path(args.config, args)

    errors = _generated_launch_errors(args)
    if errors:
        print("error: invalid no-YAML launch arguments:")
        for error in errors:
            print(f"  {error}")
        return 2

    raw_config = _generated_launch_config(args)
    with tempfile.TemporaryDirectory(prefix="esmfold2-launch-") as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        write_text_atomic(
            config_path,
            yaml.safe_dump(raw_config, sort_keys=False),
        )
        return _launch_config_path(config_path, args)


def _launch_config_path(config_path: Path, args: argparse.Namespace) -> int:
    check_result = check_campaign_config(config_path, output_override=args.out)
    _print_config_check(check_result)
    if not check_result.ok:
        return 1
    try:
        _validation_yaml_defaults_for_config(config_path)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    try:
        plan_result = plan_campaign(config_path, output_override=args.out)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    _print_plan_result(plan_result, include_next=False)
    return _run_launch_pipeline(plan_result.campaign_dir, args)


def _launch_campaign_dir(campaign_dir: Path, args: argparse.Namespace) -> int:
    print(f"resuming campaign: {campaign_dir}")
    return _run_launch_pipeline(campaign_dir, args)


def _run_launch_pipeline(campaign_dir: Path, args: argparse.Namespace) -> int:
    try:
        _validation_yaml_defaults(campaign_dir)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    try:
        msa_pool = _launch_validation_msa_worker_pool(
            campaign_dir=campaign_dir,
            args=args,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    if args.gpus is not None:
        print(
            "launching multi-GPU workers; first run may download large model "
            "checkpoints before GPU memory increases",
            file=sys.stderr,
            flush=True,
        )
        with msa_pool:
            result = run_multi_campaign(
                campaign_dir,
                esm_repo=args.esm_repo,
                gpu_ids=args.gpus,
                worker_prefix=args.worker_prefix,
                max_shards_per_worker=args.max_shards_per_worker,
                poll_interval_seconds=args.poll_interval,
                heartbeat_interval_seconds=args.heartbeat_interval,
                stale_after_seconds=args.stale_timeout,
                disable_hf_xet=not args.enable_hf_xet,
                disable_local_runtime_cache=getattr(
                    args, "disable_local_runtime_cache", False
                ),
            )
        _print_run_multi_result(result)
        _print_launch_validation_msa_result(msa_pool)
        if not result.ok:
            return 1
        return _run_launch_final_steps(campaign_dir, args)

    print(
        "launching local worker; first run may download large model checkpoints "
        "before GPU memory increases",
        file=sys.stderr,
        flush=True,
    )
    with msa_pool:
        with _local_runtime_cache_disabled_if_requested(args):
            result = run_campaign(
                campaign_dir,
                esm_repo=args.esm_repo,
                worker_id="local-worker-0",
                gpu_id=args.gpu_id,
                max_shards=args.max_shards,
                heartbeat_interval_seconds=args.heartbeat_interval,
                stale_after_seconds=args.stale_timeout,
                disable_hf_xet=not args.enable_hf_xet,
            )
    _print_run_campaign_result(result)
    _print_launch_validation_msa_result(msa_pool)
    return _run_launch_final_steps(campaign_dir, args)


def _run_launch_final_steps(campaign_dir: Path, args: argparse.Namespace) -> int:
    try:
        aggregate = aggregate_campaign(campaign_dir)
        print(f"metrics_csv: {aggregate.metrics_csv}")
        print(f"summary_json: {aggregate.summary_json}")
        print(f"metric_rows: {aggregate.row_count}")

        select = select_campaign(
            campaign_dir,
            max_designs=_launch_max_designs(args),
            min_iptm=_effective_min_iptm(args),
            require_hotspot_contact=_launch_require_hotspot_contact(args),
        )
        print(f"ranked_csv: {select.ranked_csv}")
        print(f"candidate_pool: {select.candidate_count}")
        print(f"selected: {select.selected_count}")

        if getattr(args, "skip_export", False):
            print("export: skipped")
        else:
            export = export_campaign(campaign_dir)
            print(f"selected_dir: {export.selected_dir}")
            print(f"manifest_csv: {export.manifest_csv}")
            print(f"copied_files: {export.copied_files}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    try:
        should_validate = _launch_should_run_validation(campaign_dir, args)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    if not should_validate:
        print("validation: skipped")
        return 0

    validation_args = _launch_validation_args(args, campaign_dir)
    return _run_validation_lifecycle(validation_args)


def _launch_validation_msa_worker_pool(
    *,
    campaign_dir: Path,
    args: argparse.Namespace,
) -> "_LaunchValidationMsaWorkerPool":
    count = args.validation_msa_workers
    if count is None:
        count = 1 if _launch_should_run_validation(campaign_dir, args) else 0
    if count < 0:
        raise ValueError("--validation-msa-workers must be non-negative")
    if args.validation_msa_poll_interval <= 0:
        raise ValueError("--validation-msa-poll-interval must be positive")
    if args.msa_max_requests_per_minute <= 0:
        raise ValueError("--msa-max-requests-per-minute must be positive")
    return _LaunchValidationMsaWorkerPool(
        campaign_dir=campaign_dir,
        count=count,
        max_requests_per_minute=args.msa_max_requests_per_minute,
        poll_interval_seconds=args.validation_msa_poll_interval,
    )


def _is_campaign_dir(path: Path) -> bool:
    return path.is_dir() and (path / "campaign.sqlite").exists()


def _launch_max_designs(args: argparse.Namespace) -> int:
    return int(getattr(args, "max_designs", DEFAULT_MAX_DESIGNS))


def _launch_require_hotspot_contact(args: argparse.Namespace) -> str:
    return getattr(args, "require_hotspot_contact", None) or "auto"


def _effective_min_iptm(args: argparse.Namespace) -> float:
    value = getattr(args, "min_iptm", None)
    return DEFAULT_MIN_IPTM if value is None else float(value)


def _launch_should_run_validation(
    campaign_dir: Path,
    args: argparse.Namespace,
) -> bool:
    if getattr(args, "skip_validation", False):
        return False
    return bool(_validation_yaml_mapping(campaign_dir))


def _launch_validation_args(
    args: argparse.Namespace,
    campaign_dir: Path,
) -> argparse.Namespace:
    values = dict(vars(args))
    values.update(
        {
            "campaign_dir": campaign_dir,
            "worker_id": "protenix-validation-worker-0",
            "worker_prefix": "validation-gpu",
            "max_tasks_per_worker": None,
            "max_tasks": None,
            "validate_top_k": None,
            "min_esm_iptm": getattr(args, "min_iptm", None),
            "max_attempts": None,
            "skip_msa_plan": False,
            "skip_msa_run": False,
            "msa_max_jobs": None,
            "skip_report": False,
            "skip_analysis": getattr(args, "skip_analysis", False),
            "analysis_top_k": getattr(args, "analysis_top_k", None),
            "validate_model": None,
            "validation_batch_size": None,
            "min_validation_iptm": getattr(args, "min_iptm", None),
            "min_validation_ipsae": None,
            "ipsae_script": None,
            "ipsae_python": None,
            "ipsae_pae_cutoff": None,
            "ipsae_dist_cutoff": None,
            "validation_hotspot_cutoff_angstrom": None,
            "protenix_command": None,
            "protenix_python": None,
            "protenix_root": None,
            "protenix_checkpoint_dir": None,
            "scratch_root": None,
            "keep_validation_debug": None,
            "seeds": None,
            "n_sample": None,
            "n_step": None,
            "n_cycle": None,
            "use_msa": None,
            "use_template": None,
            "target_msa_mode": None,
            "binder_msa_mode": None,
            "msa_server": None,
            "target_msa_dir": None,
            "target_msa_map_csv": None,
            "msa_cache_root": None,
            "msa_pairing_strategy": None,
            "timeout_seconds": None,
            "token_limit": None,
            "no_validation_preflight": None,
        }
    )
    if getattr(args, "require_hotspot_contact", None) is not None:
        values["require_hotspot_contact"] = args.require_hotspot_contact
    else:
        values["require_hotspot_contact"] = None
    return argparse.Namespace(**values)


class _LaunchValidationMsaWorkerPool:
    def __init__(
        self,
        *,
        campaign_dir: Path,
        count: int,
        max_requests_per_minute: float,
        poll_interval_seconds: float,
    ) -> None:
        self.campaign_dir = campaign_dir
        self.count = count
        self.max_requests_per_minute = max_requests_per_minute
        self.poll_interval_seconds = poll_interval_seconds
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.skipped_jobs = 0
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._drain_after_stop = False
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> "_LaunchValidationMsaWorkerPool":
        if self.count <= 0:
            return self
        print(f"launch validation MSA workers: {self.count}", flush=True)
        for index in range(self.count):
            thread = threading.Thread(
                target=self._run_worker,
                args=(index,),
                name=f"launch-validation-msa-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._drain_after_stop = exc_type is None
        self._stop.set()
        for thread in self._threads:
            thread.join()

    def _run_worker(self, index: int) -> None:
        worker_id = f"launch-msa-worker-{index}"
        while True:
            try:
                result = run_msa_prefetch_worker(
                    self.campaign_dir,
                    worker_id=worker_id,
                    max_jobs=1,
                    max_requests_per_minute=self.max_requests_per_minute,
                    log=lambda message: print(
                        f"[esmfold2-pipeline] {message}",
                        flush=True,
                    ),
                )
            except Exception as exc:
                with self._lock:
                    self.errors.append(str(exc))
                if self._stop.is_set():
                    return
                self._stop.wait(self.poll_interval_seconds)
                continue

            with self._lock:
                self.completed_jobs += result.completed_jobs
                self.failed_jobs += result.failed_jobs
                self.skipped_jobs += result.skipped_jobs

            if self._stop.is_set():
                if self._drain_after_stop and not result.no_pending:
                    continue
                return
            if result.no_pending:
                self._stop.wait(self.poll_interval_seconds)


def _print_launch_validation_msa_result(pool: _LaunchValidationMsaWorkerPool) -> None:
    if pool.count <= 0:
        return
    print(f"launch_msa_completed_jobs: {pool.completed_jobs}")
    print(f"launch_msa_failed_jobs: {pool.failed_jobs}")
    print(f"launch_msa_skipped_jobs: {pool.skipped_jobs}")
    if pool.errors:
        print(f"launch_msa_worker_errors: {len(pool.errors)}")
        for error in pool.errors[:5]:
            print(f"  {error}")


def _config_launch_flag_errors(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    for name in (
        "target_name",
        "target_sequence",
        "target_structure",
        "chains",
        "hotspot",
        "scaffold",
        "frameworks",
        "length",
        "num_designs",
        "seed_start",
        "model",
        "steps",
        "binder_target_contact_mode",
        "mosaic_cdr_contact_weight",
        "mosaic_cdr_contact_cutoff_angstrom",
        "mosaic_cdr_num_target_contacts",
        "mosaic_framework_contact_penalty_weight",
        "mosaic_framework_contact_penalty_scope",
    ):
        if getattr(args, name, None) is not None:
            errors.append(f"--{name.replace('_', '-')}")
    return errors


def _resume_launch_flag_errors(args: argparse.Namespace) -> list[str]:
    errors = _config_launch_flag_errors(args)
    if getattr(args, "out", None) is not None:
        errors.append("--out")
    return errors


def _generated_launch_errors(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if args.target_name is None:
        errors.append("--target-name is required")
    if args.num_designs is None:
        errors.append("--num-designs is required")
    if args.out is None:
        errors.append("--out is required")

    target_sources = [
        name
        for name, value in (
            ("--target-sequence", args.target_sequence),
            ("--target-structure", args.target_structure),
        )
        if value is not None
    ]
    if not target_sources:
        errors.append("--target-sequence or --target-structure is required")
    elif len(target_sources) > 1:
        errors.append("use --target-sequence or --target-structure, not both")

    if args.target_sequence is not None and "|" in args.target_sequence:
        errors.append("--target-sequence currently supports one target chain")
    if args.target_structure is None and args.chains is not None:
        errors.append("--chains requires --target-structure")
    if args.target_structure is None and args.hotspot is not None:
        errors.append("--hotspot requires --target-structure")

    scaffold = args.scaffold or "miniprotein"
    contact_mode = args.binder_target_contact_mode
    if contact_mode == "mosaic_cdr" and scaffold not in {"scfv", "vhh"}:
        errors.append("mosaic_cdr contact mode requires --scaffold scfv or vhh")

    mosaic_tuning_flags = (
        ("--mosaic-cdr-contact-weight", args.mosaic_cdr_contact_weight),
        (
            "--mosaic-cdr-contact-cutoff-angstrom",
            args.mosaic_cdr_contact_cutoff_angstrom,
        ),
        ("--mosaic-cdr-num-target-contacts", args.mosaic_cdr_num_target_contacts),
        (
            "--mosaic-framework-contact-penalty-weight",
            args.mosaic_framework_contact_penalty_weight,
        ),
        (
            "--mosaic-framework-contact-penalty-scope",
            args.mosaic_framework_contact_penalty_scope,
        ),
    )
    if contact_mode != "mosaic_cdr":
        for flag, value in mosaic_tuning_flags:
            if value is not None:
                errors.append(
                    f"{flag} requires --binder-target-contact-mode mosaic_cdr"
                )
    else:
        if (
            args.mosaic_cdr_contact_weight is not None
            and args.mosaic_cdr_contact_weight < 0
        ):
            errors.append("--mosaic-cdr-contact-weight must be non-negative")
        if (
            args.mosaic_cdr_contact_cutoff_angstrom is not None
            and args.mosaic_cdr_contact_cutoff_angstrom <= 0
        ):
            errors.append("--mosaic-cdr-contact-cutoff-angstrom must be positive")
        if (
            args.mosaic_cdr_num_target_contacts is not None
            and args.mosaic_cdr_num_target_contacts <= 0
        ):
            errors.append("--mosaic-cdr-num-target-contacts must be positive")
        if (
            args.mosaic_framework_contact_penalty_weight is not None
            and args.mosaic_framework_contact_penalty_weight < 0
        ):
            errors.append(
                "--mosaic-framework-contact-penalty-weight must be non-negative"
            )
    if scaffold in {"scfv", "vhh"}:
        if args.length is not None:
            errors.append("--length is only valid when --scaffold miniprotein")
    elif args.frameworks is not None:
        errors.append("--frameworks is only valid when --scaffold scfv or vhh")

    if args.frameworks is not None and scaffold in {"scfv", "vhh"}:
        framework_error = _frameworks_error(args.frameworks, scaffold=scaffold)
        if framework_error is not None:
            errors.append(framework_error)
    return errors


def _generated_launch_config(args: argparse.Namespace) -> dict:
    scaffold = args.scaffold or "miniprotein"
    binder: dict[str, object] = {"scaffold": scaffold}
    if args.length is not None:
        binder["length"] = args.length
    if scaffold in {"scfv", "vhh"}:
        binder["frameworks"] = _resolve_cli_frameworks(
            args.frameworks or ["all"],
            scaffold=scaffold,
        )

    campaign: dict[str, object] = {"num_designs": args.num_designs}
    if args.seed_start is not None:
        campaign["seed_start"] = args.seed_start
    if args.model is not None:
        model_name = resolve_esmfold2_model_name(args.model)
        campaign["inversion_model"] = model_name
        campaign["critics"] = [model_name]
    if args.steps is not None:
        campaign["steps"] = args.steps

    loss: dict[str, object] = {}
    contact_mode = args.binder_target_contact_mode
    if contact_mode is not None:
        loss["binder_target_contact_mode"] = contact_mode
    if args.mosaic_cdr_contact_weight is not None:
        loss["mosaic_cdr_contact_weight"] = args.mosaic_cdr_contact_weight
    if args.mosaic_cdr_contact_cutoff_angstrom is not None:
        loss["mosaic_cdr_contact_cutoff_angstrom"] = (
            args.mosaic_cdr_contact_cutoff_angstrom
        )
    if args.mosaic_cdr_num_target_contacts is not None:
        loss["mosaic_cdr_num_target_contacts"] = (
            args.mosaic_cdr_num_target_contacts
        )
    if args.mosaic_framework_contact_penalty_weight is not None:
        loss["mosaic_framework_contact_penalty_weight"] = (
            args.mosaic_framework_contact_penalty_weight
        )
    if args.mosaic_framework_contact_penalty_scope is not None:
        loss["mosaic_framework_contact_penalty_scope"] = (
            args.mosaic_framework_contact_penalty_scope
        )

    target: dict[str, object] = {"name": args.target_name}
    if args.target_sequence is not None:
        target["sequence"] = args.target_sequence
    elif args.target_structure is not None:
        target["structure"] = str(_resolve_cli_path(args.target_structure))
        if args.chains is not None:
            target["chains"] = _split_cli_values(args.chains)
        if args.hotspot is not None:
            target["hotspots"] = ",".join(args.hotspot)

    config = {
        "target": target,
        "binder": binder,
        "campaign": campaign,
        "output": str(args.out),
    }
    if loss:
        config["loss"] = loss
    return config


def _split_cli_values(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def _resolve_cli_frameworks(values: list[str], *, scaffold: str) -> list[str]:
    frameworks = _split_cli_values(values)
    normalized = [framework.lower() for framework in frameworks]
    if normalized == ["all"]:
        return list(_all_framework_names(scaffold))
    return [
        _resolve_framework_name(scaffold, framework) or framework
        for framework in frameworks
    ]


def _frameworks_error(values: list[str], *, scaffold: str) -> str | None:
    frameworks = _split_cli_values(values)
    if not frameworks:
        return "--frameworks must include at least one framework name or all"
    normalized = [framework.lower() for framework in frameworks]
    if "all" in normalized and normalized != ["all"]:
        return "--frameworks all cannot be combined with explicit framework names"
    if normalized == ["all"]:
        return None
    resolved: list[str] = []
    unknown: list[str] = []
    for framework in frameworks:
        canonical = _resolve_framework_name(scaffold, framework)
        if canonical is None:
            unknown.append(framework)
        else:
            resolved.append(canonical)
    if unknown:
        choices = ", ".join(_framework_alias_choices(scaffold))
        return (
            "--frameworks contains unsupported framework(s): "
            f"{', '.join(unknown)}; choices: {choices}, all"
        )
    if len(set(resolved)) != len(resolved):
        return "--frameworks values must be unique"
    return None


def _all_framework_names(scaffold: str) -> tuple[str, ...]:
    if scaffold == "scfv":
        return all_scfv_framework_names()
    if scaffold == "vhh":
        return all_vhh_framework_names()
    return ()


def _resolve_framework_name(scaffold: str, name: str) -> str | None:
    if scaffold == "scfv":
        return resolve_scfv_framework_name(name)
    if scaffold == "vhh":
        return resolve_vhh_framework_name(name)
    return None


def _framework_alias_choices(scaffold: str) -> tuple[str, ...]:
    if scaffold == "scfv":
        return scfv_framework_alias_choices()
    if scaffold == "vhh":
        return vhh_framework_alias_choices()
    return ()


def _resolve_cli_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve()


def _validate_conditioning(args: argparse.Namespace) -> int:
    binder_sequence = args.binder_sequence
    if binder_sequence is None:
        binder_sequence = default_validation_binder_sequence(args.binder_length)
    try:
        result = validate_conditioning_config(
            args.config,
            output_dir=args.out,
            esm_repo=args.esm_repo,
            gpu_id=args.gpu_id,
            critic_name=args.critic_name,
            binder_sequence=binder_sequence,
            binder_length=args.binder_length,
            num_sampling_steps=args.num_sampling_steps,
            num_loops=args.num_loops,
            seed=args.seed,
            calculate_confidence=args.calculate_confidence,
        )
    except Exception as exc:
        print("conditioning validation: failed")
        print(f"error: {exc}")
        return 1

    print("conditioning validation: ok")
    print(f"json: {result.output_json}")
    print(f"target_chain: {result.target_chain_id}")
    print(f"target_length: {result.target_length}")
    print(f"binder_chain: {result.binder_chain_id}")
    print(f"binder_length: {result.binder_length}")
    print(f"critic: {result.critic_name}")
    print(
        "baseline_target_distance_rmse: "
        f"{result.baseline_metrics.target_distance_rmse:.4f}"
    )
    print(
        "conditioned_target_distance_rmse: "
        f"{result.conditioned_metrics.target_distance_rmse:.4f}"
    )
    print(f"distance_rmse_delta: {result.distance_rmse_delta:.4f}")
    print(
        "baseline_target_aligned_rmsd: "
        f"{result.baseline_metrics.target_aligned_rmsd:.4f}"
    )
    print(
        "conditioned_target_aligned_rmsd: "
        f"{result.conditioned_metrics.target_aligned_rmsd:.4f}"
    )
    print(f"aligned_rmsd_delta: {result.aligned_rmsd_delta:.4f}")
    return 0


def _run_mock(args: argparse.Namespace) -> int:
    result = run_one_mock_shard(
        args.campaign_dir,
        worker_id=args.worker_id,
        gpu_id=args.gpu_id,
    )
    if result is None:
        print("no pending mock shard")
        return 0

    print(f"completed shard: {result.shard_id}")
    print(f"candidate: {result.candidate_id}")
    print(f"critic: {result.critic_name}")
    print(f"sequence_path: {result.sequence_path or 'none'}")
    print(f"structure_path: {result.structure_path}")
    print(f"iptm: {result.metrics['iptm']}")
    return 0


def _plan_gpu_smoke(args: argparse.Namespace) -> int:
    shard_id = plan_one_gpu_smoke_shard(
        args.campaign_dir,
        target_name=args.target_name,
        binder_name=_smoke_binder_name(args.binder_scaffold),
        inversion_model_name=args.inversion_model_name,
        critic_name=args.critic_name,
    )
    print(f"planned gpu smoke campaign: {args.campaign_dir}")
    print(f"shard: {shard_id}")
    print(f"target: {args.target_name}")
    print(f"binder_scaffold: {args.binder_scaffold}")
    print(f"inversion_model: {args.inversion_model_name}")
    print(f"critic: {args.critic_name}")
    return 0


def _run_gpu_smoke(args: argparse.Namespace) -> int:
    result = run_one_gpu_smoke_shard(
        args.campaign_dir,
        esm_repo=args.esm_repo,
        worker_id=args.worker_id,
        gpu_id=args.gpu_id,
        steps=args.steps,
        target_name=args.target_name,
        binder_name=_smoke_binder_name(args.binder_scaffold),
        inversion_model_name=args.inversion_model_name,
        critic_name=args.critic_name,
        disable_hf_xet=not args.enable_hf_xet,
    )
    if result is None:
        print("no pending gpu smoke shard")
        return 0

    print(f"completed shard: {result.shard_id}")
    print(f"candidate: {result.candidate_id}")
    print(f"critic: {result.critic_name}")
    print(f"sequence_path: {result.sequence_path or 'none'}")
    print(f"structure_path: {result.structure_path}")
    if "iptm" in result.metrics:
        print(f"iptm: {result.metrics['iptm']}")
    return 0


def _smoke_binder_name(scaffold: str) -> str:
    if scaffold == "miniprotein":
        return "minibinder"
    raise ValueError(f"unsupported binder scaffold for GPU smoke: {scaffold}")


def _run_campaign(args: argparse.Namespace) -> int:
    print(
        "running local worker; first run may download large model checkpoints "
        "before GPU memory increases",
        file=sys.stderr,
        flush=True,
    )
    with _local_runtime_cache_disabled_if_requested(args):
        result = run_campaign(
            args.campaign_dir,
            esm_repo=args.esm_repo,
            worker_id=args.worker_id,
            gpu_id=args.gpu_id,
            max_shards=args.max_shards,
            heartbeat_interval_seconds=args.heartbeat_interval,
            stale_after_seconds=args.stale_timeout,
            disable_hf_xet=not args.enable_hf_xet,
        )
    _print_run_campaign_result(result)
    return 0


def _print_run_campaign_result(result) -> None:
    if result.recovered_stale_shards:
        print(f"recovered stale shards: {result.recovered_stale_shards}")
    if result.skipped_no_pending:
        print("no pending shards")
    else:
        print(f"completed shards: {result.completed_shards}")


def _run_multi_campaign(args: argparse.Namespace) -> int:
    try:
        msa_pool = _run_multi_validation_msa_worker_pool(args)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    print(
        "running one worker per GPU; first run may download large model "
        "checkpoints before GPU memory increases",
        file=sys.stderr,
        flush=True,
    )
    with msa_pool:
        result = run_multi_campaign(
            args.campaign_dir,
            esm_repo=args.esm_repo,
            gpu_ids=args.gpus,
            worker_prefix=args.worker_prefix,
            max_shards_per_worker=args.max_shards_per_worker,
            poll_interval_seconds=args.poll_interval,
            heartbeat_interval_seconds=args.heartbeat_interval,
            stale_after_seconds=args.stale_timeout,
            disable_hf_xet=not args.enable_hf_xet,
            disable_local_runtime_cache=getattr(
                args, "disable_local_runtime_cache", False
            ),
        )
    _print_run_multi_result(result)
    _print_launch_validation_msa_result(msa_pool)
    return 0 if result.ok else 1


def _local_runtime_cache_disabled_if_requested(
    args: argparse.Namespace,
) -> AbstractContextManager[None]:
    if not getattr(args, "disable_local_runtime_cache", False):
        return nullcontext()
    return _temporary_env(_LOCAL_RUNTIME_CACHE_DISABLE_ENV, "1")


@contextmanager
def _temporary_env(name: str, value: str) -> Iterator[None]:
    sentinel = object()
    previous = os.environ.get(name, sentinel)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is sentinel:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _run_multi_validation_msa_worker_pool(
    args: argparse.Namespace,
) -> "_LaunchValidationMsaWorkerPool":
    count = args.validation_msa_workers
    if count is None:
        count = 1
    if count < 0:
        raise ValueError("--validation-msa-workers must be non-negative")
    if args.validation_msa_poll_interval <= 0:
        raise ValueError("--validation-msa-poll-interval must be positive")
    if args.msa_max_requests_per_minute <= 0:
        raise ValueError("--msa-max-requests-per-minute must be positive")
    return _LaunchValidationMsaWorkerPool(
        campaign_dir=args.campaign_dir,
        count=count,
        max_requests_per_minute=args.msa_max_requests_per_minute,
        poll_interval_seconds=args.validation_msa_poll_interval,
    )


def _print_run_multi_result(result) -> None:
    print(f"run_id: {result.run_id}")
    print(f"completed shards: {result.completed_shards}")
    print(f"recovered shards: {result.recovered_shards}")
    print(f"failed workers: {result.failed_workers}")
    for worker in result.worker_results:
        status = "ok" if worker.ok else "failed"
        print(
            f"worker: {worker.worker_id} gpu={worker.gpu_id} "
            f"status={status} returncode={worker.returncode} "
            f"completed={worker.completed_shards} recovered={worker.recovered_shards}"
        )
        print(f"  log: {worker.log_path}")


def _status(args: argparse.Namespace) -> int:
    status = inspect_campaign(args.campaign_dir)
    _print_status(status)
    return 1 if status.issues else 0


def _aggregate(args: argparse.Namespace) -> int:
    result = aggregate_campaign(args.campaign_dir)
    print(f"metrics_csv: {result.metrics_csv}")
    print(f"summary_json: {result.summary_json}")
    print(f"metric_rows: {result.row_count}")
    return 0


def _select(args: argparse.Namespace) -> int:
    result = select_campaign(
        args.campaign_dir,
        max_designs=args.max_designs,
        min_iptm=_effective_min_iptm(args),
        require_hotspot_contact=args.require_hotspot_contact,
    )
    print(f"ranked_csv: {result.ranked_csv}")
    print(f"summary_json: {result.summary_json}")
    print(f"candidate_pool: {result.candidate_count}")
    print(f"selected: {result.selected_count}")
    return 0


def _export(args: argparse.Namespace) -> int:
    result = export_campaign(args.campaign_dir, max_designs=args.max_designs)
    print(f"selected_dir: {result.selected_dir}")
    print(f"manifest_csv: {result.manifest_csv}")
    print(f"summary_json: {result.summary_json}")
    print(f"selected: {result.selected_count}")
    print(f"copied_files: {result.copied_files}")
    return 0


def _validate_plan(args: argparse.Namespace) -> int:
    try:
        args = _with_validation_yaml_defaults(args, mode="plan")
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    try:
        result = plan_validation_tasks(
            args.campaign_dir,
            config=_validation_plan_config_from_args(args),
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print(f"validation_model: {result.model_name}")
    print(f"validation_config_hash: {result.validation_config_hash}")
    print(f"candidate_pool: {result.candidate_count}")
    print(f"selected: {result.selected_count}")
    print(f"created: {result.created_count}")
    print(f"existing: {result.existing_count}")
    return 0


def _validate_msa_plan(args: argparse.Namespace) -> int:
    result = plan_msa_prefetch(args.campaign_dir, log=lambda message: print(message))
    print(f"candidates: {result.candidate_count}")
    print(f"queued_msa_jobs: {result.queued_jobs}")
    print(f"existing_msa_jobs: {result.existing_jobs}")
    print(f"skipped_candidates: {result.skipped_candidates}")
    return 0


def _validate_msa_run(args: argparse.Namespace) -> int:
    result = run_msa_prefetch_worker(
        args.campaign_dir,
        worker_id=args.worker_id,
        max_jobs=args.max_jobs,
        max_requests_per_minute=args.msa_max_requests_per_minute,
        stale_timeout_seconds=args.stale_timeout,
        log=lambda message: print(message),
    )
    if result.no_pending:
        print("no pending MSA jobs")
    print(f"recovered_stale_msa_jobs: {result.recovered_stale_jobs}")
    print(f"completed_msa_jobs: {result.completed_jobs}")
    print(f"failed_msa_jobs: {result.failed_jobs}")
    print(f"skipped_msa_jobs: {result.skipped_jobs}")
    return 0 if result.failed_jobs == 0 else 1


def _validate_msa_retry(args: argparse.Namespace) -> int:
    db_path = args.campaign_dir / "campaign.sqlite"
    if not db_path.exists():
        print(f"error: missing campaign database: {db_path}")
        return 1
    conn = connect_database(db_path)
    try:
        store = CampaignStore(conn)
        retried = store.retry_failed_msa_jobs(
            msa_job_ids=tuple(args.msa_job_id or ()),
            candidate_ids=tuple(args.candidate_id or ()),
            reset_attempt_count=not args.keep_attempt_count,
        )
    finally:
        conn.close()
    print(f"retried_msa_jobs: {retried}")
    if args.msa_job_id:
        print(f"filtered_msa_job_ids: {', '.join(args.msa_job_id)}")
    if args.candidate_id:
        print(f"filtered_candidate_ids: {', '.join(args.candidate_id)}")
    print(
        "attempt_counts_reset: "
        f"{str(not args.keep_attempt_count).lower()}"
    )
    return 0


def _validate_run_mock(args: argparse.Namespace) -> int:
    result = run_mock_validation(
        args.campaign_dir,
        worker_id=args.worker_id,
        gpu_id=args.gpu_id,
        max_tasks=args.max_tasks,
    )
    print(f"completed validation tasks: {result.completed_tasks}")
    print(f"recorded structures: {result.recorded_structures}")
    if result.skipped_no_pending:
        print("no pending validation tasks")
    return 0


def _validate_run(args: argparse.Namespace) -> int:
    try:
        args = _with_validation_yaml_defaults(args, mode="run")
        config = _protenix_runner_config_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    if not args.no_validation_preflight:
        preflight = check_protenix_environment(
            protenix_command=config.protenix_command,
            protenix_root=config.protenix_root,
            checkpoint_dir=config.checkpoint_dir,
            ipsae_script_path=config.ipsae_script_path,
            gpu_id=args.gpu_id,
            require_template_support=_validation_run_requires_template_support(
                args.campaign_dir,
                config=config,
            ),
        )
        if not preflight.ok:
            print("Protenix validation preflight: failed")
            _print_protenix_check(preflight)
            return 1

    print(
        "running Protenix validation worker; first run may load large "
        "checkpoint files before GPU memory increases",
        file=sys.stderr,
        flush=True,
    )
    result = run_local_protenix_validation(
        args.campaign_dir,
        worker_id=args.worker_id,
        gpu_id=args.gpu_id,
        config=config,
    )
    _print_validation_run_result(result)
    return 0 if result.failed_tasks == 0 else 1


def _validation_run_requires_template_support(
    campaign_dir: Path,
    *,
    config: ProtenixRunnerConfig,
) -> bool:
    if config.use_template == "false":
        return False
    if config.use_template == "true":
        return True
    root = Path(campaign_dir)
    pending_design_metrics = _pending_validation_design_metrics(root)
    if not pending_design_metrics:
        return False
    if (root / "target" / "normalized_target.cif").exists():
        return True
    return any(
        _design_metrics_have_builtin_framework_template(metrics)
        for metrics in pending_design_metrics
    )


def _pending_validation_design_metrics(root: Path) -> list[dict[str, Any]]:
    db_path = root / "campaign.sqlite"
    if not db_path.exists():
        return []
    conn = connect_database(db_path)
    try:
        rows = conn.execute(
            """
            SELECT c.design_metrics_json
            FROM validation_tasks AS vt
            JOIN candidates AS c
              ON c.candidate_id = vt.candidate_id
            WHERE vt.status = 'pending'
            """
        ).fetchall()
    finally:
        conn.close()
    metrics: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = yaml.safe_load(row["design_metrics_json"] or "{}")
        except yaml.YAMLError:
            continue
        if isinstance(payload, dict):
            metrics.append(payload)
    return metrics


def _design_metrics_have_builtin_framework_template(metrics: dict[str, Any]) -> bool:
    if str(metrics.get("framework_source") or "").strip().lower() != "builtin":
        return False
    framework = str(
        metrics.get("framework")
        or metrics.get("framework_name")
        or ""
    ).strip()
    if not framework:
        return False
    scaffold = str(metrics.get("binder_scaffold") or "").strip().lower()
    try:
        if scaffold in {"scfv", "sc_fv"}:
            return get_scfv_framework_template_cif(framework) is not None
        if scaffold == "vhh":
            return get_vhh_framework_template_cif(framework) is not None
    except KeyError:
        return False
    return False


def _print_validation_run_result(result) -> None:
    if result.skipped_no_pending:
        print("no pending validation tasks")
    print(f"completed validation tasks: {result.completed_tasks}")
    print(f"failed validation tasks: {result.failed_tasks}")
    if getattr(result, "retryable_failed_attempts", 0):
        print(
            "retryable failed validation attempts: "
            f"{result.retryable_failed_attempts}"
        )
    print(f"skipped validation tasks: {result.skipped_tasks}")
    print(f"recorded structures: {result.recorded_structures}")


def _validate_run_multi(args: argparse.Namespace) -> int:
    try:
        args = _with_validation_yaml_defaults(args, mode="run")
        args = _with_effective_validation_batch_size(
            args,
            worker_count=len(_normalize_validation_gpu_ids(args.gpus)),
        )
        worker_args = _validation_worker_args(args)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    _print_effective_validation_batch_size(args)
    print(
        "running one Protenix validation worker per GPU",
        file=sys.stderr,
        flush=True,
    )
    result = run_multi_validation(
        args.campaign_dir,
        gpu_ids=args.gpus,
        worker_prefix=args.worker_prefix,
        max_tasks_per_worker=args.max_tasks_per_worker,
        poll_interval_seconds=args.poll_interval,
        heartbeat_interval_seconds=args.heartbeat_interval,
        stale_after_seconds=args.stale_timeout,
        worker_args=worker_args,
    )
    _print_validation_multi_result(result)
    return 0 if result.ok else 1


def _print_validation_multi_result(result) -> None:
    print(f"run_id: {result.run_id}")
    print(f"completed validation tasks: {result.completed_tasks}")
    print(f"recovered validation tasks: {result.recovered_tasks}")
    print(f"failed workers: {result.failed_workers}")
    for worker in result.worker_results:
        status = "ok" if worker.ok else "failed"
        print(
            f"worker: {worker.worker_id} gpu={worker.gpu_id} "
            f"status={status} returncode={worker.returncode} "
            f"completed={worker.completed_tasks} recovered={worker.recovered_tasks}"
        )
        print(f"  log: {worker.log_path}")


def _validate_report(args: argparse.Namespace) -> int:
    result = report_validation(args.campaign_dir)
    _print_validation_report_result(result)
    return 0


def _print_validation_report_result(result) -> None:
    multi_model = len(result.model_reports) > 1
    for report in result.model_reports:
        prefix = f"{report.model_name} " if multi_model else ""
        print(f"{prefix}validator_dir: {report.validated_dir}")
        print(f"{prefix}validation_results_csv: {report.manifest_csv}")
        print(f"{prefix}structure_samples_csv: {report.structures_manifest_csv}")
        print(f"{prefix}validation_summary_json: {report.summary_json}")
        print(f"{prefix}validation_tasks: {report.task_rows}")
        print(f"{prefix}validation_structures: {report.structure_rows}")
    if multi_model:
        print(f"total_validation_tasks: {result.task_rows}")
        print(f"total_validation_structures: {result.structure_rows}")


def _analyze(args: argparse.Namespace) -> int:
    try:
        result = analyze_campaign(
            args.campaign_dir,
            top_k=args.top_k,
            max_binder_rmsd_angstrom=args.analysis_max_binder_rmsd_angstrom,
            rmsd_weight=args.analysis_rmsd_weight,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print(f"combined_ranking_csv: {result.combined_ranking_csv}")
    print(f"ranking_diagnostics_csv: {result.diagnostics_csv}")
    print(f"ranking_summary_json: {result.summary_json}")
    print(f"plots_dir: {result.plots_dir}")
    print(f"top_ranked_dir: {result.top_ranked_dir}")
    print(f"ranked_designs: {result.ranked_count}")
    print(f"diagnostic_rows: {result.diagnostic_count}")
    print(f"copied_designs: {result.copied_designs}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    return _run_validation_lifecycle(args)


def _run_validation_lifecycle(args: argparse.Namespace) -> int:
    try:
        args = _with_validation_yaml_defaults(args, mode="run")
        plan_config = _validation_plan_config_from_args(args)
        _protenix_runner_config_from_args(args)
        if args.gpus is not None and args.gpu_id is not None:
            print("error: use --gpu-id or --gpus, not both")
            return 2
        if args.gpus is not None and args.max_tasks is not None:
            print("error: use --max-tasks-per-worker with --gpus")
            return 2
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    if not args.skip_msa_plan:
        msa_plan = plan_msa_prefetch(
            args.campaign_dir,
            validation_config_hash=plan_config.validation_config_hash,
            validation_config_override=_validation_prefetch_config_from_args(args),
            log=lambda message: print(message),
        )
        print(f"msa_candidates: {msa_plan.candidate_count}")
        print(f"queued_msa_jobs: {msa_plan.queued_jobs}")
        print(f"existing_msa_jobs: {msa_plan.existing_jobs}")
        print(f"skipped_msa_candidates: {msa_plan.skipped_candidates}")

    if not args.skip_msa_run:
        msa_run = run_msa_prefetch_worker(
            args.campaign_dir,
            worker_id="validate-msa-worker-0",
            max_jobs=args.msa_max_jobs,
            max_requests_per_minute=args.msa_max_requests_per_minute,
            log=lambda message: print(message),
        )
        if msa_run.no_pending:
            print("no pending MSA jobs")
        print(f"completed_msa_jobs: {msa_run.completed_jobs}")
        print(f"failed_msa_jobs: {msa_run.failed_jobs}")
        print(f"skipped_msa_jobs: {msa_run.skipped_jobs}")
        if msa_run.failed_jobs:
            return 1

    try:
        plan_result = plan_validation_tasks(
            args.campaign_dir,
            config=plan_config,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print(f"validation_model: {plan_result.model_name}")
    print(f"validation_config_hash: {plan_result.validation_config_hash}")
    print(f"candidate_pool: {plan_result.candidate_count}")
    print(f"selected: {plan_result.selected_count}")
    print(f"created: {plan_result.created_count}")
    print(f"existing: {plan_result.existing_count}")

    if args.gpus is not None:
        try:
            args = _with_effective_validation_batch_size(
                args,
                worker_count=len(_normalize_validation_gpu_ids(args.gpus)),
            )
        except ValueError as exc:
            print(f"error: {exc}")
            return 2
        _print_effective_validation_batch_size(args)
        print(
            "running one Protenix validation worker per GPU",
            file=sys.stderr,
            flush=True,
        )
        run_result = run_multi_validation(
            args.campaign_dir,
            gpu_ids=args.gpus,
            worker_prefix=args.worker_prefix,
            max_tasks_per_worker=args.max_tasks_per_worker,
            poll_interval_seconds=args.poll_interval,
            heartbeat_interval_seconds=args.heartbeat_interval,
            stale_after_seconds=args.stale_timeout,
            worker_args=_validation_worker_args(args),
        )
        _print_validation_multi_result(run_result)
        if not run_result.ok:
            return 1
    else:
        run_status = _validate_run(args)
        if run_status != 0:
            return run_status

    if not args.skip_report:
        report = report_validation(args.campaign_dir)
        _print_validation_report_result(report)
    if not args.skip_analysis:
        try:
            analysis = analyze_campaign(
                args.campaign_dir,
                top_k=args.analysis_top_k,
                max_binder_rmsd_angstrom=getattr(
                    args,
                    "analysis_max_binder_rmsd_angstrom",
                    None,
                ),
                rmsd_weight=getattr(args, "analysis_rmsd_weight", None),
            )
        except ValueError as exc:
            print(f"error: {exc}")
            return 2
        print(f"combined_ranking_csv: {analysis.combined_ranking_csv}")
        print(f"ranking_diagnostics_csv: {analysis.diagnostics_csv}")
        print(f"ranking_summary_json: {analysis.summary_json}")
        print(f"top_ranked_dir: {analysis.top_ranked_dir}")
        print(f"ranked_designs: {analysis.ranked_count}")
        print(f"diagnostic_rows: {analysis.diagnostic_count}")
        print(f"copied_designs: {analysis.copied_designs}")
    return 0


def _with_validation_yaml_defaults(
    args: argparse.Namespace,
    *,
    mode: str,
) -> argparse.Namespace:
    defaults = _validation_yaml_defaults(args.campaign_dir)
    merged = argparse.Namespace(**vars(args))
    for name, value in defaults.items():
        if getattr(merged, name, None) is None:
            setattr(merged, name, value)
    for name, value in _validation_builtin_defaults(mode=mode).items():
        if name == "protenix_python" and getattr(merged, "protenix_command", None):
            continue
        if getattr(merged, name, None) is None:
            setattr(merged, name, value)
    return merged


def _validation_builtin_defaults(*, mode: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "validate_model": DEFAULT_VALIDATE_MODEL,
        "min_esm_iptm": DEFAULT_MIN_IPTM,
        "min_validation_iptm": DEFAULT_MIN_IPTM,
        "min_validation_ipsae": None,
        "require_hotspot_contact": "auto",
        "max_attempts": 3,
    }
    if mode in {"plan", "run"}:
        values.update(_validation_hash_builtin_defaults())
    if mode == "run":
        values.update(
            {
                "validation_batch_size": 10,
                "scratch_root": None,
                "keep_validation_debug": False,
                "no_validation_preflight": False,
                "timeout_seconds": 7200,
                "heartbeat_interval": 30.0,
            }
        )
    return values


def _validation_hash_builtin_defaults() -> dict[str, Any]:
    return {
        "validation_hotspot_cutoff_angstrom": None,
        "protenix_command": None,
        "protenix_python": _default_protenix_python(),
        "protenix_root": None,
        "protenix_checkpoint_dir": _default_protenix_checkpoint_dir(),
        "ipsae_script": None,
        "ipsae_python": None,
        "ipsae_pae_cutoff": 15.0,
        "ipsae_dist_cutoff": 15.0,
        "seeds": "101",
        "n_sample": 1,
        "n_step": 200,
        "n_cycle": 10,
        "use_msa": None,
        "use_template": "auto",
        "target_msa_mode": None,
        "binder_msa_mode": "auto",
        "msa_server": None,
        "target_msa_dir": None,
        "target_msa_map_csv": None,
        "msa_cache_root": None,
        "msa_pairing_strategy": "greedy",
        "token_limit": DEFAULT_PROTENIX_TOKEN_LIMIT,
    }


def _validation_yaml_defaults(campaign_dir: Path) -> dict[str, Any]:
    root = Path(campaign_dir)
    validation = _validation_yaml_mapping(root)
    if not validation:
        return {}
    return _validation_defaults_from_mapping(validation, root)


def _validation_defaults_from_mapping(
    validation: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    protenix = _optional_mapping_value(validation, "protenix")
    msa = _optional_mapping_value(validation, "msa")
    defaults: dict[str, Any] = {}

    _set_first_str(
        defaults,
        validation,
        "validate_model",
        ("validate_model", "model", "model_name"),
    )
    if "top_k" in validation or "validate_top_k" in validation:
        defaults["validate_top_k"] = _coerce_top_k(
            _first_present(validation, ("validate_top_k", "top_k")),
            "validation.top_k",
        )
    _set_first_float(defaults, validation, "min_esm_iptm", ("min_esm_iptm",))
    _set_first_float(
        defaults,
        validation,
        "min_validation_iptm",
        ("min_validation_iptm", "min_iptm"),
    )
    _set_first_float(
        defaults,
        validation,
        "min_validation_ipsae",
        ("min_validation_ipsae", "min_ipsae", "min_validation_ipSAE"),
    )
    _set_first_choice(
        defaults,
        validation,
        "require_hotspot_contact",
        ("require_hotspot_contact",),
        choices={"auto", "always", "never"},
    )
    _set_first_int(defaults, validation, "max_attempts", ("max_attempts",))
    _set_first_float(
        defaults,
        validation,
        "validation_hotspot_cutoff_angstrom",
        (
            "validation_hotspot_cutoff_angstrom",
            "hotspot_cutoff_angstrom",
            "hotspot_contact_cutoff_angstrom",
        ),
    )

    runtime = {**validation, **msa, **protenix}
    _set_first_int(
        defaults,
        runtime,
        "validation_batch_size",
        ("validation_batch_size", "batch_size"),
    )
    _set_first_str(defaults, runtime, "protenix_command", ("protenix_command", "command"))
    _set_first_str(defaults, runtime, "protenix_python", ("protenix_python", "python"))
    _set_first_path(
        defaults,
        runtime,
        "protenix_root",
        ("protenix_root", "root"),
        base_dir,
    )
    _set_first_path(
        defaults,
        runtime,
        "protenix_checkpoint_dir",
        ("protenix_checkpoint_dir", "checkpoint_dir"),
        base_dir,
    )
    _set_first_path(defaults, runtime, "ipsae_script", ("ipsae_script",), base_dir)
    _set_first_str(defaults, runtime, "ipsae_python", ("ipsae_python",))
    _set_first_float(defaults, runtime, "ipsae_pae_cutoff", ("ipsae_pae_cutoff",))
    _set_first_float(defaults, runtime, "ipsae_dist_cutoff", ("ipsae_dist_cutoff",))
    _set_first_path(defaults, runtime, "scratch_root", ("scratch_root",), base_dir)
    _set_first_bool(
        defaults,
        runtime,
        "keep_validation_debug",
        ("keep_validation_debug", "keep_debug"),
    )
    if "seeds" in runtime:
        defaults["seeds"] = _coerce_seeds(runtime["seeds"], "validation.protenix.seeds")
    _set_first_int(defaults, runtime, "n_sample", ("n_sample", "samples"))
    _set_first_int(defaults, runtime, "n_step", ("n_step", "steps"))
    _set_first_int(defaults, runtime, "n_cycle", ("n_cycle", "cycles"))
    _set_first_bool(defaults, runtime, "use_msa", ("use_msa",))
    _set_first_template_mode(defaults, runtime, "use_template", ("use_template",))
    _set_first_choice(
        defaults,
        runtime,
        "target_msa_mode",
        ("target_msa_mode", "target"),
        choices={"none", "provided", "server"},
    )
    _set_first_choice(
        defaults,
        runtime,
        "binder_msa_mode",
        ("binder_msa_mode", "binder_msa", "binder"),
        choices={"auto", "none", "single_sequence"},
    )
    _set_first_str(
        defaults,
        runtime,
        "msa_server",
        ("msa_server", "msa_server_url", "server_url"),
    )
    _set_first_path(defaults, runtime, "target_msa_dir", ("target_msa_dir",), base_dir)
    _set_first_path(
        defaults,
        runtime,
        "target_msa_map_csv",
        ("target_msa_map_csv",),
        base_dir,
    )
    _set_first_path(defaults, runtime, "msa_cache_root", ("msa_cache_root",), base_dir)
    _set_first_choice(
        defaults,
        runtime,
        "msa_pairing_strategy",
        ("msa_pairing_strategy",),
        choices={"greedy", "query_only", "copy_non_pairing"},
    )
    _set_first_int(defaults, runtime, "timeout_seconds", ("timeout_seconds",))
    _set_first_float(
        defaults,
        runtime,
        "heartbeat_interval",
        ("heartbeat_interval", "heartbeat_interval_seconds"),
    )
    _set_first_int(defaults, runtime, "token_limit", ("token_limit",))
    if "use_msa" not in defaults and _runtime_implies_use_msa(runtime):
        defaults["use_msa"] = True
    return defaults


def _runtime_implies_use_msa(runtime: dict[str, Any]) -> bool:
    target_msa_mode = runtime.get("target_msa_mode", runtime.get("target"))
    if target_msa_mode in {"provided", "server"}:
        return True
    return any(
        runtime.get(key) is not None
        for key in (
            "msa_server",
            "msa_server_url",
            "server_url",
            "target_msa_dir",
            "target_msa_map_csv",
        )
    )


def _validation_yaml_mapping(campaign_dir: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in ("config.yaml", "resolved_config.yaml"):
        path = campaign_dir / name
        if not path.exists():
            continue
        try:
            payload = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"could not parse {path}: {exc}") from exc
        if not isinstance(payload, dict):
            continue
        validation = payload.get("validation")
        if validation is None:
            continue
        if not isinstance(validation, dict):
            raise ValueError("validation must be a mapping")
        merged = _deep_merge(merged, validation)
    return merged


def _validation_yaml_defaults_for_config(config_path: Path) -> dict[str, Any]:
    path = Path(config_path)
    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"could not parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        return {}
    validation = payload.get("validation")
    if validation is None:
        return {}
    if not isinstance(validation, dict):
        raise ValueError("validation must be a mapping")
    return _validation_defaults_from_mapping(validation, path.parent)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _optional_mapping_value(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"validation.{key} must be a mapping")
    return value


def _first_present(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _set_first_str(
    defaults: dict[str, Any],
    mapping: dict[str, Any],
    dest: str,
    keys: Sequence[str],
) -> None:
    value = _first_present(mapping, keys)
    if value is None:
        return
    defaults[dest] = str(value)


def _set_first_float(
    defaults: dict[str, Any],
    mapping: dict[str, Any],
    dest: str,
    keys: Sequence[str],
) -> None:
    value = _first_present(mapping, keys)
    if value is None:
        return
    try:
        defaults[dest] = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"validation.{keys[0]} must be a number") from exc


def _set_first_int(
    defaults: dict[str, Any],
    mapping: dict[str, Any],
    dest: str,
    keys: Sequence[str],
) -> None:
    value = _first_present(mapping, keys)
    if value is None:
        return
    try:
        defaults[dest] = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"validation.{keys[0]} must be an integer") from exc


def _set_first_bool(
    defaults: dict[str, Any],
    mapping: dict[str, Any],
    dest: str,
    keys: Sequence[str],
) -> None:
    value = _first_present(mapping, keys)
    if value is None:
        return
    if isinstance(value, bool):
        defaults[dest] = value
        return
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            defaults[dest] = True
            return
        if text in {"0", "false", "no", "off"}:
            defaults[dest] = False
            return
    raise ValueError(f"validation.{keys[0]} must be a boolean")


def _set_first_template_mode(
    defaults: dict[str, Any],
    mapping: dict[str, Any],
    dest: str,
    keys: Sequence[str],
) -> None:
    value = _first_present(mapping, keys)
    if value is None:
        return
    defaults[dest] = _coerce_template_mode(value, f"validation.{keys[0]}")


def _coerce_template_mode(value: Any, field_name: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"auto", "true", "false"}:
        return text
    if text in {"1", "yes", "on"}:
        return "true"
    if text in {"0", "no", "off"}:
        return "false"
    raise ValueError(f"{field_name} must be one of: auto, true, false")


def _set_first_choice(
    defaults: dict[str, Any],
    mapping: dict[str, Any],
    dest: str,
    keys: Sequence[str],
    *,
    choices: set[str],
) -> None:
    value = _first_present(mapping, keys)
    if value is None:
        return
    text = str(value)
    if text not in choices:
        joined = ", ".join(sorted(choices))
        raise ValueError(f"validation.{keys[0]} must be one of: {joined}")
    defaults[dest] = text


def _set_first_path(
    defaults: dict[str, Any],
    mapping: dict[str, Any],
    dest: str,
    keys: Sequence[str],
    base_dir: Path,
) -> None:
    value = _first_present(mapping, keys)
    if value is None:
        return
    path = Path(str(value)).expanduser()
    defaults[dest] = path if path.is_absolute() else base_dir / path


def _coerce_top_k(value: Any, field_name: str) -> int | None:
    if isinstance(value, str) and value.strip().lower() == "all":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer or all") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer or all")
    return parsed


def _coerce_seeds(value: Any, field_name: str) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{field_name} must not be empty")
        return ",".join(str(int(item)) for item in value)
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _protenix_runner_config_from_args(args: argparse.Namespace) -> ProtenixRunnerConfig:
    target_msa_mode = _target_msa_mode_from_args(args)
    use_msa = _effective_validation_use_msa(args, target_msa_mode=target_msa_mode)
    return ProtenixRunnerConfig(
        model_name=args.validate_model,
        protenix_command=_protenix_command_from_args(args),
        checkpoint_dir=args.protenix_checkpoint_dir,
        protenix_root=args.protenix_root,
        scratch_root=args.scratch_root,
        keep_debug=args.keep_validation_debug,
        seeds=_parse_protenix_seeds(args.seeds),
        n_sample=args.n_sample,
        n_step=args.n_step,
        n_cycle=args.n_cycle,
        use_msa=use_msa,
        use_template=args.use_template,
        target_msa_mode=target_msa_mode,
        binder_msa_mode=args.binder_msa_mode,
        target_msa_dir=args.target_msa_dir,
        target_msa_map_csv=args.target_msa_map_csv,
        msa_server_url=args.msa_server,
        msa_cache_root=args.msa_cache_root,
        msa_pairing_strategy=args.msa_pairing_strategy,
        timeout_seconds=args.timeout_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval,
        batch_size=args.validation_batch_size,
        max_tasks=args.max_tasks,
        min_validation_iptm=args.min_validation_iptm,
        min_validation_ipsae=args.min_validation_ipsae,
        ipsae_script_path=args.ipsae_script,
        ipsae_python=args.ipsae_python,
        ipsae_pae_cutoff=args.ipsae_pae_cutoff,
        ipsae_dist_cutoff=args.ipsae_dist_cutoff,
        validation_hotspot_cutoff_angstrom=(
            args.validation_hotspot_cutoff_angstrom
        ),
        token_limit=args.token_limit,
    )


def _validation_plan_config_from_args(args: argparse.Namespace) -> ValidationPlanConfig:
    target_msa_mode = _target_msa_mode_from_args(args)
    use_msa = _effective_validation_use_msa(args, target_msa_mode=target_msa_mode)
    protenix_command = _protenix_command_from_args(args)
    return ValidationPlanConfig(
        model_name=args.validate_model,
        top_k=args.validate_top_k,
        min_esm_iptm=args.min_esm_iptm,
        min_validation_iptm=args.min_validation_iptm,
        min_validation_ipsae=args.min_validation_ipsae,
        require_hotspot_contact=args.require_hotspot_contact,
        validation_hotspot_cutoff_angstrom=args.validation_hotspot_cutoff_angstrom,
        protenix_command=protenix_command,
        protenix_python=None if protenix_command is not None else args.protenix_python,
        protenix_root=args.protenix_root,
        checkpoint_dir=args.protenix_checkpoint_dir,
        seeds=_parse_protenix_seeds(args.seeds),
        n_sample=args.n_sample,
        n_step=args.n_step,
        n_cycle=args.n_cycle,
        token_limit=args.token_limit,
        use_msa=use_msa,
        use_template=args.use_template,
        target_msa_mode=target_msa_mode,
        binder_msa_mode=args.binder_msa_mode,
        target_msa_dir=args.target_msa_dir,
        target_msa_map_csv=args.target_msa_map_csv,
        msa_server_url=args.msa_server,
        msa_cache_root=args.msa_cache_root,
        msa_pairing_strategy=args.msa_pairing_strategy,
        ipsae_script_path=args.ipsae_script,
        ipsae_python=args.ipsae_python,
        ipsae_pae_cutoff=args.ipsae_pae_cutoff,
        ipsae_dist_cutoff=args.ipsae_dist_cutoff,
        max_attempts=args.max_attempts,
    )


def _validation_worker_args(args: argparse.Namespace) -> list[str]:
    _protenix_command_from_args(args)
    _parse_protenix_seeds(args.seeds)
    _target_msa_mode_from_args(args)

    worker_args = [
        "--validate-model",
        args.validate_model,
        "--validation-batch-size",
        str(args.validation_batch_size),
        "--seeds",
        args.seeds,
        "--n-sample",
        str(args.n_sample),
        "--n-step",
        str(args.n_step),
        "--n-cycle",
        str(args.n_cycle),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.min_validation_iptm is not None:
        worker_args.extend(["--min-validation-iptm", str(args.min_validation_iptm)])
    if args.min_validation_ipsae is not None:
        worker_args.extend(["--min-validation-ipsae", str(args.min_validation_ipsae)])
    if args.ipsae_script is not None:
        worker_args.extend(["--ipsae-script", str(args.ipsae_script)])
    if args.ipsae_python is not None:
        worker_args.extend(["--ipsae-python", str(args.ipsae_python)])
    if args.ipsae_pae_cutoff != 15.0:
        worker_args.extend(["--ipsae-pae-cutoff", str(args.ipsae_pae_cutoff)])
    if args.ipsae_dist_cutoff != 15.0:
        worker_args.extend(["--ipsae-dist-cutoff", str(args.ipsae_dist_cutoff)])
    if args.validation_hotspot_cutoff_angstrom is not None:
        worker_args.extend(
            [
                "--validation-hotspot-cutoff-angstrom",
                str(args.validation_hotspot_cutoff_angstrom),
            ]
        )
    if args.protenix_command is not None:
        worker_args.extend(["--protenix-command", args.protenix_command])
    if args.protenix_python is not None:
        worker_args.extend(["--protenix-python", str(args.protenix_python)])
    if args.protenix_root is not None:
        worker_args.extend(["--protenix-root", str(args.protenix_root)])
    if args.protenix_checkpoint_dir is not None:
        worker_args.extend(
            ["--protenix-checkpoint-dir", str(args.protenix_checkpoint_dir)]
        )
    if args.scratch_root is not None:
        worker_args.extend(["--scratch-root", str(args.scratch_root)])
    if args.keep_validation_debug:
        worker_args.append("--keep-validation-debug")
    if args.no_validation_preflight:
        worker_args.append("--no-validation-preflight")
    if _effective_validation_use_msa(
        args,
        target_msa_mode=_target_msa_mode_from_args(args),
    ):
        worker_args.append("--use-msa")
    if args.use_template != "auto":
        worker_args.extend(["--use-template", args.use_template])
    if args.target_msa_mode is not None:
        worker_args.extend(["--target-msa-mode", args.target_msa_mode])
    if args.binder_msa_mode != "auto":
        worker_args.extend(["--binder-msa-mode", args.binder_msa_mode])
    if args.msa_server is not None:
        worker_args.extend(["--msa-server", args.msa_server])
    if args.target_msa_dir is not None:
        worker_args.extend(["--target-msa-dir", str(args.target_msa_dir)])
    if args.target_msa_map_csv is not None:
        worker_args.extend(["--target-msa-map-csv", str(args.target_msa_map_csv)])
    if args.msa_cache_root is not None:
        worker_args.extend(["--msa-cache-root", str(args.msa_cache_root)])
    if args.msa_pairing_strategy != "greedy":
        worker_args.extend(["--msa-pairing-strategy", args.msa_pairing_strategy])
    if args.token_limit is not None:
        worker_args.extend(["--token-limit", str(args.token_limit)])
    return worker_args


def _with_effective_validation_batch_size(
    args: argparse.Namespace,
    *,
    worker_count: int,
) -> argparse.Namespace:
    if worker_count <= 0:
        raise ValueError("validation worker count must be positive")
    max_batch_size = int(args.validation_batch_size)
    if max_batch_size <= 0:
        raise ValueError("--validation-batch-size must be positive")
    ready_tasks = _ready_validation_task_count(Path(args.campaign_dir))
    if ready_tasks <= 0:
        effective_batch_size = max_batch_size
    else:
        effective_batch_size = min(
            max_batch_size,
            max(1, (ready_tasks + worker_count - 1) // worker_count),
        )
    merged = argparse.Namespace(**vars(args))
    merged.validation_max_batch_size = max_batch_size
    merged.validation_ready_task_count = ready_tasks
    merged.validation_worker_count = worker_count
    merged.validation_batch_size = effective_batch_size
    return merged


def _ready_validation_task_count(campaign_dir: Path) -> int:
    db_path = Path(campaign_dir) / "campaign.sqlite"
    if not db_path.exists():
        return 0
    conn = connect_database(db_path)
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
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
            """
        ).fetchone()
    finally:
        conn.close()
    return int(row["count"] if row is not None else 0)


def _print_effective_validation_batch_size(args: argparse.Namespace) -> None:
    max_batch_size = getattr(args, "validation_max_batch_size", None)
    if max_batch_size is None:
        return
    print(
        "validation_batch_size: "
        f"{args.validation_batch_size} "
        f"(max={max_batch_size}, ready_tasks={args.validation_ready_task_count}, "
        f"workers={args.validation_worker_count})"
    )


def _validation_prefetch_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    target_msa_mode = _target_msa_mode_from_args(args)
    config: dict[str, Any] = {
        "min_esm_iptm": args.min_esm_iptm,
        "require_hotspot_contact": args.require_hotspot_contact,
        "use_msa": _effective_validation_use_msa(
            args,
            target_msa_mode=target_msa_mode,
        ),
        "use_template": args.use_template,
        "target_msa_mode": target_msa_mode,
        "binder_msa_mode": args.binder_msa_mode,
        "msa_server": args.msa_server,
        "target_msa_dir": _path_override_value(args.target_msa_dir),
        "target_msa_map_csv": _path_override_value(args.target_msa_map_csv),
        "msa_cache_root": _path_override_value(args.msa_cache_root),
        "msa_pairing_strategy": args.msa_pairing_strategy,
    }
    return {key: value for key, value in config.items() if value is not None}


def _path_override_value(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.expanduser().resolve())


def _target_msa_mode_from_args(args: argparse.Namespace) -> str:
    provided_paths = [
        value
        for value in (args.target_msa_dir, args.target_msa_map_csv)
        if value is not None
    ]
    mode = args.target_msa_mode
    if mode is None:
        if args.msa_server is not None:
            mode = "server"
        elif provided_paths:
            mode = "provided"
        else:
            mode = "none"

    if mode == "provided":
        if args.msa_server is not None:
            raise ValueError("--msa-server cannot be used with provided target MSA mode")
        if len(provided_paths) != 1:
            raise ValueError(
                "provided target MSA mode requires exactly one of "
                "--target-msa-dir or --target-msa-map-csv"
            )
    elif mode == "server":
        if provided_paths:
            raise ValueError("provided target MSA paths cannot be used with server mode")
        if args.msa_server is None:
            raise ValueError("server target MSA mode requires --msa-server")
    elif mode == "none":
        if args.msa_server is not None or provided_paths:
            raise ValueError(
                "target MSA paths or --msa-server require --target-msa-mode "
                "provided/server, or no explicit mode"
            )
    return mode


def _effective_validation_use_msa(
    args: argparse.Namespace,
    *,
    target_msa_mode: str,
) -> bool:
    use_msa = getattr(args, "use_msa", None)
    if use_msa is not None:
        return bool(use_msa)
    return target_msa_mode != "none"


def _protenix_command_from_args(args: argparse.Namespace) -> tuple[str, ...] | None:
    if args.protenix_command is not None and args.protenix_python is not None:
        raise ValueError("use --protenix-command or --protenix-python, not both")
    if args.protenix_command is not None:
        parts = shlex.split(args.protenix_command)
        if not parts:
            raise ValueError("--protenix-command cannot be empty")
        return tuple(parts)
    if args.protenix_python is not None:
        return (str(args.protenix_python), "-m", "runner.inference")
    protenix_python = _default_protenix_python()
    if protenix_python is not None:
        return (protenix_python, "-m", "runner.inference")
    return None


def _default_protenix_python() -> str | None:
    value = os.environ.get("PROTENIX_PYTHON")
    if value is None or not value.strip():
        return None
    return value


def _default_protenix_checkpoint_dir() -> Path | None:
    value = os.environ.get("PROTENIX_CHECKPOINT_DIR")
    if value is None or not value.strip():
        return None
    return Path(value)


def _parse_protenix_seeds(value: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for raw in str(value).split(","):
        text = raw.strip()
        if not text:
            continue
        try:
            seed = int(text)
        except ValueError as exc:
            raise ValueError("--seeds must be a comma-separated list of integers") from exc
        if seed < 0:
            raise ValueError("--seeds must contain non-negative integers")
        seeds.append(seed)
    if not seeds:
        raise ValueError("--seeds must include at least one integer")
    return tuple(seeds)


def _model_help() -> str:
    aliases = ", ".join(sorted(ESMFOLD2_MODEL_ALIASES))
    return f"model alias or full ESMFold2 model name; aliases: {aliases}"


def _parse_validate_top_k(value: str) -> int | None:
    text = value.strip().lower()
    if text == "all":
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--validate-top-k must be a positive integer or all"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "--validate-top-k must be a positive integer or all"
        )
    return parsed


def _print_next_commands(campaign_dir: Path) -> None:
    campaign_text = shlex.quote(str(campaign_dir))
    esm_repo_arg = _next_command_esm_repo_arg()
    print("next single GPU:")
    print(f"  esmfold2-pipeline run {campaign_text}{esm_repo_arg} --gpu-id 0")
    print("next multi GPU:")
    print(f"  esmfold2-pipeline run-multi {campaign_text}{esm_repo_arg} --gpus all")
    print("next status:")
    print(f"  esmfold2-pipeline status {campaign_text}")


def _next_command_esm_repo_arg() -> str:
    esm_repo = os.environ.get("ESM_REPO")
    if esm_repo:
        return f" --esm-repo {shlex.quote(esm_repo)}"
    return " --esm-repo /path/to/esm"


def _print_status(status: CampaignStatus) -> None:
    print("tables:")
    for name, count in status.table_counts.items():
        print(f"  {name}: {count}")

    _print_counts("shards", status.shard_status_counts)
    _print_counts("candidates", status.candidate_status_counts)
    _print_counts("critics", status.critic_status_counts)
    _print_counts("validation_tasks", status.validation_status_counts)
    _print_counts(
        "validation_structures",
        status.validation_structure_status_counts,
    )
    _print_counts("validation_msa_jobs", status.validation_msa_status_counts)
    _print_counts(
        "validation_tasks_blocked_by_msa",
        status.validation_msa_blocked_counts,
    )
    if status.validation_msa_failures:
        print(f"validation_msa_failures: {len(status.validation_msa_failures)}")
        for failure in status.validation_msa_failures[:5]:
            candidate_text = ",".join(failure.candidate_ids[:5]) or "none"
            if len(failure.candidate_ids) > 5:
                candidate_text += ",..."
            print(
                f"  {failure.msa_job_id} scope={failure.scope} "
                f"candidates={candidate_text}"
            )
            if failure.error_message:
                print(f"    {failure.error_message}")
    _print_counts("attempts", status.attempt_status_counts)
    print(f"missing_artifacts: {status.missing_artifact_count}")
    print(f"untracked_artifacts: {status.untracked_artifact_count}")

    if not status.issues:
        print("issues: none")
        return

    print(f"issues: {len(status.issues)}")
    for issue in status.issues:
        row = f"  {issue.kind}: {issue.path}"
        if issue.table:
            row += f" [{issue.table}:{issue.row_id}]"
        print(row)
        print(f"    {issue.message}")


def _print_config_check(result: ConfigCheckResult) -> None:
    print(f"ok: {str(result.ok).lower()}")
    print(f"config: {result.config_path}")
    if result.config is not None:
        config = result.config
        print(f"target: {config.target_name}")
        print(f"binder_scaffold: {config.binder.scaffold}")
        if config.binder.framework_names:
            print(f"binder_frameworks: {', '.join(config.binder.framework_names)}")
        if config.binder.length_range is not None:
            low, high = config.binder.length_range
            print(f"binder_length: {low}-{high}")
        print(f"inversion_model: {config.inversion_model_name}")
        print(f"critic: {config.critic_name}")
        print(f"designs: {len(config.seeds)}")
        print(f"steps: {config.steps}")
        print(f"output: {config.output}")
        if config.target_structure is not None:
            target_structure = config.target_structure
            print(f"target_structure: {target_structure.path}")
            print(
                "target_conditioning: "
                f"{target_structure.conditioning_mode}"
            )
            print(
                "target_conditioning_assembly: "
                f"{str(target_structure.conditioning_assembly).lower()}"
            )
            if target_structure.conditioning_chain_pairs is not None:
                chain_pairs = ",".join(
                    f"{left}-{right}"
                    for left, right in target_structure.conditioning_chain_pairs
                )
                print(f"target_conditioning_chain_pairs: {chain_pairs}")
    if result.prepared_target is not None:
        prepared_target = result.prepared_target
        print(f"target_format: {prepared_target.input_format}")
        for chain in prepared_target.chains:
            hotspots = ",".join(str(index) for index in chain.hotspot_indices)
            print(
                f"target_chain: {chain.canonical_chain_id} "
                f"length={len(chain.residues)} "
                f"auth={chain.auth_asym_id} label={chain.label_asym_id} "
                f"hotspots={hotspots or 'none'}"
            )

    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"  {warning}")
    else:
        print("warnings: none")

    if result.errors:
        print("errors:")
        for error in result.errors:
            print(f"  {error}")
    else:
        print("errors: none")


def _print_env_check(result) -> None:
    print(f"ok: {str(result.ok).lower()}")
    print("checks:")
    for name, value in result.checks.items():
        print(f"  {name}: {value}")
    if not result.errors:
        print("errors: none")
        return
    print("errors:")
    for error in result.errors:
        print(f"  {error}")


def _print_protenix_check(result) -> None:
    print(f"ok: {str(result.ok).lower()}")
    print("checks:")
    for name, value in result.checks.items():
        print(f"  {name}: {value}")
    if not result.errors:
        print("errors: none")
        return
    print("errors:")
    for error in result.errors:
        print(f"  {error}")


def _print_counts(label: str, counts: dict[str, int]) -> None:
    if not counts:
        print(f"{label}: none")
        return

    text = ", ".join(f"{status}={count}" for status, count in counts.items())
    print(f"{label}: {text}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
