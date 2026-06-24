"""Adapters that isolate optional ESMFold2 dependencies."""

from typing import TYPE_CHECKING, Any

from esmfold2_pipeline.esm_adapter.imports import EnvCheck, check_environment

if TYPE_CHECKING:
    from esmfold2_pipeline.esm_adapter.binder_design import (
        DesignCandidateArtifact,
        ModelPreflightResult,
    )

__all__ = [
    "DesignCandidateArtifact",
    "EnvCheck",
    "ModelPreflightResult",
    "check_environment",
    "preflight_models",
    "run_binder_design_artifact",
]


def __getattr__(name: str) -> Any:
    if name in {
        "DesignCandidateArtifact",
        "ModelPreflightResult",
        "preflight_models",
        "run_binder_design_artifact",
    }:
        from esmfold2_pipeline.esm_adapter import binder_design

        return getattr(binder_design, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
