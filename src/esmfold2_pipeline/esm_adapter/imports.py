from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvCheck:
    ok: bool
    checks: dict[str, Any]
    errors: list[str]

    def to_json(self) -> str:
        return json.dumps(
            {"ok": self.ok, "checks": self.checks, "errors": self.errors},
            indent=2,
            sort_keys=True,
        )


def check_environment(
    *,
    esm_repo: str | Path | None = None,
    require_cuda: bool = True,
    require_tutorial: bool = True,
    require_local_runtime: bool = False,
) -> EnvCheck:
    """Check optional ESM/Torch/CUDA dependencies without importing pipeline code."""

    errors: list[str] = []
    checks: dict[str, Any] = {}

    if esm_repo is not None:
        repo = Path(esm_repo).expanduser().resolve()
        checks["esm_repo"] = str(repo)
        if not repo.exists():
            errors.append(f"ESM repo does not exist: {repo}")
        else:
            _prepend_sys_path(repo)

    try:
        torch = importlib.import_module("torch")
        checks["torch"] = {
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
        if torch.cuda.is_available():
            current = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(current)
            checks["torch"]["current_device"] = int(current)
            checks["torch"]["device_name"] = torch.cuda.get_device_name(current)
            checks["torch"]["total_memory_gb"] = round(
                props.total_memory / (1024**3),
                2,
            )
        elif require_cuda:
            errors.append("CUDA is not available to torch")
    except Exception as exc:
        errors.append(f"Could not import torch: {exc}")

    try:
        esm = importlib.import_module("esm")
        checks["esm"] = {
            "path": str(Path(esm.__file__).resolve()) if getattr(esm, "__file__", None) else None,
            "version": getattr(esm, "__version__", "unknown"),
        }
    except Exception as exc:
        errors.append(f"Could not import esm: {exc}")

    if require_local_runtime:
        _check_local_runtime_apis(checks, errors)

    tutorial_path = _find_binder_design(esm_repo)
    checks["binder_design_py"] = str(tutorial_path) if tutorial_path else None
    if require_tutorial and tutorial_path is None:
        errors.append(
            "Could not find cookbook/tutorials/binder_design.py; pass --esm-repo"
        )

    return EnvCheck(ok=not errors, checks=checks, errors=errors)


def _check_local_runtime_apis(checks: dict[str, Any], errors: list[str]) -> None:
    runtime_checks: dict[str, bool] = {}
    required_symbols = {
        "ESMFold2ExperimentalModel": (
            "transformers.models.esmfold2.modeling_esmfold2_experimental",
            "ESMFold2ExperimentalModel",
        ),
        "ESMCForMaskedLM": (
            "transformers.models.esmc.modeling_esmc",
            "ESMCForMaskedLM",
        ),
        "ESMCTokenizer": (
            "transformers.models.esmc.tokenization_esmc",
            "ESMCTokenizer",
        ),
        "esmfold2_seed_context": (
            "transformers.models.esmfold2.modeling_esmfold2_common",
            "_seed_context",
        ),
        "esmfold2_cue_available": (
            "transformers.models.esmfold2.modeling_esmfold2_common",
            "CUE_AVAILABLE",
        ),
        "ProteinInput": ("esm.models.esmfold2", "ProteinInput"),
        "StructurePredictionInput": (
            "esm.models.esmfold2",
            "StructurePredictionInput",
        ),
        "load_ccd": ("esm.models.esmfold2", "load_ccd"),
        "prepare_esmfold2_input": (
            "esm.models.esmfold2",
            "prepare_esmfold2_input",
        ),
        "ELEMENT_NUMBER_TO_SYMBOL": (
            "esm.models.esmfold2",
            "ELEMENT_NUMBER_TO_SYMBOL",
        ),
        "PROTEIN_3TO1": ("esm.models.esmfold2.constants", "PROTEIN_3TO1"),
        "RES_TYPE_TO_CCD": ("esm.models.esmfold2.constants", "RES_TYPE_TO_CCD"),
        "MOL_TYPE_NONPOLYMER": (
            "esm.models.esmfold2.constants",
            "MOL_TYPE_NONPOLYMER",
        ),
        "ProteinChain": (
            "esm.utils.structure.protein_chain",
            "ProteinChain",
        ),
        "ProteinComplex": (
            "esm.utils.structure.protein_complex",
            "ProteinComplex",
        ),
        "biotite_structure": ("biotite.structure", "Atom"),
        "biotite_structure_array": ("biotite.structure", "array"),
        "biotite_structure_chain_iter": ("biotite.structure", "chain_iter"),
    }
    for name, (module_name, symbol_name) in required_symbols.items():
        try:
            module = importlib.import_module(module_name)
            getattr(module, symbol_name)
        except Exception as exc:
            runtime_checks[name] = False
            errors.append(
                "Could not import local ESM runtime API "
                f"{name} from {module_name}: {exc}"
            )
        else:
            runtime_checks[name] = True

    required_attributes = {
        "ProteinChain.from_atomarray": (
            "esm.utils.structure.protein_chain",
            "ProteinChain",
            "from_atomarray",
        ),
        "ProteinComplex.from_chains": (
            "esm.utils.structure.protein_complex",
            "ProteinComplex",
            "from_chains",
        ),
    }
    for name, (module_name, symbol_name, attribute_name) in required_attributes.items():
        try:
            module = importlib.import_module(module_name)
            symbol = getattr(module, symbol_name)
            getattr(symbol, attribute_name)
        except Exception as exc:
            runtime_checks[name] = False
            errors.append(
                "Could not import local ESM runtime API "
                f"{name} from {module_name}: {exc}"
            )
        else:
            runtime_checks[name] = True
    checks["local_runtime"] = runtime_checks


def load_binder_design_module(esm_repo: str | Path | None = None):
    """Load the ESM tutorial binder_design module, with a small Modal stub if needed."""

    if esm_repo is not None:
        _prepend_sys_path(Path(esm_repo).expanduser().resolve())

    tutorial_path = _find_binder_design(esm_repo)
    if tutorial_path is None:
        raise RuntimeError("Could not find cookbook/tutorials/binder_design.py")

    _ensure_modal_stub()
    module_name = "esmfold2_pipeline_external_binder_design"
    spec = importlib.util.spec_from_file_location(module_name, tutorial_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {tutorial_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_binder_design(esm_repo: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if esm_repo is not None:
        repo = Path(esm_repo).expanduser().resolve()
        candidates.append(repo / "cookbook" / "tutorials" / "binder_design.py")

    try:
        esm = importlib.import_module("esm")
        esm_root = Path(esm.__file__).resolve().parents[1]
        candidates.append(esm_root / "cookbook" / "tutorials" / "binder_design.py")
    except Exception:
        pass

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _prepend_sys_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _ensure_modal_stub() -> None:
    try:
        importlib.import_module("modal")
        return
    except Exception:
        pass

    modal = types.ModuleType("modal")
    modal.Image = _ModalImage
    modal.Volume = _ModalVolume
    modal.App = _ModalApp
    modal.parameter = lambda default=None: default
    modal.enter = lambda *args, **kwargs: _identity_decorator
    modal.method = lambda *args, **kwargs: _identity_decorator
    sys.modules["modal"] = modal


def _identity_decorator(fn):
    return fn


class _ModalImage:
    @classmethod
    def micromamba(cls, *args, **kwargs):
        return cls()

    def run_commands(self, *args, **kwargs):
        return self

    def micromamba_install(self, *args, **kwargs):
        return self

    def pip_install(self, *args, **kwargs):
        return self

    def env(self, *args, **kwargs):
        return self


class _ModalVolume:
    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()


class _ModalApp:
    def __init__(self, *args, **kwargs):
        pass

    def cls(self, *args, **kwargs):
        return _identity_decorator

    def local_entrypoint(self, *args, **kwargs):
        return _identity_decorator
