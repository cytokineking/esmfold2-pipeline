from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence


@dataclass(frozen=True)
class ProtenixEnvCheck:
    ok: bool
    checks: dict[str, Any]
    errors: list[str]

    def to_json(self) -> str:
        return json.dumps(
            {"ok": self.ok, "checks": self.checks, "errors": self.errors},
            indent=2,
            sort_keys=True,
        )


def check_protenix_environment(
    *,
    protenix_command: Sequence[str] | None = None,
    protenix_python: str | Path | None = None,
    protenix_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    ipsae_script_path: str | Path | None = None,
    gpu_id: str | None = None,
    require_template_support: bool = False,
) -> ProtenixEnvCheck:
    errors: list[str] = []
    checks: dict[str, Any] = {}

    root = Path(protenix_root).expanduser().resolve() if protenix_root else None
    if root is not None:
        checks["protenix_root"] = str(root)
        if not root.exists():
            errors.append(f"Protenix root does not exist: {root}")

    command = _protenix_check_command(
        protenix_command=protenix_command,
        protenix_python=protenix_python,
    )
    checks["command"] = list(command)
    executable_error = _executable_error(command[0])
    if executable_error:
        errors.append(executable_error)

    checkpoint = _checkpoint_dir(checkpoint_dir)
    checks["checkpoint_dir"] = str(checkpoint) if checkpoint is not None else None
    if checkpoint is not None and not checkpoint.exists():
        errors.append(f"Protenix checkpoint directory does not exist: {checkpoint}")
    if checkpoint is not None and checkpoint.exists():
        checkpoint_file = checkpoint / "protenix-v2.pt"
        checks["checkpoint_file"] = str(checkpoint_file)
        if not checkpoint_file.exists():
            errors.append(f"Protenix checkpoint file does not exist: {checkpoint_file}")

    ipsae_script = _ipsae_script_path(ipsae_script_path)
    checks["ipsae_script"] = str(ipsae_script) if ipsae_script is not None else None
    if ipsae_script_path is not None and (
        ipsae_script is None or not ipsae_script.exists()
    ):
        errors.append(f"ipSAE script does not exist: {Path(ipsae_script_path).expanduser()}")

    cuda_check = _cuda_visibility_check(gpu_id)
    checks["cuda"] = cuda_check["checks"]
    errors.extend(cuda_check["errors"])

    if not executable_error and _should_check_runner_import(command):
        import_check = _run_runner_import_check(
            command[0],
            protenix_root=root,
            require_template_support=require_template_support,
        )
        checks["imports"] = import_check.get("checks", {})
        errors.extend(str(error) for error in import_check.get("errors", []))

    return ProtenixEnvCheck(ok=not errors, checks=checks, errors=errors)


def _protenix_check_command(
    *,
    protenix_command: Sequence[str] | None,
    protenix_python: str | Path | None,
) -> tuple[str, ...]:
    if protenix_command is not None and protenix_python is not None:
        raise ValueError("use protenix_command or protenix_python, not both")
    if protenix_command is not None:
        command = tuple(str(part) for part in protenix_command)
        if not command:
            raise ValueError("protenix_command cannot be empty")
        return command
    python = str(protenix_python) if protenix_python is not None else sys.executable
    return (python, "-m", "runner.inference")


def _checkpoint_dir(value: str | Path | None) -> Path | None:
    raw = value if value is not None else os.environ.get("PROTENIX_CHECKPOINT_DIR")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _executable_error(value: str) -> str | None:
    path = Path(value).expanduser()
    if path.parent != Path(".") or value.startswith("."):
        return None if path.exists() else f"Protenix executable does not exist: {path}"
    if shutil.which(value) is None:
        return f"Protenix executable is not on PATH: {value}"
    return None


def _should_check_runner_import(command: Sequence[str]) -> bool:
    if len(command) >= 3 and command[1] == "-m" and command[2] == "runner.inference":
        return True
    name = Path(command[0]).name.lower()
    return len(command) == 1 and name.startswith("python")


def _ipsae_script_path(value: str | Path | None) -> Path | None:
    if value is not None:
        return Path(value).expanduser().resolve()
    for env_name in ("ESMFOLD2_IPSAE_SCRIPT", "PROTENIX_IPSAE_SCRIPT"):
        raw = os.environ.get(env_name)
        if raw:
            candidate = Path(raw).expanduser().resolve()
            if candidate.exists():
                return candidate
    bundled = Path(__file__).with_name("ipsae.py")
    if bundled.exists():
        return bundled.resolve()
    return None


def _cuda_visibility_check(gpu_id: str | None) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    checks: dict[str, Any] = {
        "requested_gpu_id": gpu_id,
        "CUDA_VISIBLE_DEVICES": visible,
    }
    errors: list[str] = []
    if gpu_id is None:
        return {"checks": checks, "errors": errors}

    requested = str(gpu_id).strip()
    if not requested:
        errors.append("GPU ID is empty")
        return {"checks": checks, "errors": errors}
    if visible is None:
        return {"checks": checks, "errors": errors}

    text = visible.strip()
    if text in {"", "-1"}:
        errors.append(
            f"CUDA_VISIBLE_DEVICES={visible!r} hides requested GPU {requested}"
        )
        return {"checks": checks, "errors": errors}
    if text.lower() in {"all", "none"}:
        if text.lower() == "none":
            errors.append("CUDA_VISIBLE_DEVICES=none hides all GPUs")
        return {"checks": checks, "errors": errors}
    visible_ids = {part.strip() for part in text.split(",") if part.strip()}
    if requested not in visible_ids and requested.isdigit():
        errors.append(
            f"requested GPU {requested} is not in CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return {"checks": checks, "errors": errors}


def _run_runner_import_check(
    python_executable: str,
    *,
    protenix_root: Path | None,
    require_template_support: bool = False,
) -> dict[str, Any]:
    code = r"""
import importlib.util
import json
import shutil
import sys

checks = {}
errors = []
checks["ninja"] = {"path": shutil.which("ninja")}
required = ["runner.inference", "protenix"]
if "--require-template-support" in sys.argv:
    required.append("protenix.data.template.structural_template")
for name in required:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        checks[name] = {"ok": False, "error": str(exc)}
        if name in {"runner.inference", "protenix.data.template.structural_template"}:
            errors.append(f"Could not inspect {name}: {exc}")
    else:
        checks[name] = {
            "ok": spec is not None,
            "origin": getattr(spec, "origin", None) if spec is not None else None,
        }
        if name == "runner.inference" and spec is None:
            errors.append("Could not import runner.inference")
        if name == "protenix.data.template.structural_template" and spec is None:
            errors.append(
                "Protenix environment does not expose structural template support; "
                "use the cytokineking/Protenix fork or run with --use-template false"
            )

if checks.get("protenix", {}).get("ok"):
    required_runtime_modules = ["accelerate"]
    for name in required_runtime_modules:
        try:
            spec = importlib.util.find_spec(name)
        except Exception as exc:
            checks[name] = {"ok": False, "error": str(exc)}
            errors.append(f"Could not inspect {name}: {exc}")
        else:
            checks[name] = {
                "ok": spec is not None,
                "origin": getattr(spec, "origin", None) if spec is not None else None,
            }
            if spec is None:
                errors.append(f"Protenix environment cannot import {name}")

if (
    checks["ninja"]["path"] is None
    and checks.get("runner.inference", {}).get("ok")
    and checks.get("protenix", {}).get("ok")
):
    errors.append("Protenix environment cannot find ninja on PATH")

print(json.dumps({"checks": checks, "errors": errors}, sort_keys=True))
"""
    env = os.environ.copy()
    _prepend_executable_dir_to_path(env, python_executable)
    if protenix_root is not None:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(protenix_root)
            if not existing
            else f"{protenix_root}{os.pathsep}{existing}"
        )
    proc = subprocess.run(
        [python_executable, "-c", code]
        + (["--require-template-support"] if require_template_support else []),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if proc.returncode != 0:
        return {
            "checks": {},
            "errors": [
                "Protenix import check failed: "
                + " ".join((proc.stderr or proc.stdout).split())[-1000:]
            ],
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"checks": {}, "errors": [f"Protenix import check returned invalid JSON: {exc}"]}
    return payload if isinstance(payload, dict) else {"checks": {}, "errors": []}


def _prepend_executable_dir_to_path(env: dict[str, str], executable: str) -> None:
    path = Path(executable).expanduser()
    if path.parent == Path(".") or not path.parent.exists():
        return
    existing = env.get("PATH")
    env["PATH"] = str(path.parent) if not existing else f"{path.parent}{os.pathsep}{existing}"
