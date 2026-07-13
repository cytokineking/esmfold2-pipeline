from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy" / "image"


@pytest.mark.parametrize("script_name", ["bootstrap-image.sh", "qualify-image.sh"])
def test_image_scripts_are_valid_bash(script_name: str) -> None:
    result = subprocess.run(
        ["bash", "-n", str(DEPLOY / script_name)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_bootstrap_records_sanitized_reproducibility_evidence() -> None:
    script = (DEPLOY / "bootstrap-image.sh").read_text()

    assert "map({name,version}) | sort_by(.name)" in script
    assert "editable_project_location" not in script
    assert "esmfold2-pipeline-sbom.json" in script
    assert "esmfold2-os-packages.txt" in script
    assert "esmfold2-models.sha256" in script
    assert "protenix_checkpoint_sha256" in script
    assert "protenix_cuda_runtime" in script
    assert "protenix_cuequivariance_version" in script
    assert 'CUDA_VERSION="${ESMFOLD2_IMAGE_CUDA_VERSION:-12.8}"' in script
    assert 'EXPECTED_TORCH_BACKEND="cu130"' in script
    assert 'EXPECTED_ACCELERATOR_BACKEND="cu13"' in script
    assert "CUDA 13 removes offline compilation support for pre-Ampere targets" in script
    assert '--argjson cuda_arch_targets "${CUDA_ARCH_TARGETS_JSON}"' in script
    assert '22.04) CUDA_REPOSITORY_DISTRO="ubuntu2204"' in script
    assert '24.04) CUDA_REPOSITORY_DISTRO="ubuntu2404"' in script
    assert "repos/${CUDA_REPOSITORY_DISTRO}/x86_64" in script
    assert 'STALE_CUDA_PACKAGE_SUFFIX="12-8"' in script
    assert 'rm -rf "/usr/local/cuda-${STALE_CUDA_VERSION}"' in script
    assert '"cuda-nvcc-${CUDA_PACKAGE_SUFFIX}"' in script
    assert '"cuda-cuobjdump-${CUDA_PACKAGE_SUFFIX}"' in script
    assert '[[ ! -x "${CUDA_TOOLKIT_ROOT}/bin/cuobjdump" ]]' in script
    assert 'ESMFOLD2_TORCH_BACKEND="${TORCH_BACKEND}"' in script
    assert "torch_backend" in script
    assert "accelerator_backend" in script
    assert "model_manifest_sha256" in script
    assert "uv_lock_sha256" in script
    assert '--argjson models_preloaded "${models_preloaded_json}"' in script
    assert "models_preloaded:$models_preloaded" in script
    assert 'DECLARED_DISK_GB="${ESMFOLD2_IMAGE_DISK_GB:-}"' in script
    assert 'MINIMUM_ROOT_GB="${ESMFOLD2_IMAGE_MIN_ROOT_GB:-}"' in script
    assert 'utilities:["cuobjdump","curl"' in script


def test_installer_pins_protenix_and_selects_its_accelerator_backend() -> None:
    script = (ROOT / "install.sh").read_text()

    assert (
        "git+https://github.com/cytokineking/Protenix.git@"
        "2a4a6a516466fe3b1f830f515875da65ebcec049"
    ) in script
    assert (
        'CUEQUIVARIANCE_VERSION="${ESMFOLD2_CUEQUIVARIANCE_VERSION:-0.10.0}"'
        in script
    )
    assert "PROTENIX_ACCELERATOR_SPECS_RAW" in script
    assert '"$PROTENIX_SOURCE" "${PROTENIX_ACCELERATOR_SPECS[@]}"' in script


def test_full_qualification_verifies_evidence_models_and_extension() -> None:
    script = (DEPLOY / "qualify-image.sh").read_text()

    assert 'sha256sum --quiet -c "${model_manifest}"' in script
    assert 'cmp "${os_packages}"' in script
    assert ".cuda_runtime == .protenix_cuda_runtime" in script
    assert "actual_cuda_compiler" in script
    assert '.torch_backend == "cu128" or .torch_backend == "cu130"' in script
    assert '.accelerator_backend == "cu12" or .accelerator_backend == "cu13"' in script
    assert 'expected = f"cuequivariance-ops-torch-{sys.argv[1]}"' in script
    assert (
        'for accelerator_python in "${CHECKOUT}/.venv/bin/python" '
        '"${PROTENIX_PYTHON}"' in script
    )
    assert "test ! -e /usr/local/cuda-12.8" in script
    assert "test ! -e /usr/local/cuda-13.0" in script
    assert "done < <(jq -r '.models_preloaded[]'" in script
    assert "miniprotein.yaml vhh.yaml scfv.yaml" in script
    assert "PROTENIX_EXTENSION_CUDART" in script
    assert '"${protenix_site_packages}/protenix/model/layer_norm"' in script
    assert "/etc/esmfold2-pipeline/esmfold2-pipeline-qualification.json" in script
    assert "grep -q 'sm_100'" in script
    assert "grep -q 'sm_120'" in script
    assert "package_inventory_verified:true" in script
    assert "model_inventory_verified:true" in script
    assert "remote_round_trip:$remote_qualified" in script
