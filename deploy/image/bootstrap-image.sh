#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible bootstrap for a reusable ESMFold2-pipeline machine image.
# Run as root on a fresh Ubuntu GPU image. The checkout ref defaults to
# origin/main; set an immutable ESMFOLD2_PIPELINE_REF for reproducible builds.

REPOSITORY="${ESMFOLD2_PIPELINE_REPOSITORY:-https://github.com/cytokineking/esmfold2-pipeline.git}"
REF="${ESMFOLD2_PIPELINE_REF:-origin/main}"
CHECKOUT="${ESMFOLD2_PIPELINE_CHECKOUT:-/opt/esmfold2-pipeline}"
PREFIX="${ESMFOLD2_INSTALL_PREFIX:-/opt/esmfold2}"
DECLARED_DISK_GB="${ESMFOLD2_IMAGE_DISK_GB:-}"
MINIMUM_ROOT_GB="${ESMFOLD2_IMAGE_MIN_ROOT_GB:-}"
CUDA_VERSION="${ESMFOLD2_IMAGE_CUDA_VERSION:-12.8}"
PRELOAD_MODELS_RAW="${ESMFOLD2_PRELOAD_MODELS:-fast}"
CLEAN_USER_CACHES="${ESMFOLD2_IMAGE_CLEAN_USER_CACHES:-0}"

for disk_value in DECLARED_DISK_GB MINIMUM_ROOT_GB; do
  if [[ -n "${!disk_value}" && ! "${!disk_value}" =~ ^[0-9]+$ ]]; then
    echo "${disk_value} must be an integer when set" >&2
    exit 2
  fi
done
if [[ -z "${PRELOAD_MODELS_RAW//[[:space:],]/}" ]]; then
  echo "ESMFOLD2_PRELOAD_MODELS must select at least one model" >&2
  exit 2
fi
if [[ "${CLEAN_USER_CACHES}" != 0 && "${CLEAN_USER_CACHES}" != 1 ]]; then
  echo "ESMFOLD2_IMAGE_CLEAN_USER_CACHES must be 0 or 1" >&2
  exit 2
fi

case "${CUDA_VERSION}" in
  12.8)
    CUDA_PACKAGE_SUFFIX="12-8"
    STALE_CUDA_VERSION="13.0"
    STALE_CUDA_PACKAGE_SUFFIX="13-0"
    EXPECTED_TORCH_BACKEND="cu128"
    EXPECTED_ACCELERATOR_BACKEND="cu12"
    CUDA_ARCH_TARGETS_JSON='["sm_70","sm_80","sm_86","sm_89","sm_90","sm_100","sm_120"]'
    MINIMUM_DRIVER_MAJOR=570
    ;;
  13.0)
    CUDA_PACKAGE_SUFFIX="13-0"
    STALE_CUDA_VERSION="12.8"
    STALE_CUDA_PACKAGE_SUFFIX="12-8"
    EXPECTED_TORCH_BACKEND="cu130"
    EXPECTED_ACCELERATOR_BACKEND="cu13"
    # CUDA 13 removes offline compilation support for pre-Ampere targets.
    CUDA_ARCH_TARGETS_JSON='["sm_80","sm_86","sm_89","sm_90","sm_100","sm_120"]'
    MINIMUM_DRIVER_MAJOR=580
    ;;
  *)
    echo "ESMFOLD2_IMAGE_CUDA_VERSION must be 12.8 or 13.0" >&2
    exit 2
    ;;
esac
TORCH_BACKEND="${ESMFOLD2_TORCH_BACKEND:-${EXPECTED_TORCH_BACKEND}}"
if [[ "${TORCH_BACKEND}" != "${EXPECTED_TORCH_BACKEND}" ]]; then
  echo "CUDA ${CUDA_VERSION} requires ESMFOLD2_TORCH_BACKEND=${EXPECTED_TORCH_BACKEND}" >&2
  exit 2
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "bootstrap-image.sh must run as root" >&2
  exit 1
fi

# Existing image builders may opt in to removing duplicate per-user caches.
# This never removes arbitrary user projects or files.
if [[ "${CLEAN_USER_CACHES}" -eq 1 ]]; then
  for user_home in /home/*; do
    [[ -d "${user_home}" ]] || continue
    rm -rf \
      "${user_home}/.cache/huggingface" \
      "${user_home}/.cache/torch_extensions" \
      "${user_home}/.cache/torchinductor" \
      "${user_home}/.cache/uv" \
      "${user_home}/.triton"
  done
fi
rm -rf \
  /root/.cache/torch_extensions \
  /root/.cache/torchinductor \
  /root/.triton \
  /tmp/torch_extensions*

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git hmmer jq lsof pigz procps rclone sqlite3 tmux \
  unzip util-linux wget build-essential pkg-config rsync

if [[ ! -r /etc/os-release ]]; then
  echo "cannot determine the Ubuntu release for the NVIDIA CUDA repository" >&2
  exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "ESMFold2-pipeline images require Ubuntu; found ${ID:-unknown}" >&2
  exit 1
fi
case "${VERSION_ID:-}" in
  22.04) CUDA_REPOSITORY_DISTRO="ubuntu2204" ;;
  24.04) CUDA_REPOSITORY_DISTRO="ubuntu2404" ;;
  *)
    echo "unsupported Ubuntu release for the NVIDIA CUDA repository: ${VERSION_ID:-unknown}" >&2
    exit 1
    ;;
esac

driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if [[ ! "${driver_major}" =~ ^[0-9]+$ || "${driver_major}" -lt "${MINIMUM_DRIVER_MAJOR}" ]]; then
  echo "CUDA ${CUDA_VERSION} image requires NVIDIA driver branch ${MINIMUM_DRIVER_MAJOR} or newer; found ${driver_major:-unknown}" >&2
  exit 1
fi

# An existing image can contain the opposite CUDA development profile.
# Remove its explicitly installed metapackages and toolkit directory before
# resolving the selected compiler so the captured image has one coherent build
# toolchain and does not retain stale libraries.
stale_cuda_packages=()
for package in \
  "cuda-nvcc-${STALE_CUDA_PACKAGE_SUFFIX}" \
  "cuda-cuobjdump-${STALE_CUDA_PACKAGE_SUFFIX}" \
  "cuda-cudart-dev-${STALE_CUDA_PACKAGE_SUFFIX}" \
  "cuda-libraries-dev-${STALE_CUDA_PACKAGE_SUFFIX}"
do
  if dpkg-query -W -f='${db:Status-Abbrev}' "${package}" 2>/dev/null | grep -q '^ii'; then
    stale_cuda_packages+=("${package}")
  fi
done
if [[ "${#stale_cuda_packages[@]}" -gt 0 ]]; then
  apt-get purge -y "${stale_cuda_packages[@]}"
  apt-get autoremove -y
fi
rm -rf "/usr/local/cuda-${STALE_CUDA_VERSION}"

find_cuda_toolkit_root() {
  local candidate
  for candidate in "/usr/local/cuda-${CUDA_VERSION}" /usr/local/cuda; do
    if [[ -x "${candidate}/bin/nvcc" ]] \
      && "${candidate}/bin/nvcc" --version 2>/dev/null | grep -Fq "release ${CUDA_VERSION}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

# Protenix JIT-compiles fused CUDA extensions, so selecting a compiler from a
# different CUDA major than the Torch environments silently creates an
# incompatible libcudart dependency. Install the selected official compiler
# and development toolkit when the base image does not already expose it.
if ! CUDA_TOOLKIT_ROOT="$(find_cuda_toolkit_root)"; then
  cuda_keyring=/tmp/cuda-keyring.deb
  curl --fail --silent --show-error --location --retry 3 \
    "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPOSITORY_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb" \
    --output "${cuda_keyring}"
  dpkg -i "${cuda_keyring}"
  rm -f "${cuda_keyring}"
  apt-get update
  apt-get install -y --no-install-recommends \
    "cuda-nvcc-${CUDA_PACKAGE_SUFFIX}" \
    "cuda-cuobjdump-${CUDA_PACKAGE_SUFFIX}" \
    "cuda-cudart-dev-${CUDA_PACKAGE_SUFFIX}" \
    "cuda-libraries-dev-${CUDA_PACKAGE_SUFFIX}"
  CUDA_TOOLKIT_ROOT="$(find_cuda_toolkit_root)"
fi

# A base image can expose nvcc without the CUDA development headers used by
# PyTorch JIT extensions. Protenix imports ATen's CUDAContextLight.h while
# compiling its fused layer norm, which in turn requires cusparse.h. Install the
# complete selected development-library set when that contract is incomplete.
if [[ ! -f "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/include/cusparse.h" ]]; then
  apt-get update
  apt-get install -y --no-install-recommends "cuda-libraries-dev-${CUDA_PACKAGE_SUFFIX}"
fi
if [[ ! -x "${CUDA_TOOLKIT_ROOT}/bin/cuobjdump" ]]; then
  apt-get update
  apt-get install -y --no-install-recommends "cuda-cuobjdump-${CUDA_PACKAGE_SUFFIX}"
fi
export CUDA_HOME="${CUDA_TOOLKIT_ROOT}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
nvcc --version | grep -Fq "release ${CUDA_VERSION}"
test -f "${CUDA_HOME}/targets/x86_64-linux/include/cusparse.h"
test ! -e "/usr/local/cuda-${STALE_CUDA_VERSION}"
rm -rf /var/lib/apt/lists/*

actual_disk_gb="$(df -BG / | awk 'NR==2 {gsub(/G/,"",$2); print $2}')"
if [[ -z "${actual_disk_gb}" ]]; then
  echo "cannot determine root disk size" >&2
  exit 1
fi
if [[ -n "${MINIMUM_ROOT_GB}" && "${actual_disk_gb}" -lt "${MINIMUM_ROOT_GB}" ]]; then
  echo "root filesystem must provide at least ${MINIMUM_ROOT_GB} GiB; found ${actual_disk_gb} GiB" >&2
  exit 1
fi

if [[ -d "${CHECKOUT}/.git" ]]; then
  git -C "${CHECKOUT}" remote set-url origin "${REPOSITORY}"
  git -C "${CHECKOUT}" fetch --prune origin
else
  rm -rf "${CHECKOUT}"
  git clone "${REPOSITORY}" "${CHECKOUT}"
  git -C "${CHECKOUT}" fetch --prune origin
fi
git -C "${CHECKOUT}" checkout --detach "${REF}"

cd "${CHECKOUT}"
# Recreate both environments so an existing image cannot retain a compiled
# extension or package from an older CUDA/Python contract.
rm -rf "${CHECKOUT}/.venv" "${PREFIX}/protenix-venv"
UV_VENV_CLEAR=1 \
ESMFOLD2_INSTALL_PREFIX="${PREFIX}" \
ESMFOLD2_TORCH_BACKEND="${TORCH_BACKEND}" \
./install.sh \
  --prefix "${PREFIX}" \
  --preload-models "${PRELOAD_MODELS_RAW}"

cat >> "${PREFIX}/env.sh" <<EOF
export CUDA_HOME=${CUDA_TOOLKIT_ROOT}
export PATH=${CUDA_TOOLKIT_ROOT}/bin:\$PATH
export LD_LIBRARY_PATH=${CUDA_TOOLKIT_ROOT}/lib64:${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}
EOF

ln -sfn "${CHECKOUT}/.venv/bin/esmfold2-pipeline" /usr/local/bin/esmfold2-pipeline
install -d -m 755 /etc/esmfold2-pipeline
cat > /etc/profile.d/esmfold2-pipeline.sh <<EOF
source ${PREFIX}/env.sh
export ESMFOLD2_PIPELINE_EXECUTABLE=${CHECKOUT}/.venv/bin/esmfold2-pipeline
EOF
chmod 644 /etc/profile.d/esmfold2-pipeline.sh

source "${PREFIX}/env.sh"
"${CHECKOUT}/.venv/bin/esmfold2-pipeline" check-env --esm-repo "${ESM_REPO}"
"${CHECKOUT}/.venv/bin/esmfold2-pipeline" check-protenix \
  --protenix-python "${PROTENIX_PYTHON}" \
  --protenix-checkpoint-dir "${PROTENIX_CHECKPOINT_DIR}"

pipeline_commit="$(git -C "${CHECKOUT}" rev-parse HEAD)"
esm_commit="$(git -C "${ESM_REPO}" rev-parse HEAD)"
checkpoint_sha="$(sha256sum "${PROTENIX_CHECKPOINT_DIR}/protenix-v2.pt" | awk '{print $1}')"
cuda_driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
cuda_runtime="$("${CHECKOUT}/.venv/bin/python" -c 'import torch; print(torch.version.cuda or "unknown")')"
protenix_cuda_runtime="$("${PROTENIX_PYTHON}" -c 'import torch; print(torch.version.cuda or "unknown")')"
cuequivariance_version="$("${CHECKOUT}/.venv/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("cuequivariance"))')"
protenix_cuequivariance_version="$("${PROTENIX_PYTHON}" -c 'import importlib.metadata; print(importlib.metadata.version("cuequivariance"))')"
test "${cuequivariance_version}" = "${protenix_cuequivariance_version}"
cuda_compiler="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | tail -1)"
created_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Record a sanitized SBOM. Package names and versions are retained, while
# editable paths and source URLs are deliberately omitted. The independently
# recorded Git revisions provide exact provenance for the three source trees.
pipeline_packages_tmp="$(mktemp /tmp/esmfold2-pipeline-packages.XXXXXX)"
protenix_packages_tmp="$(mktemp /tmp/esmfold2-protenix-packages.XXXXXX)"
os_packages_tmp="$(mktemp /tmp/esmfold2-os-packages.XXXXXX)"
sbom_tmp="$(mktemp /tmp/esmfold2-sbom.XXXXXX)"
model_manifest_tmp="$(mktemp /tmp/esmfold2-models.XXXXXX)"
trap 'rm -f "${pipeline_packages_tmp:-}" "${protenix_packages_tmp:-}" "${os_packages_tmp:-}" "${sbom_tmp:-}" "${model_manifest_tmp:-}"' EXIT

uv pip list --python "${CHECKOUT}/.venv/bin/python" --format json 2>/dev/null \
  | jq 'map({name,version}) | sort_by(.name)' > "${pipeline_packages_tmp}"
uv pip list --python "${PROTENIX_PYTHON}" --format json 2>/dev/null \
  | jq 'map({name,version}) | sort_by(.name)' > "${protenix_packages_tmp}"
dpkg-query -W -f='${binary:Package}=${Version}\n' | LC_ALL=C sort > "${os_packages_tmp}"

pipeline_packages_sha="$(sha256sum "${pipeline_packages_tmp}" | awk '{print $1}')"
protenix_packages_sha="$(sha256sum "${protenix_packages_tmp}" | awk '{print $1}')"
os_packages_sha="$(sha256sum "${os_packages_tmp}" | awk '{print $1}')"
protenix_revision="$("${PROTENIX_PYTHON}" - <<'PY'
import importlib.metadata
import json

distribution = importlib.metadata.distribution("protenix")
direct_url = distribution.read_text("direct_url.json")
payload = json.loads(direct_url) if direct_url else {}
revision = (payload.get("vcs_info") or {}).get("commit_id")
print(revision or f"package:{distribution.version}")
PY
)"

jq -n \
  --arg created_at "${created_at}" \
  --arg pipeline_commit "${pipeline_commit}" \
  --arg esm_commit "${esm_commit}" \
  --arg protenix_revision "${protenix_revision}" \
  --arg pipeline_packages_sha256 "${pipeline_packages_sha}" \
  --arg protenix_packages_sha256 "${protenix_packages_sha}" \
  --arg os_packages_sha256 "${os_packages_sha}" \
  --slurpfile pipeline_packages "${pipeline_packages_tmp}" \
  --slurpfile protenix_packages "${protenix_packages_tmp}" \
  '{schema_version:1,created_at:$created_at,sources:{pipeline_commit:$pipeline_commit,esm_commit:$esm_commit,protenix_revision:$protenix_revision},inventories:{pipeline:{sha256:$pipeline_packages_sha256,packages:$pipeline_packages[0]},protenix:{sha256:$protenix_packages_sha256,packages:$protenix_packages[0]},operating_system:{sha256:$os_packages_sha256}}}' \
  > "${sbom_tmp}"
install -m 644 "${sbom_tmp}" /etc/esmfold2-pipeline/esmfold2-pipeline-sbom.json
install -m 644 "${os_packages_tmp}" /etc/esmfold2-pipeline/esmfold2-os-packages.txt

# Hash the content-addressed Hugging Face blobs after all requested aliases have
# been preloaded. Relative paths make the manifest portable across images.
hf_cache_root="${HF_HOME:-${HOME}/.cache/huggingface}"
model_file_count=0
model_bytes=0
while IFS= read -r -d '' model_file; do
  model_relative="${model_file#"${hf_cache_root}/"}"
  model_sha="$(sha256sum "${model_file}" | awk '{print $1}')"
  printf '%s  %s\n' "${model_sha}" "${model_relative}" >> "${model_manifest_tmp}"
  model_file_count=$((model_file_count + 1))
  model_bytes=$((model_bytes + $(stat -c %s "${model_file}")))
done < <(find "${hf_cache_root}/hub" -type f -path '*/blobs/*' -print0 | LC_ALL=C sort -z)
if [[ "${model_file_count}" -eq 0 ]]; then
  echo "no preloaded Hugging Face model blobs found under ${hf_cache_root}/hub" >&2
  exit 1
fi
install -m 644 "${model_manifest_tmp}" /etc/esmfold2-pipeline/esmfold2-models.sha256
model_manifest_sha="$(sha256sum /etc/esmfold2-pipeline/esmfold2-models.sha256 | awk '{print $1}')"
sbom_sha="$(sha256sum /etc/esmfold2-pipeline/esmfold2-pipeline-sbom.json | awk '{print $1}')"
os_packages_manifest_sha="$(sha256sum /etc/esmfold2-pipeline/esmfold2-os-packages.txt | awk '{print $1}')"
uv_lock_sha="$(sha256sum "${CHECKOUT}/uv.lock" | awk '{print $1}')"
bootstrap_sha="$(sha256sum "${CHECKOUT}/deploy/image/bootstrap-image.sh" | awk '{print $1}')"
models_preloaded_json="$(jq -Rn --arg models "${PRELOAD_MODELS_RAW}" '
  $models
  | gsub(","; " ")
  | split(" ")
  | map(select(length > 0))
')"
declared_disk_gb_json="${DECLARED_DISK_GB:-null}"
minimum_root_gb_json="${MINIMUM_ROOT_GB:-null}"

jq -n \
  --arg created_at "${created_at}" \
  --arg image_revision "esmfold2-pipeline-${pipeline_commit:0:12}" \
  --arg pipeline_commit "${pipeline_commit}" \
  --arg esm_commit "${esm_commit}" \
  --arg protenix_revision "${protenix_revision}" \
  --arg protenix_checkpoint_sha256 "${checkpoint_sha}" \
  --arg nvidia_driver "${cuda_driver}" \
  --arg cuda_runtime "${cuda_runtime}" \
  --arg protenix_cuda_runtime "${protenix_cuda_runtime}" \
  --arg cuequivariance_version "${cuequivariance_version}" \
  --arg protenix_cuequivariance_version "${protenix_cuequivariance_version}" \
  --arg cuda_compiler "${cuda_compiler}" \
  --arg torch_backend "${TORCH_BACKEND}" \
  --arg accelerator_backend "${EXPECTED_ACCELERATOR_BACKEND}" \
  --arg sbom_sha256 "${sbom_sha}" \
  --arg os_packages_sha256 "${os_packages_manifest_sha}" \
  --arg model_manifest_sha256 "${model_manifest_sha}" \
  --arg uv_lock_sha256 "${uv_lock_sha}" \
  --arg bootstrap_sha256 "${bootstrap_sha}" \
  --arg model_cache_root "${hf_cache_root}" \
  --argjson model_file_count "${model_file_count}" \
  --argjson model_bytes "${model_bytes}" \
  --argjson disk_gb "${actual_disk_gb}" \
  --argjson declared_disk_gb "${declared_disk_gb_json}" \
  --argjson minimum_root_gb "${minimum_root_gb_json}" \
  --argjson models_preloaded "${models_preloaded_json}" \
  --argjson cuda_arch_targets "${CUDA_ARCH_TARGETS_JSON}" \
  '{schema_version:2,created_at:$created_at,image_revision:$image_revision,pipeline_revision:$pipeline_commit,pipeline_commit:$pipeline_commit,esm_commit:$esm_commit,protenix_revision:$protenix_revision,protenix_checkpoint_sha256:$protenix_checkpoint_sha256,nvidia_driver:$nvidia_driver,torch_backend:$torch_backend,accelerator_backend:$accelerator_backend,cuda_runtime:$cuda_runtime,protenix_cuda_runtime:$protenix_cuda_runtime,cuequivariance_version:$cuequivariance_version,protenix_cuequivariance_version:$protenix_cuequivariance_version,cuda_compiler:$cuda_compiler,cuda_arch_targets:$cuda_arch_targets,disk_gb:$disk_gb,declared_disk_gb:$declared_disk_gb,minimum_root_gb:$minimum_root_gb,models_preloaded:$models_preloaded,model_cache_root:$model_cache_root,model_file_count:$model_file_count,model_bytes:$model_bytes,model_manifest_sha256:$model_manifest_sha256,sbom_sha256:$sbom_sha256,os_packages_sha256:$os_packages_sha256,uv_lock_sha256:$uv_lock_sha256,bootstrap_sha256:$bootstrap_sha256,utilities:["cuobjdump","curl","flock","lsof","pigz","rclone","sqlite3","tmux","hmmscan"]}' \
  > /etc/esmfold2-pipeline/esmfold2-pipeline-image.json

for command in cuobjdump curl flock lsof rclone pigz sqlite3 tmux hmmscan; do command -v "${command}" >/dev/null; done
rm -rf /root/.cache/uv /tmp/torch_extensions*
sync
echo "ESMFold2-pipeline image bootstrap complete at ${pipeline_commit}"
