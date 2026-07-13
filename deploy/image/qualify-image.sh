#!/usr/bin/env bash
set -Eeuo pipefail

CHECKOUT="${ESMFOLD2_PIPELINE_CHECKOUT:-/opt/esmfold2-pipeline}"
PREFIX="${ESMFOLD2_INSTALL_PREFIX:-/opt/esmfold2}"
OUTPUT_ROOT="${ESMFOLD2_QUALIFICATION_OUTPUT_ROOT:-/root/esmfold2-image-qualification}"
REMOTE_BASE="${ESMFOLD2_QUALIFICATION_REMOTE:-}"
MINIMUM_GPU_MEMORY_MIB="${ESMFOLD2_QUALIFICATION_MIN_GPU_MEMORY_MIB:-}"
RUN_MODALITY_SMOKES=0
SMOKE_TARGET_SHA256=""
SMOKE_FIXTURE_MANIFEST_SHA256=""
PROTENIX_EXTENSION_SHA256=""
PROTENIX_EXTENSION_CUDART=""
REMOTE_TEST_PATH=""

if [[ -n "${MINIMUM_GPU_MEMORY_MIB}" && ! "${MINIMUM_GPU_MEMORY_MIB}" =~ ^[0-9]+$ ]]; then
  echo "ESMFOLD2_QUALIFICATION_MIN_GPU_MEMORY_MIB must be an integer when set" >&2
  exit 2
fi

if [[ "${1:-}" == "--full" ]]; then RUN_MODALITY_SMOKES=1; shift; fi
if [[ $# -ne 0 ]]; then echo "usage: qualify-image.sh [--full]" >&2; exit 2; fi

source "${PREFIX}/env.sh"
test -x "${CUDA_HOME}/bin/nvcc"
test -f "${CUDA_HOME}/targets/x86_64-linux/include/cusparse.h"
executable="${CHECKOUT}/.venv/bin/esmfold2-pipeline"
test -x "${executable}"
image_manifest=/etc/esmfold2-pipeline/esmfold2-pipeline-image.json
sbom=/etc/esmfold2-pipeline/esmfold2-pipeline-sbom.json
os_packages=/etc/esmfold2-pipeline/esmfold2-os-packages.txt
model_manifest=/etc/esmfold2-pipeline/esmfold2-models.sha256
for evidence_file in "${image_manifest}" "${sbom}" "${os_packages}" "${model_manifest}"; do
  test -s "${evidence_file}"
done
expected_cuda_compiler="$(jq -r '.cuda_compiler' "${image_manifest}")"
actual_cuda_compiler="$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | tail -1)"
test "${actual_cuda_compiler}" = "${expected_cuda_compiler}"
case "${actual_cuda_compiler}" in
  12.8) test ! -e /usr/local/cuda-13.0 ;;
  13.0) test ! -e /usr/local/cuda-12.8 ;;
  *) echo "unsupported qualified CUDA compiler: ${actual_cuda_compiler}" >&2; exit 1 ;;
esac
jq -e '
  .schema_version == 2 and
  (.models_preloaded | type == "array" and length > 0) and
  (.models_preloaded | all(
    . == "fast" or
    . == "fast-cutoff2025" or
    . == "cutoff2025" or
    . == "experimental"
  )) and
  (.declared_disk_gb == null or (.declared_disk_gb | type == "number")) and
  (.minimum_root_gb == null or (.minimum_root_gb | type == "number")) and
  (.torch_backend == "cu128" or .torch_backend == "cu130") and
  (.accelerator_backend == "cu12" or .accelerator_backend == "cu13") and
  (.cuda_runtime | type == "string" and length > 0) and
  (.protenix_cuda_runtime | type == "string" and length > 0) and
  (.cuequivariance_version | type == "string" and length > 0) and
  .cuequivariance_version == .protenix_cuequivariance_version and
  .cuda_runtime == .protenix_cuda_runtime and
  (.cuda_runtime | split(".")[0]) == (.cuda_compiler | split(".")[0]) and
  (.torch_backend | sub("^cu"; "") | .[0:2]) == (.cuda_runtime | split(".")[0]) and
  (if .torch_backend == "cu128" then .accelerator_backend == "cu12" else .accelerator_backend == "cu13" end) and
  .model_file_count > 0 and
  .model_bytes > 0
' "${image_manifest}" >/dev/null
test "$("${CHECKOUT}/.venv/bin/python" -c 'import torch; print(torch.version.cuda or "unknown")')" = "$(jq -r '.cuda_runtime' "${image_manifest}")"
test "$("${PROTENIX_PYTHON}" -c 'import torch; print(torch.version.cuda or "unknown")')" = "$(jq -r '.protenix_cuda_runtime' "${image_manifest}")"
accelerator_backend="$(jq -r '.accelerator_backend' "${image_manifest}")"
cuequivariance_version="$(jq -r '.cuequivariance_version' "${image_manifest}")"
for accelerator_python in "${CHECKOUT}/.venv/bin/python" "${PROTENIX_PYTHON}"; do
"${accelerator_python}" - "${accelerator_backend}" "${cuequivariance_version}" <<'PY'
import importlib.metadata
import sys

expected = f"cuequivariance-ops-torch-{sys.argv[1]}"
opposite = "cuequivariance-ops-torch-cu13" if sys.argv[1] == "cu12" else "cuequivariance-ops-torch-cu12"
expected_version = sys.argv[2]
for package in ("cuequivariance", "cuequivariance-torch", expected):
    actual_version = importlib.metadata.version(package)
    if actual_version != expected_version:
        raise SystemExit(
            f"{package} has version {actual_version}; expected {expected_version}"
        )
try:
    importlib.metadata.version(opposite)
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit(f"mixed cuEquivariance accelerator packages: {expected} and {opposite}")
PY
done
test "$(sha256sum "${sbom}" | awk '{print $1}')" = "$(jq -r '.sbom_sha256' "${image_manifest}")"
test "$(sha256sum "${os_packages}" | awk '{print $1}')" = "$(jq -r '.os_packages_sha256' "${image_manifest}")"
test "$(sha256sum "${model_manifest}" | awk '{print $1}')" = "$(jq -r '.model_manifest_sha256' "${image_manifest}")"
test "$(sha256sum "${CHECKOUT}/uv.lock" | awk '{print $1}')" = "$(jq -r '.uv_lock_sha256' "${image_manifest}")"
test "$(sha256sum "${CHECKOUT}/deploy/image/bootstrap-image.sh" | awk '{print $1}')" = "$(jq -r '.bootstrap_sha256' "${image_manifest}")"
test "$(sha256sum "${PROTENIX_CHECKPOINT_DIR}/protenix-v2.pt" | awk '{print $1}')" = "$(jq -r '.protenix_checkpoint_sha256' "${image_manifest}")"

evidence_tmp="$(mktemp -d /tmp/esmfold2-qualification-evidence.XXXXXX)"
cleanup() {
  rm -rf "${evidence_tmp:-}"
  if [[ -n "${REMOTE_TEST_PATH:-}" ]]; then
    rclone purge "${REMOTE_TEST_PATH}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
uv pip list --python "${CHECKOUT}/.venv/bin/python" --format json 2>/dev/null \
  | jq 'map({name,version}) | sort_by(.name)' > "${evidence_tmp}/pipeline-packages.json"
uv pip list --python "${PROTENIX_PYTHON}" --format json 2>/dev/null \
  | jq 'map({name,version}) | sort_by(.name)' > "${evidence_tmp}/protenix-packages.json"
jq '.inventories.pipeline.packages' "${sbom}" > "${evidence_tmp}/expected-pipeline-packages.json"
jq '.inventories.protenix.packages' "${sbom}" > "${evidence_tmp}/expected-protenix-packages.json"
cmp "${evidence_tmp}/pipeline-packages.json" "${evidence_tmp}/expected-pipeline-packages.json"
cmp "${evidence_tmp}/protenix-packages.json" "${evidence_tmp}/expected-protenix-packages.json"
dpkg-query -W -f='${binary:Package}=${Version}\n' | LC_ALL=C sort > "${evidence_tmp}/os-packages.txt"
cmp "${os_packages}" "${evidence_tmp}/os-packages.txt"

model_cache_root="$(jq -r '.model_cache_root' "${image_manifest}")"
[[ "${model_cache_root}" == /* ]]
! cut -d ' ' -f 3- "${model_manifest}" | grep -qE '(^/|(^|/)\.\.(/|$))'
(
  cd "${model_cache_root}"
  sha256sum --quiet -c "${model_manifest}"
)
test "$(wc -l < "${model_manifest}" | tr -d ' ')" = "$(jq -r '.model_file_count' "${image_manifest}")"
verified_model_bytes=0
while IFS= read -r model_record; do
  model_relative="${model_record#*  }"
  verified_model_bytes=$((verified_model_bytes + $(stat -c %s "${model_cache_root}/${model_relative}")))
done < "${model_manifest}"
test "${verified_model_bytes}" = "$(jq -r '.model_bytes' "${image_manifest}")"
"${PROTENIX_PYTHON}" - <<'PY'
from pathlib import Path
import sysconfig

path = (
    Path(sysconfig.get_paths()["purelib"])
    / "protenix"
    / "model"
    / "layer_norm"
    / "torch_ext_compile.py"
)
if '("120", "120")' not in path.read_text():
    raise SystemExit(f"Protenix extension build omits sm_120: {path}")
PY
for command in cuobjdump curl flock ldd lsof nvidia-smi rclone pigz sqlite3 tmux hmmscan; do command -v "${command}" >/dev/null; done
disk_gb="$(df -BG / | awk 'NR==2 {gsub(/G/,"",$2); print $2}')"
[[ -n "${disk_gb}" ]]
minimum_root_gb="$(jq -r '.minimum_root_gb // empty' "${image_manifest}")"
if [[ -n "${minimum_root_gb}" ]]; then
  [[ "${disk_gb}" -ge "${minimum_root_gb}" ]] || {
    echo "root filesystem has ${disk_gb} GiB; image requires ${minimum_root_gb} GiB" >&2
    exit 1
  }
fi
if [[ -n "${MINIMUM_GPU_MEMORY_MIB}" ]]; then
  while read -r memory_mib; do
    memory_mib="${memory_mib// /}"
    [[ "${memory_mib}" -ge "${MINIMUM_GPU_MEMORY_MIB}" ]] || {
      echo "GPU has ${memory_mib} MiB; qualification requires ${MINIMUM_GPU_MEMORY_MIB} MiB" >&2
      exit 1
    }
  done < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
fi

mkdir -p "${OUTPUT_ROOT}"
rm -rf "${OUTPUT_ROOT}/io-contract"
mkdir -p "${OUTPUT_ROOT}/io-contract/visible" "${OUTPUT_ROOT}/io-contract/.scratch" "${OUTPUT_ROOT}/io-contract/visible/.hidden"
printf 'durable\n' > "${OUTPUT_ROOT}/io-contract/visible/result.txt"
printf 'never-transfer\n' > "${OUTPUT_ROOT}/io-contract/.scratch/secret.txt"
printf 'never-transfer\n' > "${OUTPUT_ROOT}/io-contract/visible/.hidden/secret.txt"
python3 - "${OUTPUT_ROOT}/io-contract" <<'PY'
import pathlib,sqlite3,sys
root=pathlib.Path(sys.argv[1]); live=root/'campaign.sqlite'; backup=root/'campaign.backup.sqlite'
db=sqlite3.connect(live); db.execute('create table qualification(value text)'); db.execute('insert into qualification values (?)',('ok',)); db.commit()
dst=sqlite3.connect(backup); db.backup(dst); dst.close(); db.close()
check=sqlite3.connect(f'file:{backup}?mode=ro',uri=True).execute('pragma integrity_check').fetchone()[0]
if check != 'ok': raise RuntimeError(f'SQLite backup integrity check failed: {check}')
PY
(
  cd "${OUTPUT_ROOT}/io-contract"
  find visible -type f ! -path '*/.*' -print0 > archive.members
  tar --null -T archive.members -cf - | pigz > outputs.tar.gz.partial
  pigz -t outputs.tar.gz.partial
  tar -tzf outputs.tar.gz.partial > archive.list
  ! grep -qE '(^|/)\.' archive.list
  mv outputs.tar.gz.partial outputs.tar.gz
)
if [[ -n "${REMOTE_BASE}" ]]; then
  REMOTE_TEST_PATH="${REMOTE_BASE%/}/esmfold2-image-qualification-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  rclone copy "${OUTPUT_ROOT}/io-contract" "${REMOTE_TEST_PATH}" \
    --filter '+ /visible/result.txt' --filter '+ /campaign.backup.sqlite' --filter '- **' \
    --transfers 4 --checkers 8
  rclone lsf "${REMOTE_TEST_PATH}" --recursive > "${OUTPUT_ROOT}/io-contract/remote.list"
  grep -qx 'visible/result.txt' "${OUTPUT_ROOT}/io-contract/remote.list"
  grep -qx 'campaign.backup.sqlite' "${OUTPUT_ROOT}/io-contract/remote.list"
  ! grep -qE '(^|/)\.' "${OUTPUT_ROOT}/io-contract/remote.list"
  rclone purge "${REMOTE_TEST_PATH}"
  REMOTE_TEST_PATH=""
fi

rm -rf "${OUTPUT_ROOT}/gpu-smoke"
"${executable}" check-env --esm-repo "${ESM_REPO}"
"${executable}" check-protenix --protenix-python "${PROTENIX_PYTHON}" --protenix-checkpoint-dir "${PROTENIX_CHECKPOINT_DIR}"
while IFS= read -r model; do
  "${executable}" check-models --esm-repo "${ESM_REPO}" --model "${model}"
done < <(jq -r '.models_preloaded[]' "${image_manifest}")
"${executable}" plan-gpu-smoke "${OUTPUT_ROOT}/gpu-smoke"
"${executable}" run-gpu-smoke "${OUTPUT_ROOT}/gpu-smoke" --gpu-id 0 --steps 2
"${executable}" status "${OUTPUT_ROOT}/gpu-smoke" --json | jq -e '.schema_version == 1 and .counts.shards.completed == 1' >/dev/null

if [[ "${RUN_MODALITY_SMOKES}" -eq 1 ]]; then
  fixture_dir="${OUTPUT_ROOT}/fixtures"
  rm -rf "${fixture_dir}"
  mkdir -p "${fixture_dir}"
  smoke_target="${fixture_dir}/2b5i.cif"
  curl --fail --silent --show-error --location --retry 3 \
    https://files.rcsb.org/download/2B5I.cif --output "${smoke_target}.partial"
  test -s "${smoke_target}.partial"
  mv "${smoke_target}.partial" "${smoke_target}"
  SMOKE_TARGET_SHA256="$(sha256sum "${smoke_target}" | awk '{print $1}')"

  write_smoke_config() {
    local path="$1" scaffold="$2" binder_value="$3" seed="$4"
    {
      cat <<EOF
target:
  name: 2b5i_chain_b_tyr134_image_smoke
  structure: ${smoke_target}
  chains: [B]
  structure_indexing: auth_seq_id
  hotspots: "B:134"
  conditioning:
    mode: distogram

binder:
  scaffold: ${scaffold}
EOF
      if [[ "${scaffold}" == "miniprotein" ]]; then
        printf '  length: %s\n' "${binder_value}"
      else
        printf '  framework: %s\n' "${binder_value}"
      fi
      cat <<EOF

campaign:
  num_designs: 1
  seed_start: ${seed}
  inversion_model: fast
  critics: [fast]
  steps: 1

validation:
  model: protenix-v2
  top_k: all
  require_hotspot_contact: never
  protenix:
    use_template: true
    seeds: [101]
    n_sample: 1
    n_step: 1
    n_cycle: 1
    keep_validation_debug: true
    timeout_seconds: 1800
  msa:
    use_msa: false
    target: none
    binder: auto

output: ${OUTPUT_ROOT}/unused-config-output
EOF
    } > "${path}"
  }

  write_smoke_config "${fixture_dir}/miniprotein.yaml" miniprotein 60-80 1301
  write_smoke_config "${fixture_dir}/vhh.yaml" vhh caplacizumab 1201
  write_smoke_config "${fixture_dir}/scfv.yaml" scfv trastuzumab_framework_vhvl 1401
  (
    cd "${fixture_dir}"
    sha256sum miniprotein.yaml vhh.yaml scfv.yaml | LC_ALL=C sort \
      > smoke-fixtures.sha256
  )
  SMOKE_FIXTURE_MANIFEST_SHA256="$(sha256sum "${fixture_dir}/smoke-fixtures.sha256" | awk '{print $1}')"

  for config in \
    "${fixture_dir}/miniprotein.yaml" \
    "${fixture_dir}/vhh.yaml" \
    "${fixture_dir}/scfv.yaml"
  do
    name="$(basename "${config}" .yaml)"
    rm -rf "${OUTPUT_ROOT}/${name}"
    "${executable}" launch "${config}" --out "${OUTPUT_ROOT}/${name}" --gpus all --max-designs 1 --min-iptm 0 --analysis-top-k 1
    "${executable}" status "${OUTPUT_ROOT}/${name}" --json | jq -e '
      .campaign.successful == true and
      .campaign.validation_configured == true and
      .campaign.terminal_failure_count == 0 and
      .expected_artifacts.validation_results == true and
      .expected_artifacts.combined_ranking == true
    ' >/dev/null
  done

  protenix_site_packages="$("${PROTENIX_PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  protenix_extension="$(
    for extension_root in \
      "${protenix_site_packages}/protenix/model/layer_norm" \
      /root/.cache/torch_extensions
    do
      [[ -d "${extension_root}" ]] || continue
      find "${extension_root}" -type f -name '*.so' \
        -path '*fast_layer_norm_cuda_v2*' -print -quit
    done | head -1
  )"
  test -n "${protenix_extension}"
  cuda_major="$(jq -r '.cuda_compiler' "${image_manifest}" | cut -d. -f1)"
  PROTENIX_EXTENSION_CUDART="libcudart.so.${cuda_major}"
  ldd "${protenix_extension}" > "${evidence_tmp}/protenix-extension.ldd"
  grep -q "${PROTENIX_EXTENSION_CUDART}" "${evidence_tmp}/protenix-extension.ldd"
  cuobjdump -lelf "${protenix_extension}" > "${evidence_tmp}/protenix-extension.elf"
  grep -q 'sm_100' "${evidence_tmp}/protenix-extension.elf"
  grep -q 'sm_120' "${evidence_tmp}/protenix-extension.elf"
  PROTENIX_EXTENSION_SHA256="$(sha256sum "${protenix_extension}" | awk '{print $1}')"
fi

jq -n \
  --arg qualified_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg hostname "$(hostname)" \
  --arg gpu "$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)" \
  --arg driver "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
  --arg cuda_compiler "$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | tail -1)" \
  --arg pipeline_commit "$(git -C "${CHECKOUT}" rev-parse HEAD)" \
  --arg smoke_target_sha256 "${SMOKE_TARGET_SHA256}" \
  --arg smoke_fixture_manifest_sha256 "${SMOKE_FIXTURE_MANIFEST_SHA256}" \
  --arg sbom_sha256 "$(sha256sum "${sbom}" | awk '{print $1}')" \
  --arg model_manifest_sha256 "$(sha256sum "${model_manifest}" | awk '{print $1}')" \
  --arg protenix_extension_sha256 "${PROTENIX_EXTENSION_SHA256}" \
  --arg protenix_extension_cudart "${PROTENIX_EXTENSION_CUDART}" \
  --argjson disk_gb "${disk_gb}" \
  --argjson remote_qualified "$([[ -n "${REMOTE_BASE}" ]] && echo true || echo false)" \
  --argjson full "${RUN_MODALITY_SMOKES}" \
  '{schema_version:2,qualified_at:$qualified_at,hostname:$hostname,gpu:$gpu,driver:$driver,cuda_compiler:$cuda_compiler,pipeline_commit:$pipeline_commit,disk_gb:$disk_gb,package_inventory_verified:true,model_inventory_verified:true,sbom_sha256:$sbom_sha256,model_manifest_sha256:$model_manifest_sha256,sqlite_backup:true,hidden_path_exclusion:true,pigz_archives:true,remote_round_trip:$remote_qualified,full_modality_validation:($full == 1),smoke_target_sha256:(if $smoke_target_sha256 == "" then null else $smoke_target_sha256 end),smoke_fixture_manifest_sha256:(if $smoke_fixture_manifest_sha256 == "" then null else $smoke_fixture_manifest_sha256 end),protenix_extension_sha256:(if $protenix_extension_sha256 == "" then null else $protenix_extension_sha256 end),protenix_extension_cudart:(if $protenix_extension_cudart == "" then null else $protenix_extension_cudart end),passed:true}' \
  > "${OUTPUT_ROOT}/qualification.json"
install -m 0644 "${OUTPUT_ROOT}/qualification.json" \
  /etc/esmfold2-pipeline/esmfold2-pipeline-qualification.json
cat "${OUTPUT_ROOT}/qualification.json"
