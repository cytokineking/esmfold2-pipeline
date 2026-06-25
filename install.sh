#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Install ESMFold2 Pipeline and its ESM runtime dependencies.

Usage:
  ./install.sh [options]

Options:
  --prefix DIR             Install root. Default: $HOME/esmfold2
  --esm-repo URL_OR_PATH   ESM git URL or local checkout. Default: https://github.com/Biohub/esm.git
  --esm-ref REF            Optional git ref to checkout for ESM.
  --python VERSION         Python version for uv. Default: 3.12
  --preload-models LIST    Comma/space separated model aliases to download.
                           Default: cutoff2025,fast-cutoff2025
  --skip-model-preload     Do not pre-download model checkpoints.
  --protenix-source SPEC   pip spec or local checkout for Protenix.
                           Default: cytokineking/Protenix template-capable fork.
  --protenix-env DIR       Separate Protenix virtualenv. Default: PREFIX/protenix-venv
  --protenix-checkpoint-dir DIR
                           Protenix checkpoint directory. Default: PREFIX/protenix-checkpoints
  --protenix-checkpoint-url URL
                           Protenix checkpoint URL. Default: Hugging Face mirror.
  --protenix-checkpoint-sha256 SHA256
                           Expected checkpoint SHA-256.
  --protenix-torch-specs LIST
                           Comma/space separated Torch package specs for the
                           Protenix venv. Default: torch>=2.11.0,
                           torchvision>=0.26.0,torchaudio>=2.11.0
  --skip-protenix-checkpoint-download
                           Export the checkpoint directory without downloading weights.
  --no-accelerators        Skip ESMFold2 accelerator packages
                           (xformers and cuEquivariance).
  --no-hmmer               Skip system HMMER install/check for VHH MSA validation.
  --no-protenix            Skip Protenix package install and Protenix checks.
  --no-check               Skip the final environment check.
  -h, --help               Show this help.

Environment overrides:
  ESMFOLD2_INSTALL_PREFIX
  ESMFOLD2_ESM_REPO
  ESMFOLD2_ESM_REF
  ESMFOLD2_PYTHON_VERSION
  ESMFOLD2_PRELOAD_MODELS
  ESMFOLD2_PROTENIX_SOURCE
  ESMFOLD2_PROTENIX_ENV
  ESMFOLD2_PROTENIX_CHECKPOINT_DIR
  ESMFOLD2_PROTENIX_CHECKPOINT_URL
  ESMFOLD2_PROTENIX_CHECKPOINT_SHA256
  ESMFOLD2_PROTENIX_TORCH_SPECS
  ESMFOLD2_DOWNLOAD_PROTENIX_CHECKPOINT=0
  ESMFOLD2_INSTALL_ACCELERATORS=0
  ESMFOLD2_ACCELERATOR_SPECS
  ESMFOLD2_INSTALL_HMMER=0
  ESMFOLD2_INSTALL_PROTENIX=0
EOF
}

log() {
  printf '[esmfold2-install] %s\n' "$*"
}

die() {
  printf '[esmfold2-install] error: %s\n' "$*" >&2
  exit 1
}

PREFIX="${ESMFOLD2_INSTALL_PREFIX:-$HOME/esmfold2}"
ESM_SOURCE="${ESMFOLD2_ESM_REPO:-https://github.com/Biohub/esm.git}"
ESM_REF="${ESMFOLD2_ESM_REF:-}"
PYTHON_VERSION="${ESMFOLD2_PYTHON_VERSION:-3.12}"
PRELOAD_MODELS_RAW="${ESMFOLD2_PRELOAD_MODELS:-cutoff2025,fast-cutoff2025}"
DEFAULT_PROTENIX_SOURCE="git+https://github.com/cytokineking/Protenix.git"
PROTENIX_SOURCE="${ESMFOLD2_PROTENIX_SOURCE:-$DEFAULT_PROTENIX_SOURCE}"
PROTENIX_ENV="${ESMFOLD2_PROTENIX_ENV:-}"
PROTENIX_CHECKPOINT_DIR="${ESMFOLD2_PROTENIX_CHECKPOINT_DIR:-${PROTENIX_CHECKPOINT_DIR:-}}"
DEFAULT_PROTENIX_CHECKPOINT_URL="https://huggingface.co/TMF001/pxdesign-weights/resolve/main/checkpoint/protenix-v2.pt"
DEFAULT_PROTENIX_CHECKPOINT_SHA256="8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599"
PROTENIX_CHECKPOINT_URL="${ESMFOLD2_PROTENIX_CHECKPOINT_URL:-$DEFAULT_PROTENIX_CHECKPOINT_URL}"
PROTENIX_CHECKPOINT_SHA256="${ESMFOLD2_PROTENIX_CHECKPOINT_SHA256:-$DEFAULT_PROTENIX_CHECKPOINT_SHA256}"
PROTENIX_TORCH_SPECS_RAW="${ESMFOLD2_PROTENIX_TORCH_SPECS:-torch>=2.11.0,torchvision>=0.26.0,torchaudio>=2.11.0}"
DOWNLOAD_PROTENIX_CHECKPOINT="${ESMFOLD2_DOWNLOAD_PROTENIX_CHECKPOINT:-1}"
ACCELERATOR_SPECS_RAW="${ESMFOLD2_ACCELERATOR_SPECS:-xformers,cuequivariance,cuequivariance-torch,cuequivariance-ops-torch-cu12}"
INSTALL_ACCELERATORS="${ESMFOLD2_INSTALL_ACCELERATORS:-1}"
INSTALL_HMMER="${ESMFOLD2_INSTALL_HMMER:-1}"
PRELOAD_MODELS=1
RUN_CHECK=1
INSTALL_PROTENIX="${ESMFOLD2_INSTALL_PROTENIX:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="${2:-}"
      shift 2
      ;;
    --esm-repo)
      ESM_SOURCE="${2:-}"
      shift 2
      ;;
    --esm-ref)
      ESM_REF="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="${2:-}"
      shift 2
      ;;
    --preload-models)
      PRELOAD_MODELS_RAW="${2:-}"
      shift 2
      ;;
    --skip-model-preload)
      PRELOAD_MODELS=0
      shift
      ;;
    --protenix-source)
      PROTENIX_SOURCE="${2:-}"
      shift 2
      ;;
    --protenix-env)
      PROTENIX_ENV="${2:-}"
      shift 2
      ;;
    --protenix-checkpoint-dir)
      PROTENIX_CHECKPOINT_DIR="${2:-}"
      shift 2
      ;;
    --protenix-checkpoint-url)
      PROTENIX_CHECKPOINT_URL="${2:-}"
      shift 2
      ;;
    --protenix-checkpoint-sha256)
      PROTENIX_CHECKPOINT_SHA256="${2:-}"
      shift 2
      ;;
    --protenix-torch-specs)
      PROTENIX_TORCH_SPECS_RAW="${2:-}"
      shift 2
      ;;
    --skip-protenix-checkpoint-download|--no-protenix-checkpoint-download)
      DOWNLOAD_PROTENIX_CHECKPOINT=0
      shift
      ;;
    --no-accelerators)
      INSTALL_ACCELERATORS=0
      shift
      ;;
    --no-hmmer)
      INSTALL_HMMER=0
      shift
      ;;
    --no-protenix)
      INSTALL_PROTENIX=0
      shift
      ;;
    --no-check)
      RUN_CHECK=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "$PREFIX" ]] || die "--prefix must not be empty"
[[ -n "$PYTHON_VERSION" ]] || die "--python must not be empty"
[[ -n "$ESM_SOURCE" ]] || die "--esm-repo must not be empty"
if [[ "$PRELOAD_MODELS" -eq 1 && -z "$PRELOAD_MODELS_RAW" ]]; then
  die "--preload-models must not be empty unless --skip-model-preload is set"
fi
if [[ "$INSTALL_PROTENIX" -eq 1 && -z "$PROTENIX_SOURCE" ]]; then
  die "--protenix-source must not be empty unless --no-protenix is set"
fi
if [[ "$INSTALL_PROTENIX" -eq 1 && -z "$PROTENIX_ENV" ]]; then
  PROTENIX_ENV="$PREFIX/protenix-venv"
fi
if [[ "$INSTALL_PROTENIX" -eq 1 && -z "$PROTENIX_CHECKPOINT_DIR" ]]; then
  PROTENIX_CHECKPOINT_DIR="$PREFIX/protenix-checkpoints"
fi
if [[ "$INSTALL_PROTENIX" -eq 1 && "$DOWNLOAD_PROTENIX_CHECKPOINT" -eq 1 && -z "$PROTENIX_CHECKPOINT_URL" ]]; then
  die "--protenix-checkpoint-url must not be empty unless --skip-protenix-checkpoint-download is set"
fi
if [[ "$INSTALL_PROTENIX" -eq 1 && -z "$PROTENIX_TORCH_SPECS_RAW" ]]; then
  die "--protenix-torch-specs must not be empty unless --no-protenix is set"
fi
if [[ "$INSTALL_ACCELERATORS" -eq 1 && -z "$ACCELERATOR_SPECS_RAW" ]]; then
  die "ESMFOLD2_ACCELERATOR_SPECS must not be empty unless --no-accelerators is set"
fi

need_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
    return
  fi
  die "missing sha256sum or shasum for checkpoint verification"
}

verify_sha256() {
  local file="$1"
  local expected="$2"
  [[ -n "$expected" ]] || return 0

  local actual
  actual="$(sha256_file "$file")"
  [[ "$actual" == "$expected" ]]
}

download_protenix_checkpoint() {
  local checkpoint_dir="$1"
  local checkpoint_url="$2"
  local expected_sha256="$3"
  local checkpoint_file="$checkpoint_dir/protenix-v2.pt"
  local tmp_file="$checkpoint_file.download"

  need_command curl
  mkdir -p "$checkpoint_dir"

  if [[ -f "$checkpoint_file" ]]; then
    if verify_sha256 "$checkpoint_file" "$expected_sha256"; then
      log "using existing Protenix checkpoint at $checkpoint_file"
      return
    fi
    log "existing Protenix checkpoint checksum mismatch; redownloading"
    rm -f "$tmp_file"
  fi

  log "downloading Protenix weights from $checkpoint_url"
  log "destination: $checkpoint_file"
  if ! curl -L --fail --retry 3 -C - -o "$tmp_file" "$checkpoint_url"; then
    log "resumable checkpoint download failed; retrying from scratch"
    rm -f "$tmp_file"
    curl -L --fail --retry 3 -o "$tmp_file" "$checkpoint_url"
  fi

  if ! verify_sha256 "$tmp_file" "$expected_sha256"; then
    rm -f "$tmp_file"
    die "downloaded Protenix checkpoint failed SHA-256 verification"
  fi

  mv "$tmp_file" "$checkpoint_file"
}

install_hmmer_if_needed() {
  if command -v hmmscan >/dev/null 2>&1; then
    log "using HMMER at $(command -v hmmscan)"
    return
  fi

  if [[ "$INSTALL_HMMER" -ne 1 ]]; then
    log "skipping HMMER install (--no-hmmer)"
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    log "installing HMMER with apt-get"
    if [[ "$EUID" -eq 0 ]]; then
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y hmmer
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y hmmer
    else
      die "HMMER is missing and apt-get requires root; install hmmer or rerun with --no-hmmer"
    fi
  elif command -v brew >/dev/null 2>&1; then
    log "installing HMMER with Homebrew"
    brew install hmmer
  else
    die "HMMER is missing; install hmmer so hmmscan is on PATH or rerun with --no-hmmer"
  fi

  command -v hmmscan >/dev/null 2>&1 || die "HMMER install completed but hmmscan is not on PATH"
}

install_uv_if_needed() {
  if command -v uv >/dev/null 2>&1; then
    log "using uv at $(command -v uv)"
    return
  fi
  need_command curl
  log "uv not found; installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv install completed but uv is not on PATH"
}

script_checkout_dir() {
  local source_path="${BASH_SOURCE[0]:-}"
  if [[ -z "$source_path" || "$source_path" == "bash" || "$source_path" == "sh" ]]; then
    return 1
  fi
  if [[ "$source_path" == /dev/fd/* || "$source_path" == /proc/*/fd/* ]]; then
    return 1
  fi

  local script_dir
  script_dir="$(cd "$(dirname "$source_path")" && pwd)"
  if [[ -f "$script_dir/pyproject.toml" ]] \
    && grep -q '^name = "esmfold2-pipeline"$' "$script_dir/pyproject.toml"; then
    printf '%s\n' "$script_dir"
    return 0
  fi
  return 1
}

require_pipeline_checkout_dir() {
  local checkout_dir
  if checkout_dir="$(script_checkout_dir)"; then
    printf '%s\n' "$checkout_dir"
    return 0
  fi
  die "install.sh must be run from an esmfold2-pipeline checkout"
}

checkout_git_repo() {
  local repo_url="$1"
  local dest="$2"
  local ref="$3"

  need_command git
  if [[ -d "$dest/.git" ]]; then
    log "updating $(basename "$dest")"
    git -C "$dest" fetch --prune
  else
    log "cloning $repo_url -> $dest"
    rm -rf "$dest"
    git clone "$repo_url" "$dest"
  fi

  if [[ -n "$ref" ]]; then
    git -C "$dest" checkout "$ref"
  fi
}

split_model_list() {
  local raw="${1//,/ }"
  local model
  for model in $raw; do
    printf '%s\n' "$model"
  done
}

mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"

install_uv_if_needed

PIPELINE_DIR="$(require_pipeline_checkout_dir)"
log "using pipeline checkout at $PIPELINE_DIR"

if [[ -d "$ESM_SOURCE" ]]; then
  ESM_DIR="$(cd "$ESM_SOURCE" && pwd)"
  log "using ESM checkout at $ESM_DIR"
else
  ESM_DIR="$PREFIX/esm"
  checkout_git_repo "$ESM_SOURCE" "$ESM_DIR" "$ESM_REF"
fi

cd "$PIPELINE_DIR"

log "installing Python $PYTHON_VERSION with uv if needed"
uv python install "$PYTHON_VERSION"

log "syncing pipeline environment"
uv sync --python "$PYTHON_VERSION"

log "installing ESM into the pipeline environment"
uv pip install -e "$ESM_DIR"

if [[ "$INSTALL_ACCELERATORS" -eq 1 ]]; then
  log "installing ESMFold2 accelerator packages: $ACCELERATOR_SPECS_RAW"
  mapfile -t ACCELERATOR_SPECS < <(split_model_list "$ACCELERATOR_SPECS_RAW")
  uv pip install --upgrade "${ACCELERATOR_SPECS[@]}"
else
  log "skipping ESMFold2 accelerator packages (--no-accelerators)"
fi

if [[ "$INSTALL_PROTENIX" -eq 1 ]]; then
  install_hmmer_if_needed

  log "installing VHH MSA numbering dependencies into the pipeline environment"
  uv pip install --upgrade abnumber anarcii
  log "checking VHH MSA numbering dependencies"
  if [[ "$INSTALL_HMMER" -eq 1 ]]; then
    uv run python - <<'PY'
import shutil

import abnumber  # noqa: F401
import anarcii  # noqa: F401

if shutil.which("hmmscan") is None:
    raise SystemExit("hmmscan is not on PATH")
PY
  else
    uv run python - <<'PY'
import abnumber  # noqa: F401
import anarcii  # noqa: F401
PY
  fi

  PROTENIX_ENV="$(mkdir -p "$(dirname "$PROTENIX_ENV")" && cd "$(dirname "$PROTENIX_ENV")" && pwd)/$(basename "$PROTENIX_ENV")"
  log "creating separate Protenix environment at $PROTENIX_ENV"
  uv venv --python "$PYTHON_VERSION" "$PROTENIX_ENV"
  PROTENIX_PYTHON="$PROTENIX_ENV/bin/python"
  [[ -x "$PROTENIX_PYTHON" ]] || die "Protenix environment did not create $PROTENIX_PYTHON"

  if [[ -d "$PROTENIX_SOURCE" ]]; then
    PROTENIX_DIR="$(cd "$PROTENIX_SOURCE" && pwd)"
    log "installing Protenix from local checkout $PROTENIX_DIR"
    uv pip install --python "$PROTENIX_PYTHON" -e "$PROTENIX_DIR"
  else
    log "installing Protenix package: $PROTENIX_SOURCE"
    if [[ "$PROTENIX_SOURCE" == "protenix" || "$PROTENIX_SOURCE" == protenix==* || "$PROTENIX_SOURCE" == protenix\>* || "$PROTENIX_SOURCE" == protenix\<* ]]; then
      uv pip install --python "$PROTENIX_PYTHON" --upgrade "$PROTENIX_SOURCE" --index-url https://pypi.org/simple
    else
      uv pip install --python "$PROTENIX_PYTHON" --upgrade "$PROTENIX_SOURCE"
    fi
  fi

  log "upgrading Protenix Torch stack: $PROTENIX_TORCH_SPECS_RAW"
  mapfile -t PROTENIX_TORCH_SPECS < <(split_model_list "$PROTENIX_TORCH_SPECS_RAW")
  uv pip install --python "$PROTENIX_PYTHON" --upgrade "${PROTENIX_TORCH_SPECS[@]}"

  log "installing Protenix runtime dependency: accelerate>=0.21.0"
  uv pip install --python "$PROTENIX_PYTHON" --upgrade 'accelerate>=0.21.0'
else
  log "skipping Protenix install (--no-protenix)"
fi

if [[ "$INSTALL_PROTENIX" -eq 1 ]]; then
  PROTENIX_CHECKPOINT_DIR="$(mkdir -p "$PROTENIX_CHECKPOINT_DIR" && cd "$PROTENIX_CHECKPOINT_DIR" && pwd)"
  if [[ "$DOWNLOAD_PROTENIX_CHECKPOINT" -eq 1 ]]; then
    download_protenix_checkpoint \
      "$PROTENIX_CHECKPOINT_DIR" \
      "$PROTENIX_CHECKPOINT_URL" \
      "$PROTENIX_CHECKPOINT_SHA256"
  else
    log "skipping Protenix checkpoint download; using $PROTENIX_CHECKPOINT_DIR"
  fi
fi

ENV_FILE="$PREFIX/env.sh"
cat > "$ENV_FILE" <<EOF
export ESM_REPO="$ESM_DIR"
export ESMFOLD2_PIPELINE_DIR="$PIPELINE_DIR"
export PATH="$HOME/.local/bin:\$PATH"
EOF
if [[ "$INSTALL_PROTENIX" -eq 1 ]]; then
  cat >> "$ENV_FILE" <<EOF
export PROTENIX_PYTHON="$PROTENIX_PYTHON"
EOF
fi
if [[ "$INSTALL_PROTENIX" -eq 1 && -n "$PROTENIX_CHECKPOINT_DIR" ]]; then
  cat >> "$ENV_FILE" <<EOF
export PROTENIX_CHECKPOINT_DIR="$PROTENIX_CHECKPOINT_DIR"
EOF
fi

if [[ "$RUN_CHECK" -eq 1 ]]; then
  check_args=(check-env --esm-repo "$ESM_DIR")
  log "running environment check"
  uv run esmfold2-pipeline "${check_args[@]}"
  if [[ "$INSTALL_PROTENIX" -eq 1 ]]; then
    protenix_check_args=(check-protenix --protenix-python "$PROTENIX_PYTHON")
    if [[ -n "$PROTENIX_CHECKPOINT_DIR" ]]; then
      protenix_check_args+=(--protenix-checkpoint-dir "$PROTENIX_CHECKPOINT_DIR")
    fi
    log "running Protenix environment check"
    uv run esmfold2-pipeline "${protenix_check_args[@]}"
  fi
fi

if [[ "$PRELOAD_MODELS" -eq 1 ]]; then
  log "pre-downloading model checkpoints: $PRELOAD_MODELS_RAW"
  log "this can take several minutes on a fresh machine"
  while IFS= read -r model_name; do
    [[ -n "$model_name" ]] || continue
    log "preloading model checkpoint for $model_name"
    uv run esmfold2-pipeline check-models \
      --esm-repo "$ESM_DIR" \
      --model "$model_name"
  done < <(split_model_list "$PRELOAD_MODELS_RAW")
fi

NEXT_PROTENIX_CHECK=""
if [[ "$INSTALL_PROTENIX" -eq 1 ]]; then
  NEXT_PROTENIX_CHECK='  uv run esmfold2-pipeline check-protenix'
fi

cat <<EOF

Install complete.

Pipeline: $PIPELINE_DIR
ESM:      $ESM_DIR
Env file: $ENV_FILE
Protenix: $([[ "$INSTALL_PROTENIX" -eq 1 ]] && printf '%s' "$PROTENIX_SOURCE" || printf 'skipped')
$([[ "$INSTALL_PROTENIX" -eq 1 ]] && printf 'Protenix Python: %s\n' "$PROTENIX_PYTHON" || true)
$([[ "$INSTALL_PROTENIX" -eq 1 ]] && printf 'Protenix checkpoint dir: %s\n' "$PROTENIX_CHECKPOINT_DIR" || true)

Next:
  cd "$PIPELINE_DIR"
  source "$ENV_FILE"
  uv run esmfold2-pipeline check-env --esm-repo "\$ESM_REPO"
$NEXT_PROTENIX_CHECK
EOF
