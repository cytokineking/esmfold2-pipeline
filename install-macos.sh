#!/bin/bash
# Native arm64 PyTorch/MPS installer for the ESMFold2 design runtime.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./install-macos.sh [--prefix DIRECTORY] [--protenix-data DIRECTORY]

Install the single-worker Apple Silicon design runtime (Python 3.12) and
check Metal access. Requires git and uv. Default: $HOME/esmfold2-mac.
ESMFold2 weights are downloaded on first use. Reuse a cache by setting HF_HOME.
To add Protenix V2, provide its data root containing checkpoint/protenix-v2.pt.
Common chemical data are read from common/ in that root or fetched on first use.
USAGE
}
fail() { printf 'esmfold2-mac: %s\n' "$*" >&2; exit 1; }

prefix="${ESMFOLD2_INSTALL_PREFIX:-$HOME/esmfold2-mac}"
assets_dir=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) [[ -n "${2:-}" ]] || fail "--prefix needs a directory"; prefix="$2"; shift 2 ;;
    --protenix-data) [[ -n "${2:-}" ]] || fail "--protenix-data needs a directory"; assets_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; fail "unknown argument: $1" ;;
  esac
done
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || fail "use a native arm64 shell on Apple Silicon"
command -v git >/dev/null || fail "git is required"
command -v uv >/dev/null || fail "install uv, then rerun (e.g. brew install uv)"

source_dir=$(cd "$(dirname "$0")" && pwd -P)
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd -P)
esm_dir="$prefix/esm"
esm_ref=ba4d7124864eed323a93bf3cfefcd958f573b75a
if [[ ! -e "$esm_dir" ]]; then
  staging=$(mktemp -d "$prefix/.esm-source.XXXXXX")
  trap 'rm -rf "$staging"' EXIT
  git -C "$staging" init -q
  git -C "$staging" fetch -q --depth 1 https://github.com/Biohub/esm.git "$esm_ref"
  git -C "$staging" checkout -q --detach FETCH_HEAD
  mv "$staging" "$esm_dir"
  trap - EXIT
fi
[[ $(git -C "$esm_dir" rev-parse HEAD) == "$esm_ref" ]] && git -C "$esm_dir" diff --quiet HEAD -- \
  || fail "expected an unmodified ESM checkout at $esm_ref; choose a separate --prefix"
venv="$prefix/venv"
if [[ ! -e "$venv" ]]; then uv venv --python 3.12 "$venv"; fi
[[ -x "$venv/bin/python" ]] || fail "existing venv is incomplete; choose a separate --prefix"
"$venv/bin/python" -c 'import platform, sys; sys.exit(not (sys.version_info[:2] == (3, 12) and platform.machine() == "arm64"))' \
  || fail "expected a native Python 3.12 environment; choose a separate --prefix"

uv pip install --python "$venv/bin/python" \
  --constraints "$source_dir/requirements/apple-mps-py312.txt" \
  --overrides "$source_dir/requirements/apple-transformers.txt" \
  "$source_dir" -e "$esm_dir" psutil

{
  printf 'export ESMFOLD2_PIPELINE_DEVICE=mps\nexport ESMFOLD2_PIPELINE_DESIGN_BACKEND=local\n'
  printf 'export ESM_REPO=%q\n' "$esm_dir"
  printf 'export PATH=%q:"$PATH"\n' "$venv/bin"
  printf 'if [ -z "${HF_HOME:-}" ]; then export HF_HOME=%q; fi\n' "$prefix/huggingface"
} > "$prefix/env.sh"
source "$prefix/env.sh"
"$venv/bin/esmfold2-pipeline" check-env --esm-repo "$ESM_REPO" --local-runtime
if [[ -n "$assets_dir" ]]; then
  "$source_dir/scripts/install-macos-protenix.sh" "$prefix" "$assets_dir"
fi
if [[ -f "$prefix/protenix-env.sh" ]]; then
  printf 'source %q\n' "$prefix/protenix-env.sh" >> "$prefix/env.sh"
fi
printf 'To use this environment: source %q\n' "$prefix/env.sh"
