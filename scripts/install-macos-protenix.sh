#!/bin/bash
# Invoked by install-macos.sh after the design runtime has passed preflight.
set -euo pipefail
fail() { printf 'protenix-mac: %s\n' "$*" >&2; exit 1; }
[[ $# -eq 2 ]] || fail "use install-macos.sh --protenix-data DIRECTORY"
prefix=$1
data_root=$(cd "$2" && pwd -P) || fail "data root not found: $2"
source_dir=$(cd "$(dirname "$0")/.." && pwd -P)
base_ref=b806d2540ab326d71edcc47128343e627248c099
checkpoint="$data_root/checkpoint/protenix-v2.pt"
sha=8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599
[[ -f "$checkpoint" ]] || fail "expected $checkpoint"
actual=$(shasum -a 256 < "$checkpoint")
[[ ${actual%% *} == "$sha" ]] || fail "checkpoint SHA-256 mismatch; expected $sha"
# The template featurizer requires this directory, even for supplied templates.
mkdir -p "$data_root/mmcif"

patch_file="$source_dir/vendor/protenix/apple-v2.patch"
tree=$(cat "$source_dir/vendor/protenix/tree.txt")
checkout="$prefix/protenix-src-${tree:0:12}"
if [[ ! -e "$checkout" ]]; then
  staging=$(mktemp -d "$prefix/.protenix-source.XXXXXX")
  trap 'rm -rf "$staging"' EXIT
  git -C "$staging" init -q
  git -C "$staging" fetch -q --depth 1 https://github.com/cytokineking/Protenix.git "$base_ref"
  git -C "$staging" checkout -q --detach FETCH_HEAD
  git -C "$staging" apply --index "$patch_file"
  mv "$staging" "$checkout"
  trap - EXIT
fi
[[ $(git -C "$checkout" rev-parse HEAD) == "$base_ref" ]] && \
  [[ $(git -C "$checkout" write-tree) == "$tree" ]] && git -C "$checkout" diff --quiet -- \
  || fail "Protenix source differs from the qualified revision at $checkout"

venv="$prefix/protenix-venv"
if [[ ! -e "$venv" ]]; then uv venv --python "$prefix/venv/bin/python" "$venv"; fi
"$venv/bin/python" -c 'import platform, sys; sys.exit(not (sys.version_info[:2] == (3, 12) and platform.machine() == "arm64"))' \
  || fail "expected native Python 3.12 in the Protenix environment"
uv pip install --python "$venv/bin/python" --constraints "$source_dir/requirements/protenix-mps-py312.txt" \
  -e "$checkout" 'accelerate==1.15.0'
export PROTENIX_DEVICE=mps PROTENIX_PYTHON="$venv/bin/python"
export PROTENIX_ROOT_DIR="$data_root" PROTENIX_CHECKPOINT_DIR="$data_root/checkpoint"
"$prefix/venv/bin/esmfold2-pipeline" check-protenix
{
  printf 'export PROTENIX_DEVICE=mps\n'
  printf 'export PROTENIX_PYTHON=%q\n' "$PROTENIX_PYTHON"
  printf 'export PROTENIX_ROOT_DIR=%q\n' "$PROTENIX_ROOT_DIR"
  printf 'export PROTENIX_CHECKPOINT_DIR=%q\n' "$PROTENIX_CHECKPOINT_DIR"
} > "$prefix/protenix-env.sh"
