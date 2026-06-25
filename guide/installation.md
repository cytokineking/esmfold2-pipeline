# Installation & Requirements

[← Back to README](../README.md) · [Documentation index](README.md)

## Requirements

### Hardware

Use a CUDA-capable NVIDIA GPU. An 80 GB GPU is strongly recommended.

Practical choices:

- Recommended: H100 80 GB, H200, A100 80 GB, NVIDIA RTX PRO 6000 Blackwell, or
  B200.
- The RTX PRO 6000 Blackwell is a strong, cost-effective choice — it outperforms
  the A100 and H100 on this workload.
- scFv, VHH, and long target/binder complexes require more memory and run
  slower than short miniprotein campaigns.
- Multi-GPU execution uses one worker process per GPU. It does not split a
  single design across GPUs.

**Validated configurations.** All of the GPUs above have been tested with
**CUDA 12.8**. The Blackwell GPUs (RTX PRO 6000 Blackwell, B200) require CUDA
12.8 or later.

Keep enough local disk for Hugging Face model cache files, the project
environment, and campaign outputs. On a fresh H100 instance, the default
installer with `cutoff2025` and `fast-cutoff2025` preloaded used about 32 GB:
roughly 26 GB for Hugging Face checkpoints, 5.5 GB for the checkout plus
virtual environment, 0.5 GB for the uv cache, and 0.2 GB for uv itself. Plan
for at least 50 GB free for installation and initial checks, and more for real
campaign outputs.

### Software

- Linux GPU host recommended.
- Python 3.12 recommended because the current ESM package is pinned to Python
  3.12.
- NVIDIA driver with CUDA-capable GPU access. Validated with **CUDA 12.8**
  (required for Blackwell GPUs such as the RTX PRO 6000 Blackwell and B200).
- `git` for cloning the repository.

## Install

Clone the pipeline repo and run the installer:

```bash
git clone https://github.com/cytokineking/esmfold2-pipeline.git
cd esmfold2-pipeline
./install.sh
```

The installer installs `uv` if needed, installs Python 3.12, clones or updates
[Biohub ESM](https://github.com/Biohub/esm), syncs this pipeline environment,
installs ESM into that environment, installs the default ESMFold2 accelerator
packages (`xformers` and cuEquivariance), and pre-downloads the common ESMFold2
checkpoints. It installs into `$HOME/esmfold2` by default and writes
`$HOME/esmfold2/env.sh`.

The installer does not modify system Python and does not run system `pip` or
`python -m pip`. The only user-level Python tool it installs is `uv`; Python
package dependencies are managed by `uv` inside this project's environment.
When Protenix validation support is enabled, the installer also ensures HMMER is
available for VHH MSA preparation, using `apt-get` or Homebrew when `hmmscan` is
not already on `PATH`.

After installation, activate the environment and run a preflight:

```bash
source "$HOME/esmfold2/env.sh"
cd "$ESMFOLD2_PIPELINE_DIR"
uv run esmfold2-pipeline check-env --esm-repo "$ESM_REPO"

# Optional tutorial rollback preflight.
ESMFOLD2_PIPELINE_DESIGN_BACKEND=tutorial \
  uv run esmfold2-pipeline check-env --esm-repo "$ESM_REPO"
```

## Design backend

The default backend is the **local design loop**: model loading, forward calls,
and folding stay in ESM-owned runtime APIs, while the pipeline owns prompt
planning, loss composition, PLM regularization, and loop control. This is what
you want for normal use.

The older tutorial-backed path remains available as an explicit rollback via the
`ESMFOLD2_PIPELINE_DESIGN_BACKEND=tutorial` environment variable (shown in the
optional preflight above). Use it only if you need to reproduce the original
cookbook tutorial design loop.

## Common installer options

```bash
./install.sh --prefix /opt/esmfold2
./install.sh --esm-repo /path/to/existing/esm
./install.sh --esm-ref main
./install.sh --preload-models cutoff2025,fast-cutoff2025
./install.sh --skip-model-preload
./install.sh --protenix-checkpoint-dir /path/to/protenix-checkpoints
./install.sh --skip-protenix-checkpoint-download
./install.sh --no-accelerators
./install.sh --no-hmmer
./install.sh --no-protenix
```

scFv and VHH design campaigns do not require antibody annotation packages for
CDR definition. VHH validation with `validation.msa.use_msa: true` does require
VHH numbering during MSA prefetch, so the installer adds `abnumber` and
`anarcii` to the pipeline environment and checks that HMMER's `hmmscan` is on
`PATH`.

The default accelerator package set is `xformers`, `cuequivariance`,
`cuequivariance-torch`, and `cuequivariance-ops-torch-cu12`. Use
`ESMFOLD2_ACCELERATOR_SPECS` to override that list, or `--no-accelerators` when
installing on a system where those wheels are unavailable. `transformer-engine`
and `flash-attn` remain system-specific optional installs.

## First-run and preload behavior

By default, install preloads `cutoff2025` and `fast-cutoff2025` so the first
production run and first quick check do not pay the checkpoint download
cost. This makes installation slower on a fresh machine, but shifts the wait to
setup time. Use `--preload-models` to choose a different comma/space separated
set, or `--skip-model-preload` if you want installation to stop after the
environment check.

With the default preloads, expect the installer to use about 30–35 GB before
campaign outputs. Skipping model preload substantially reduces initial disk use,
but the same checkpoint cache will be populated later on the first model
preflight or launch.

The first model preload, preflight, or launch on a new machine is slower because
ESMC and ESMFold2 checkpoints are downloaded and deserialized before design
starts. The commands print progress messages for the major phases; during
first-time checkpoint downloads, GPU memory can remain low for several minutes.

The optional Protenix validation runtime has its own cold-start costs. The
installer downloads and verifies the Protenix checkpoint when validation support
is enabled. The first real Protenix validation run on a fresh OS image can also
spend several minutes loading that checkpoint and compiling CUDA extensions
before writing output structures. If a validation attempt times out during this
one-time warmup but the retry budget is not exhausted, `launch`/`validate`
retries the task and records the transient failed attempt in the campaign
attempt log. The command succeeds once the final validation task state is clean.

### Hugging Face Xet backend

On some hosts, Hugging Face's Xet transfer backend can stall on large checkpoint
downloads. Model preflight and run commands set `HF_HUB_DISABLE_XET=1` by
default. Pass `--enable-hf-xet` when you want to leave that backend enabled.

## Optional Protenix validation runtime

By default, the installer also prepares the optional Protenix validation runtime.
It creates a sibling Protenix virtual environment, installs the
template-capable [`cytokineking/Protenix`](https://github.com/cytokineking/Protenix)
fork, downloads `protenix-v2.pt` from the Hugging Face mirror into
`$HOME/esmfold2/protenix-checkpoints`, verifies the checkpoint SHA-256, and
exports `PROTENIX_PYTHON` and `PROTENIX_CHECKPOINT_DIR` in `env.sh`.

Use `--protenix-source` to point at another checkout or package,
`--protenix-checkpoint-dir` for a preseeded or shared weight directory, or
`--skip-protenix-checkpoint-download` when the directory will be populated
separately. Use `--no-protenix` when you only want the ESMFold2 design
pipeline.

Protenix validation supports miniprotein and VHH campaigns, plus built-in scFv
campaigns when bundled structural framework templates are used. See
[Validation](validation.md) for the validation lifecycle.
