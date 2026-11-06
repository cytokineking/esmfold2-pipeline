# Apple Silicon design preview

This preview runs one ESMFold2 design worker and its final ESMFold2 critic on
the Apple GPU through PyTorch/MPS. It has been exercised on a 128 GB M5 Max
running macOS 27.0.1. The example is a 115-residue PD-L1 target plus a
125-residue VHH. The selection threshold is unchanged. A complete design is
a software test; it does not establish binding activity or performance on
other targets, Mac models, or memory sizes.

## Install and run

Use a native arm64 terminal with Metal access, git and
[uv](https://docs.astral.sh/uv/). Extract the source distribution and run the
following from its top-level directory:

```bash
./install-macos.sh
source "$HOME/esmfold2-mac/env.sh"
esmfold2-pipeline check example_configs/apple/pdl1-vhh.yaml
esmfold2-pipeline launch example_configs/apple/pdl1-vhh.yaml --esm-repo "$ESM_REPO"
```

The installer creates a separate Python 3.12 environment, installs the pinned
Mac dependencies, checks the ESM APIs and Metal visibility, and writes
`env.sh`. Use `--prefix DIRECTORY` to choose a different install location.
Run subsequent commands from the source directory for access to the examples
and measurement scripts. The wheel contains the Python package; the installer
and example inputs are in the source archive.

On first use, allow around 25 GiB for the tested Hugging Face model cache,
plus space for Python packages, the download cache and outputs. Set `HF_HOME`
before sourcing the environment to reuse an existing cache. The default is
`huggingface/` under the install prefix. Supply an authorized Hugging Face
token if the upstream checkpoint requires one. Weights are downloaded from
their providers under their own terms and are not bundled.

Use one local worker, with no `--gpus` or `--gpu-id`. The example does 150
optimization steps with seed 42 and writes to `gpu_runs/apple-pdl1-vhh`.
Give `launch` a new `--out` directory for a fresh run. The design runtime
uses unquantized FP32 ESMC-6B weights shared by the folding model and the
masked language-model loss. Some upstream attention operations cast to BF16.
Integer scatter bookkeeping runs on the CPU. Other operations do not require
enabling PyTorch's blanket CPU fallback. For a larger target, try e.g.
`export ESMFOLD2_MPS_CHUNK_SIZE=64` to split trunk and confidence work. The
measured default is 0 (unchunked). No larger workload has been qualified here.

## Reproduce the measurements

The [example](../example_configs/apple/pdl1-vhh.yaml) includes the archived
[PDB 4ZQK](https://www.rcsb.org/structure/4ZQK) mmCIF revision 1.4, SHA-256
`9f4f13db7623957ff62ef294ea337ee7ac9310f7b4cb5bf8eec77cf9df587470`.
The pinned ESM and Transformers source revisions are in the installer and
[override](../requirements/apple-transformers.txt). The benchmark used Torch
2.11.0, ESMC-6B snapshot `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a`,
and ESMFold2-Experimental-Fast-Cutoff2025 snapshot
`13f892d3bc9de27aa475e8a12ec76dba4054f4a2`.

```bash
python scripts/validate_mps_gradients.py prepare --out gpu_runs/gradient
python scripts/validate_mps_gradients.py run cpu --out gpu_runs/gradient
python scripts/validate_mps_gradients.py run mps --out gpu_runs/gradient
python scripts/validate_mps_gradients.py compare --out gpu_runs/gradient
python scripts/run_mps_benchmark.py example_configs/apple/pdl1-vhh.yaml \
  --esm-repo "$ESM_REPO" --out gpu_runs/apple-benchmark \
  --measurements gpu_runs/apple-measurements.json
```

On the fixed 42-residue gradient fixture, the CPU/MPS cosine similarities for
the structure and language-model gradients were 0.9999998 and 1.000000.
Eight mutable sites updated; protected entries stayed fixed.
[Gradient results](benchmarks/apple/gradient-comparison.json). Repeating
the Metal calculation with a chunk size of 16 and identical input tensors
also passed both [gradient checks](benchmarks/apple/chunked-gradient-comparison.json).

The separate 240-residue design finished all 150 steps from step zero in
1,341.53 seconds, including 9.84 seconds for the final ESMFold2 critic. Peak
MPS driver allocation sampled once per second was 43.41 GiB; peak process
RSS was 4.46 GiB. These measure different things and should not be added.
The final critic ipTM was 0.567 (below the 0.60 selection threshold); the
design contacted five of the eight requested hotspots through its CDRs.
[Measured timings and memory](benchmarks/apple/m5-max-design.json).

## Install and exercise Protenix V2

Supply a Protenix model data directory with your authorized V2 weights at
`checkpoint/protenix-v2.pt`. The installer checks the entire checkpoint against
SHA-256 `8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599`.
Place available chemical and template metadata in `common/` in the same root.
The Protenix runner downloads missing common files from its upstream endpoints
on first use; for offline use, provide the populated cache. The data root must
contain or allow creation of an `mmcif/` directory for supplied templates. No
weights or private storage locations are included in this distribution.

```bash
./install-macos.sh --protenix-data /path/to/protenix-data
source "$HOME/esmfold2-mac/env.sh"
esmfold2-pipeline check-protenix

# After running the design example above:
esmfold2-pipeline validate gpu_runs/apple-pdl1-vhh \
  --validate-model protenix-v2 --validate-top-k 1 --min-esm-iptm 0 \
  --target-msa-mode none --binder-msa-mode none --use-template true \
  --seeds 101 --n-cycle 10 --n-step 200 --n-sample 1
```

The example deliberately admits the seed-42 sequence to validation despite its
ESMFold2 critic ipTM of 0.567. Omit `--min-esm-iptm 0` in normal use to retain
the 0.60 admission floor. The validator still applies its own normal threshold.
This workflow uses both supplied structures and no external MSA. MSA server,
ligand, nucleic acid, and multi-worker workflows have not been qualified here.

The installer puts Protenix in its own Python environment. It builds the pinned
[fork plus the Mac patch](../vendor/protenix/README.md), selects FP32 native
PyTorch kernels and configures `PROTENIX_DEVICE=mps`. The preflight checks
Metal access in the validator interpreter. In the measured 240-residue case,
Protenix strict-loaded 464,442,431 parameters; both chains contributed template
features. The fold ran 10 recycles and 200 diffusion steps. The pipeline read
the CIF and full confidence data and recorded binder–target ipTM 0.388 and
ipSAE 0.081, below the selection criteria. Synchronized prediction took 55.18 s;
the measured process took 82.71 s including imports and setup. Sampled MPS
driver allocation peaked at 4.93 GiB and process RSS at 4.51 GiB.
[Protenix run record](benchmarks/apple/m5-max-protenix-v2.json).

Using the same feature tensors and fixed coordinates, re-evaluating the
confidence on CPU changed ipTM by 0.00000146. Replaying the same denoising
step on both devices yielded an RMS coordinate difference of 0.0000275 Å.
These checks isolate numerical differences from the random diffusion
trajectory. [CPU/MPS comparison and tolerances](benchmarks/apple/protenix-cpu-comparison.json).

The [Swift/MLX Protenix implementation](https://github.com/javierbq/protenix-mlx#status)
ports much of the model but omits templates. Keeping the native PyTorch path
preserves our two-template validator. If more speed is needed, the
[RFD4 Apple port](https://huggingface.co/bios-labs/RFD4-Proteina-MLX-fp16)
suggests a useful experiment: replace only the expensive repeated trunk
and compare its fixed-input outputs against this baseline. Here the
Pairformer consumed 43.2 of 55.2 prediction seconds.
