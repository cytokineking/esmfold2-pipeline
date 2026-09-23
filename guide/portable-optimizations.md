# Portable ESMFold2 design optimizations

[← Documentation index](README.md)

Set `ESMFOLD2_PIPELINE_OPTIMIZATIONS=portable` in the worker environment to
enable source-only ESMFold2 design optimizations. The default is `off`. The
setting is read before local models are cached, so changing it requires a new
worker process. It applies to the local design backend and leaves late
confidence steps and critic folds eager.

The portable path avoids the confidence head and defers the one-step structure
sample when an early design fold only needs the distogram. A request for sampled
coordinates computes the genuine seeded result. It also transfers designed
sequence tokens to the CPU in one operation, reuses the active target's
distogram pair bias, and reuses the design's PLM vocabulary, mask token, and
score positions. The PLM still draws the same random mask shapes and performs
the same four mask passes.

The path uses the image's existing PyTorch, ESM, Transformers, and
cuEquivariance packages. No CUDA extension or model precision setting changes.
The active target cache holds one pair-bias tensor per model. A new target or
shape replaces the entry. Model, PLM, and template-cache counters are written
to the worker log after each design. Set
`ESMFOLD2_PIPELINE_PROFILE_DESIGN_STEPS=1` for synchronized per-step phase
timings during a dedicated benchmark; leave it unset for throughput runs.

On a qualified GPU image, compare the same config with separate output
directories and separate worker processes. For example, source the image's
`/opt/esmfold2/env.sh`, then use `scripts/benchmark_portable.py` with
`--executable`, `--config`, `--out`, and `--mode off|portable`. Pass
`--source-root` for a source-only candidate overlay. The harness records the
loaded source path, config hash, wall time, GPU memory peak, and a timestamped
run log. `scripts/inspect_portable_fold_variability.py` reports direct stock
run-to-run variation alongside the portable result on an H100; the pinned
stock model is not bitwise repeatable there, so this is diagnostic only.
`tests/test_design_plm_cuda_parity.py` checks PLM loss, gradient, masked
inputs, and RNG state on CUDA.

Keep the optimization default off in new images until real campaigns pass
model, gradient, memory, and validation checks on the GPU family being served.
The reference for the confidence and lazy-sampling approach is Anthropic's
[ESMFold2 design kit](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/ef2inv),
licensed under Apache-2.0; its license is retained in
`third_party_licenses/anthropic_ef2inv_apache_2_0.txt`.
