# Optional Protenix Validation

[← Back to README](../README.md) · [Documentation index](README.md)

Protenix validation is a separate post-critic stage for miniprotein, VHH, and
built-in scFv structural-template campaigns. It re-folds selected designs with
the template-capable [`cytokineking/Protenix`](https://github.com/cytokineking/Protenix)
fork and produces an orthogonal confidence signal (ipTM / ipSAE) plus structural
RMSD against the ESMFold2 prediction.

For normal campaigns, put a `validation` block in the YAML and run `launch`.
After ESMFold2 design, aggregation, selection, and export, `launch` runs the
same validation lifecycle described below and then writes the combined analysis
ranking under `ranked_results/`.

`validate-plan` supports built-in scFv campaigns when bundled structural
framework templates are available. Custom scFv frameworks still require explicit
structural-template support or future paired VH/VL MSA support.

## One-shot wrapper

```bash
uv run esmfold2-pipeline validate /path/to/runs/my-campaign-n100 \
  --validate-model protenix-v2 \
  --validate-top-k 100 \
  --min-esm-iptm 0.70 \
  --min-validation-iptm 0.75 \
  --gpus 0-3
```

Use `validate` when you want to rerun validation, change validation-only CLI
overrides, or operate on a campaign that was launched without validation. When
`launch` sees validation config, it starts one background validation MSA worker
by default so target/VHH/miniprotein MSA work can drain while critic results are
still being produced. Use `--validation-msa-workers 0` to disable this, or set a
larger worker count while keeping the shared default
`--msa-max-requests-per-minute 5` server throttle.

By default, validation planning requires ESMFold2 binder-target ipTM >= 0.6 and
Protenix validation requires binder-target ipTM >= 0.6. This lenient floor keeps
obviously weak designs out of expensive validation and final pass sets. Set
`--min-iptm 0` during `launch`, or set `--min-esm-iptm 0` and
`--min-validation-iptm 0` in the validation CLI/YAML, to disable the floor.

Target MSA server or provided-MSA settings infer `use_msa: true` when
`use_msa` is omitted. Explicit `use_msa: false` remains an opt-out.

## Final consensus ranking

ESMFold2 ipTM ranking is used first to choose which designs are worth sending
to Protenix. After validation, `analyze` replaces that selection order with a
final consensus rank based on ESMFold2 ipTM, scoped Protenix ipTM/ipSAE, and
target-aligned binder C-alpha RMSD. The default RMSD gate is `2.5` angstrom;
RMSD also contributes 10% of the final score so that, within the passing set,
closer pose agreement is preferred.

The two ranks have distinct meanings:

- `selection_rank` is the ESMFold2 pre-validation order and remains useful for
  comparing design-time confidence with validation outcomes. It is exported as
  `esmfold2_rank` in the compact CSV.
- `validator_rank` is the evaluator-only order across completed rows with valid
  evaluator metrics, ranked by evaluator score, validator ipTM, validator
  ipSAE, then design name. It does not include RMSD or ESMFold2 confidence.
- `final_rank` is the post-validation consensus order used for rank-prefixed
  paired structures and the compact user-facing ranking CSV.
- `validation_rank` in the validator report is a validator-local diagnostic,
  not the final campaign rank.

Final-ranking eligibility requires completed validation, `validator_passed`,
valid ESMFold2 and scoped validator ipTM values, and a usable RMSD when the gate
or RMSD weighting is enabled. Missing ipSAE falls back to scoped validator ipTM
unless the user explicitly configured an ipSAE validation threshold.

The default score is:

```text
evaluator_score = sqrt(protenix_iptm * protenix_ipsae)
confidence_score = esmfold2_iptm^0.50
                 * protenix_iptm^0.25
                 * protenix_ipsae^0.25
agreement_score = exp(-0.5 * (binder_rmsd / 2.5)^2)
final_score = confidence_score^0.90 * agreement_score^0.10
```

The monotonic confidence calculation is Pareto-consistent: a design that is
worse on every confidence component cannot outrank its dominator. The exported
`pareto_front` field exposes ESMFold2/evaluator confidence fronts as a
diagnostic, but final sorting uses `final_score` rather than Pareto front number.

Configure this under `analysis.ranking`, or override it for a reranking run:

```bash
uv run esmfold2-pipeline analyze /path/to/campaign \
  --max-binder-rmsd-angstrom 2.5 \
  --rmsd-weight 0.10
```

`combined_ranking.csv` contains only eligible final-ranked designs.
`ranking_diagnostics.csv` retains every validator row, including exclusion
reasons and intermediate scores. Only eligible rows are copied into the
rank-prefixed paired structure folders.

Rerunning `analyze` only reads the existing database and structures and rewrites
the compact ranking CSV, diagnostics, summary, plots, and top-ranked copies. It
does not rerun ESMFold2, MSA generation, or Protenix.

## Cold starts and retries

On a fresh machine, the first validation run can be much slower than later
runs. Protenix may need to load the checkpoint and compile CUDA extensions
before it writes any CIF outputs, so GPU utilization and scratch output can look
quiet for several minutes.

Validation attempts are durable. If a Protenix subprocess times out or exits
before the retry budget is exhausted, the failed attempt remains in the campaign
attempt log and the task returns to pending. A later successful attempt makes
the validation task complete, and `launch`/`validate` continues to reporting
and analysis once the final task state is clean.

## Batching

Protenix validation batches up to `--validation-batch-size` tasks per
subprocess invocation. The default is `10`; multi-GPU validation shrinks that
effective batch size when fewer ready tasks are available than
`batch_size * workers`, so small campaigns are still spread across workers.
Batch outputs are promoted after each Protenix subprocess exits.

## Lower-level lifecycle

The lower-level lifecycle remains available for explicit control:

```bash
uv run esmfold2-pipeline validate-plan /path/to/runs/my-campaign-n100 \
  --validate-model protenix-v2 \
  --validate-top-k 100 \
  --min-esm-iptm 0.70 \
  --min-validation-iptm 0.75

uv run esmfold2-pipeline validate-msa-run /path/to/runs/my-campaign-n100

uv run esmfold2-pipeline validate-msa-retry /path/to/runs/my-campaign-n100

uv run esmfold2-pipeline validate-run-multi \
  /path/to/runs/my-campaign-n100 \
  --gpus 0-3

uv run esmfold2-pipeline validate-report /path/to/runs/my-campaign-n100
```

Validation outputs land under `validation/{model}/` and ranked paired structures
under `ranked_results/`. See [Output layout](outputs.md) for the full tree.
