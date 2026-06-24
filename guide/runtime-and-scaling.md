# Runtime & Scaling Guidance

[← Back to README](../README.md) · [Documentation index](README.md)

## Preprint campaign scale

The ESMFold2 preprint reports the following campaign scale for binder design:

- **Minibinders:** about 15,000 candidates for a lower-compute pool and about
  67,000 candidates for a higher-compute pool.
- **scFvs:** about 28,000 candidates for a lower-compute pool and about 117,000
  candidates for a higher-compute pool.
- **Wetlab selection:** 84 designs per target/modality/compute setting.
- **Production search:** 150 optimization steps.
- **Minibinder lengths:** sampled from 60 to 200 residues.
- **Minibinder selection:** filter designs with pI above 6.0 before ranking.
- **Ranking:** ipTM and/or distogram ipTM proxy, optionally across multiple
  model critics in the paper's full selection workflow.

This pipeline currently ranks by completed critic rows in SQLite and does not
yet compute pI. If you want to reproduce the preprint's minibinder pI filter,
apply that filter externally to `esmfold2/selected_designs.csv` or
`esmfold2/metrics_all.csv` before ordering.

## Compute estimates

The same paper reports compute on H100 GPUs of roughly:

- **Minibinders:** about 500 H100-hours for 15,000 candidates and about 2,400
  H100-hours for 67,000 candidates.
- **scFvs:** about 1,800 H100-hours for 28,000 candidates and about 7,700
  H100-hours for 117,000 candidates.

As a rough planning rule from those numbers:

```text
minibinder wall-clock hours ~= num_designs * 2 minutes / 60 / num_gpus
scFv wall-clock hours       ~= num_designs * 4 minutes / 60 / num_gpus
```

Use the scFv estimate as the conservative starting point for VHH campaigns until
you have target- and framework-specific timing data. Actual runtime depends on
target length, binder/framework length, model choice, GPU type, kernel
availability, and whether model weights are already in cache.

## Recommended campaign progression

1. Quick GPU check with one design and `campaign.steps: 2`.
2. A small pilot with 3 to 10 designs and 20 to 60 steps.
3. A target-specific pilot with 20 to 100 designs and 150 steps.
4. A production campaign with hundreds to thousands of designs per target,
   scaled across available GPUs.
5. Export a shortlist of 84 designs when preparing one 96-well-plate style
   experimental batch.

Use `--max-shards` or `--max-shards-per-worker` for pilot runs before
committing all GPUs to a large campaign.
