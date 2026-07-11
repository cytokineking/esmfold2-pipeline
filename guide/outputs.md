# Output Layout

[← Back to README](../README.md) · [Documentation index](README.md)

A completed `launch` campaign looks like this. If validation is not configured,
the `validation/` and `ranked_results/` sections are absent unless you run
validation later.

```text
campaign/
  config.yaml
  resolved_config.yaml
  campaign.sqlite
  target/
    normalized_target.cif
    residue_map.csv
    chain_summary.json
    conditioning/
      chain_A_rep_coords.npy
      chain_A_distogram.npy
  logs/
    design_workers/
      local-gpu-gpu0-<run-id>.log
    validation_workers/
      validation-gpu-0.log
  esmfold2/
    structures/
      s000_seed000_c000.pdb
      s001_seed001_c000.pdb
    campaign_summary.json
    metrics_all.csv
    selected_designs.csv
    selected_structures/
      selected_manifest.csv
      s000_seed000_c000.pdb
  validation/
    protenix_v2/
      structures/
        passing/
        rejected/
        .staging/
      msa_cache/
      validation_results.csv
      structure_samples.csv
      validation_summary.json
  ranked_results/
    combined_ranking.csv
    ranking_diagnostics.csv
    ranking_summary.json
    plots/
    top_ranked/
      esmfold2/
        rank0001_<candidate_id>_esmfold2.pdb
      protenix_v2/
        rank0001_<candidate_id>_protenix_v2.cif
```

## Important details

- `campaign.sqlite` is the source of truth for shard state, attempts, designed
  sequences, scalar metrics, and artifact paths.
- Designed binder sequences are stored in SQLite and exported CSVs. The current
  worker does not write per-candidate FASTA files.
- For scFv campaigns, ESMFold2 and validation CSVs include CDR sequence columns
  (`cdrh1`, `cdrh2`, `cdrh3`, `cdrl1`, `cdrl2`, `cdrl3`) when the framework
  defines CDR positions.
- For VHH campaigns, exported CSVs include only heavy-chain CDR sequence columns
  (`hcdr1`, `hcdr2`, `hcdr3`). VHH CSVs do not include light-chain CDR columns.
- `esmfold2/structures/` contains final predicted complex PDBs from completed
  ESMFold2 critics.
- `target/` exists only for structure-derived target campaigns.
- For structure-target campaigns, exported `iptm` is binder-target scoped when
  ESMFold2 provides chain-pair iPTM. The raw all-chain complex score is retained
  in SQLite `critic_metrics.metrics_json` and final CSVs as `complex_iptm`; the
  metrics JSON and final CSVs also record `iptm_scope`. Detailed per-target-chain
  values remain in metrics JSON as `binder_target_iptm_by_chain`. This prevents
  multi-chain target-target confidence from inflating binder ranking.
- When `loss.target_geometry_drift.enabled` is true, the final CSV metadata row
  records drift settings. Candidate rows include region-aware
  `target_geometry_drift_distance_rmse` and
  `target_geometry_drift_aligned_rmsd`.
- When `loss.binder_target_contact_mode: mosaic_cdr` is used, final CSVs include
  Mosaic CDR contact settings and diagnostics: `binder_target_contact_mode`,
  `mosaic_cdr_contact_scope`, CDR contact probability summaries, the CDR contact
  loss, and optional framework contact penalty settings, scope, probability
  summaries, and loss. A framework penalty is reported as enabled only when
  `mosaic_framework_contact_penalty_weight` is greater than zero.
- `logs/design_workers/` is created by `run-multi`; `logs/validation_workers/`
  is created by `validate-run-multi`.
- `esmfold2/campaign_summary.json` is refreshed by `aggregate`, `select`, and
  `export`. It records status counts, retries/failures, hotspot pass rate,
  iPTM scope counts, selected/export counts, and top candidate paths.
- `esmfold2/metrics_all.csv` is written by `aggregate`.
- `esmfold2/selected_designs.csv` is written by `select`.
- `esmfold2/selected_structures/` is written by `export` and contains copied
  ESMFold2 PDBs plus a manifest.
- `validation/{model}/validation_results.csv` has one row per validation task.
  `structure_samples.csv` has one row per promoted validator sample. For
  Protenix, model names are slugged, for example `protenix-v2` becomes
  `validation/protenix_v2/`.
- `validation/{model}/structures/.staging/` is a hidden crash-safety staging
  folder used during CIF promotion; completed runs normally leave it empty.
- `ranked_results/combined_ranking.csv` is the compact user-facing shortlist. It
  contains eligible designs only, with one final rank and the raw metrics needed
  to interpret it.
- `ranked_results/ranking_diagnostics.csv` retains every analyzable validator
  row, including ineligible designs, exclusion reasons, intermediate scores,
  Pareto fronts, model internals, runtimes, identifiers, and copied-artifact
  paths.
- `ranked_results/top_ranked/` copies only eligible designs up to the configured
  top-k, grouped into `esmfold2/` and `{validator}/` subfolders with paired
  `rank0001_<candidate_id>_<model>` filenames. The prefix is the final consensus
  rank and joins each structure back to its `combined_ranking.csv` row.

For a VHH campaign with hotspots and ipSAE, `combined_ranking.csv` uses this
column order:

| Column | Meaning |
| --- | --- |
| `rank` | Final consensus order and the rank in copied filenames. |
| `design_name` | Stable candidate name. |
| `sequence` | Designed binder sequence. |
| `framework` | Antibody framework; omitted for miniproteins. |
| `hcdr1`, `hcdr2`, `hcdr3` | Designed VHH CDRs; scFv exports its six CDR columns instead. |
| `binder_length` | Sequence length, retained as a convenient filter. |
| `consensus_score` | Final confidence/agreement score used for ranking. |
| `esmfold2_rank` | Original ESMFold2 selection order. |
| `esmfold2_iptm` | Scoped ESMFold2 binder-target ipTM. |
| `validator_rank` | Evaluator-only confidence rank before RMSD/ESMFold2 consensus. |
| `validator_iptm`, `validator_ipsae` | Scoped evaluator confidence metrics. |
| `binder_rmsd_angstrom` | Binder C-alpha RMSD after target alignment. |
| `esmfold2_hotspot_distance_angstrom` | ESMFold2 hotspot distance; omitted without hotspots. |
| `validator_hotspot_distance_angstrom` | Evaluator hotspot distance; omitted without hotspots. |
| `esmfold2_structure`, `validator_structure` | Canonical paired structure paths. |

`validation_rank` belongs to `validation/{model}/validation_results.csv` and is
the validator-local order. It should not be used as the final campaign rank.

## Reconciling the database

Run `status` any time to reconcile the database against expected artifacts:

```bash
uv run esmfold2-pipeline status /path/to/campaign
```

If artifacts are missing or untracked, `status` exits nonzero and reports the
issue.
