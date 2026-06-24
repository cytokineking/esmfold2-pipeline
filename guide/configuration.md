# YAML Configuration Reference

[← Back to README](../README.md) · [Documentation index](README.md)

Ready-made example configs live under
[`example_configs/`](../example_configs/) (miniprotein and scFv). This page is
the field-level reference.

## Top-level keys

| Key | Required | Notes |
| --- | --- | --- |
| `target` | yes | Sequence target or structure target definition. |
| `binder` | yes | Binder scaffold definition. |
| `campaign` | yes | Number of designs, model names, and steps. |
| `output` | yes | Campaign directory. Can be overridden with `--out` on `check`, `plan`, or `launch CONFIG`. |
| `loss` | no | Hotspot loss settings. Defaults are shown below. |
| `validation` | no | Optional Protenix validation and automatic MSA settings. See [the validation section](#validation) and [Validation](validation.md). |

Minimal direct-sequence example:

```yaml
target:
  name: my_target
  sequence: MAGEDVGAPPDHLWVHQEGIYRDEYQRTWVAVVEEET   # one-chain target protein sequence

binder:
  scaffold: miniprotein
  length: 80-140

campaign:
  num_designs: 100
  inversion_model: cutoff2025
  critics:
    - cutoff2025
  steps: 150

output: /path/to/runs/my-campaign-n100
```

## `target`

| Field | Required | Default | Notes |
| --- | --- | --- | --- |
| `name` | no | structure filename stem or `sequence_target` | Display label for the target/campaign. Optional when `sequence` or `structure` is set. |
| `sequence` | sequence target | none | One-chain target protein sequence. Mutually exclusive with `structure`. |
| `structure` | structure mode only | none | Path string or mapping with `path`/`file` and optional `sequences`. Supports `.pdb`, `.cif`, `.mmcif`. Mutually exclusive with `sequence`. |
| `structure.sequences` | no | parsed from structure metadata | Chain-keyed full target sequences for structures whose coordinate records or sequence metadata are incomplete. |
| `chains` | structure targets should set this | all chains | Ordered target chains to include. Multichain miniprotein targets are supported. |
| `structure_indexing` | no | `auto` | One of `auto`, `auth_seq_id`, `label_seq_id`. |
| `crop` | no | none | Residue selectors to keep. List for one chain, or mapping by chain. |
| `hotspots` | no | none | Chain-qualified residue selectors used for design steering and final hotspot gating. |
| `conditioning.mode` | no | `none` | One of `none`, `distogram`. |
| `conditioning.assembly` | no | `true` for explicitly multichain `distogram` targets, otherwise `false` | When true, condition selected target-target chain-pair distograms. Requires `mode: distogram`. |
| `conditioning.chain_pairs` | no | `auto` | `auto` means all selected target-target pairs; otherwise list pairs such as `[[A, C]]`. |
| `conditioning.representative_atom` | no | `esmfold2_default` | Uses CB except glycine CA, matching the current parser behavior. |
| `conditioning.partial` | no | `true` | Advanced dense-only opt-out. Omit in normal configs; by default unresolved structural pairs are masked out of distogram conditioning. |
| `conditioning.require_resolved` | no | `false` | Fail if representative coordinates are missing. Set true for strict dense-template behavior. |

See [Structure targets & hotspots](structure-targets.md) for the full
explanation of target modes, indexing, and distogram conditioning.

Residue selector examples:

```yaml
crop: ["10-140"]
hotspots: "A:88,91"
hotspots:
  A: ["88", "91", "100-105"]
```

## `binder`

| Field | Required | Default | Notes |
| --- | --- | --- | --- |
| `scaffold` | yes | none | `miniprotein`, `scfv`, or `vhh`. |
| `length` | miniprotein only | `60-200` | Integer, range string, `[min, max]`, or `{min, max}`. |
| `framework` | scFv/VHH only | none | One built-in framework alias/name or one custom framework mapping. Use this for single-framework antibody campaigns. |
| `frameworks` | scFv/VHH only | none | List of built-in aliases/names or custom framework mappings. Use this to distribute total `num_designs` round-robin across frameworks. |

Use exactly one of `framework` or `frameworks` for scFv and VHH campaigns.
`num_designs` is always the total candidate count. For example, three
frameworks and `num_designs: 9` produces three candidates per framework by
round-robin assignment.

### Bundled frameworks

The bundled scFv clinical panel and VHH panel are documented in their own
sub-READMEs (with genetics, light-chain class, V genes, and selection
rationale):

- scFv frameworks: [`src/esmfold2_pipeline/frameworks/scfv/README.md`](../src/esmfold2_pipeline/frameworks/scfv/README.md)
- VHH frameworks: [`src/esmfold2_pipeline/frameworks/vhh/README.md`](../src/esmfold2_pipeline/frameworks/vhh/README.md)

Short aliases (e.g. `trastuzumab`) and canonical `*_framework_vhvl` / `*_framework_vhh`
names are both accepted; resolved campaign configs keep the canonical names for
reproducibility. Sequence-only records without exact public coordinates live
under each framework folder's `legacy/` subdirectory and are not loaded by
`--frameworks all`.

Single built-in framework:

```yaml
binder:
  scaffold: scfv
  framework: trastuzumab
```

Multi-framework campaign:

```yaml
binder:
  scaffold: scfv
  frameworks:
    - trastuzumab
    - atezolizumab
    - belimumab
```

VHH campaign:

```yaml
binder:
  scaffold: vhh
  frameworks:
    - caplacizumab
    - ozoralizumab_tnf
    - vobarilizumab_il6r
```

### Custom frameworks

Custom template framework (samples mutable CDR lengths from `cdr_lengths`):

```yaml
binder:
  scaffold: scfv
  framework:
    name: lab_template
    template: EVQL...{hcdr1}...{hcdr2}...{hcdr3}...{lcdr1}...{lcdr2}...{lcdr3}...VEIK
    cdr_lengths:
      hcdr1: 7-9
      hcdr2: 5-6
      hcdr3: 9-15
      lcdr1: 11-16
      lcdr2: 7
      lcdr3: 9
```

Custom fixed scFv sequence (keeps the full input length, mutates explicit
1-based inclusive `cdrs` ranges):

```yaml
binder:
  scaffold: scfv
  framework:
    name: lab_fixed_scfv
    sequence: QVQLKQSGPGLVQPSQSLSITCTVSGFSLTNYGVHWVRQSPGKGLEWLGVIWSGGNTDYNTPFTSRLSINKDNSKSQVFFKMNSLQSNDTAIYYCARALTYYDYEFAYWGQGTLVTVSGGGGSGGGGSGGGGSGGGGSDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGISRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELK
    mutate: cdrs
    cdrs:
      hcdr1: 26-35
      hcdr2: 51-65
      hcdr3: 98-108
      lcdr1: 162-172
      lcdr2: 188-194
      lcdr3: 226-234
```

`check` fails early if a `cdrs` range is missing, out of bounds, overlapping, or
out of sequence order. Bundled VHH frameworks use `cdr1`, `cdr2`, and `cdr3`
placeholders internally and report designed CDRs as heavy-chain CDR columns.

### Binder chain IDs

Structure-target outputs preserve selected target chain IDs when possible. The
binder chain ID is assigned automatically as the first unused one-character ID
from `A-Z`, then `a-z`, then `0-9`. For example, targets `A,C` produce binder
chain `B`; targets `A,B,C` produce binder chain `D`; targets `X,Y` produce
binder chain `A`.

The assigned binder chain is written to SQLite as `candidates.binder_chain_id`,
to `design_metrics_json` as `binder_chain_id`, and to final aggregate/ranked and
selected-manifest CSV exports. Miniprotein, scFv, and VHH designs are
represented as one binder chain. Separate heavy/light antibody chain output is
planned for a later antibody-specific workflow.

## `campaign`

| Field | Required | Default | Notes |
| --- | --- | --- | --- |
| `num_designs` | yes | none | Number of candidates. One deterministic shard is created per design. |
| `seed_start` | no | `0` | First deterministic seed. Useful when extending a campaign. |
| `inversion_model` | no | `ESMFold2-Experimental-Cutoff2025` | Model used in the design loop. Short aliases are accepted. |
| `critics` | no | `[ESMFold2-Experimental-Cutoff2025]` | Currently exactly one critic is supported per campaign. Short aliases are accepted. |
| `steps` | no | `2` | Number of gradient optimization steps. Use `150` for paper-style production runs. |

The CLI `--model` flag and YAML model fields accept these short aliases:

| Alias | Full model name |
| --- | --- |
| `cutoff2025` | `ESMFold2-Experimental-Cutoff2025` |
| `fast-cutoff2025` | `ESMFold2-Experimental-Fast-Cutoff2025` |
| `experimental` | `ESMFold2-Experimental` |
| `fast` | `ESMFold2-Experimental-Fast` |

Use fast models for quick checks, debugging, or broad exploratory runs. Use the
default cutoff model for higher-confidence production runs unless you are
intentionally comparing model settings.

## `loss`

All loss fields are optional.

| Field | Default | Notes |
| --- | --- | --- |
| `hotspot_loss_mode` | `entropy_hotspot` | One of `entropy_hotspot`, `probability_hinge`. |
| `hotspot_contact_weight` | `2.0` | Conservative temporary default from early calibration. Set to `0` to disable hotspot loss while preserving hotspot metrics. |
| `hotspot_distogram_contact_cutoff_angstrom` | `20.0` | Broad design-time distogram contact cutoff. |
| `hotspot_critic_contact_cutoff_angstrom` | `5.0` | Tight final heavy-atom contact cutoff used for hotspot pass/fail. |
| `hotspot_num_contacts` | `1` | Number of binder contacts requested per hotspot residue in the loss. |
| `hotspot_contact_probability_target` | `0.6` | Used by `probability_hinge`. |
| `target_geometry_drift.enabled` | `false` | When true, add a soft hinge penalty for target-target distance drift from the input structure. Requires `target.structure`. |
| `target_geometry_drift.weight` | `2.5` | Multiplier for the target geometry drift hinge loss. |
| `target_geometry_drift.tolerance_angstrom` | `0.1` | No drift penalty is applied below this target-target distance RMSE tolerance. |
| `target_geometry_drift.stiffness_angstrom` | `0.1` | Violation scale for the linear hinge: `relu((distance_rmse - tolerance_angstrom) / stiffness_angstrom)`. Lower values make the penalty steeper. |
| `target_geometry_drift.regions` | all selected target residues | Optional chain-keyed residue selectors limiting the drift penalty region. Omit or use `{}` for the whole selected target. |

`hotspot_contact_cutoff_angstrom` is accepted as a compatibility alias for
`hotspot_critic_contact_cutoff_angstrom`.

`entropy_hotspot` is the recommended default when hotspots are supplied. It
adds a hotspot-masked entropy contact objective on top of the original ESMFold2
structure losses.

### Target geometry drift restriction (opt-in)

```yaml
loss:
  target_geometry_drift:
    enabled: true
    weight: 2.5
    tolerance_angstrom: 0.1
    stiffness_angstrom: 0.1
```

With no `regions`, the penalty applies to every selected target residue. To
limit the penalty to specific target parts, use the same residue selector style
as `target.crop` and `target.hotspots`:

```yaml
loss:
  target_geometry_drift:
    enabled: true
    regions:
      A: ["45-70", "88"]
      B: all
```

When target geometry drift is enabled, final CSV metadata rows include
`target_geometry_drift_enabled`, `target_geometry_drift_weight`,
`target_geometry_drift_tolerance_angstrom`,
`target_geometry_drift_stiffness_angstrom`, and `target_geometry_drift_regions`.
Candidate rows stay compact and add only
`target_geometry_drift_distance_rmse` and
`target_geometry_drift_aligned_rmsd`, computed over the configured drift region.

## `validation`

Optional. Adding a `validation` block enables the post-design
[Protenix validation](validation.md) stage during `launch` and turns on
automatic MSA handling: when a campaign is launched with this block present,
`launch` also starts a background MSA prefetch worker by default, so target and
binder MSAs are fetched, cached, and reused across designs and resumes.

The block is passed through to the validation stage; its fields mirror the
`validate` CLI flags
([CLI reference](cli-reference.md#selection-export-and-validation-flags)). A
representative block:

```yaml
validation:
  top_k: all                 # validate every selected design, or an integer
  require_hotspot_contact: never
  max_attempts: 3
  msa:                       # automatic MSA fetch + cache + reuse
    use_msa: true
    target: server           # fetch the target MSA from the MMseqs2 server
    binder: auto             # auto | none | single_sequence
    server_url: https://api.colabfold.com
    pairing_strategy: greedy
  protenix:
    use_template: true       # condition Protenix on the target / framework structure
    n_sample: 1
    n_step: 200
    n_cycle: 10
    validation_batch_size: 10
```

Protenix validation covers miniprotein and VHH campaigns, plus built-in scFv
campaigns with bundled structural framework templates. See
[Validation](validation.md) for the full lifecycle.

## Recommended defaults

General miniprotein production:

```yaml
binder:
  scaffold: miniprotein
  length: 80-140

campaign:
  steps: 150
  inversion_model: cutoff2025
  critics:
    - cutoff2025
```

Structure-target hotspot design:

```yaml
target:
  chains: [A]
  structure_indexing: auth_seq_id
  hotspots: "A:88,91"
  conditioning:
    mode: distogram

loss:
  hotspot_loss_mode: entropy_hotspot
  hotspot_contact_weight: 2.0
  hotspot_distogram_contact_cutoff_angstrom: 20.0
  hotspot_critic_contact_cutoff_angstrom: 5.0
```

Quick debugging:

```yaml
campaign:
  num_designs: 3
  inversion_model: fast-cutoff2025
  critics:
    - fast-cutoff2025
  steps: 2
```
