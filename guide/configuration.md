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
| `loss` | no | Hotspot, contact-mode, and target-geometry loss settings. Defaults are shown below. |
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
| `conditioning.mode` | no | `distogram` for `target.structure`, otherwise `none` | One of `none`, `distogram`. Set `none` to disable structure conditioning for a structure-backed target. |
| `conditioning.assembly` | no | `auto` | `auto` enables selected target-target chain-pair distograms for multichain `distogram` targets and disables them for single-chain targets. Set `false` to opt out. |
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
| `length` | miniprotein only | `65-150` | Integer, range string, `[min, max]`, or `{min, max}`. |
| `framework` | scFv/VHH only | none | One built-in framework alias/name or one custom framework mapping. Use this for single-framework antibody campaigns. |
| `frameworks` | scFv/VHH only | all bundled frameworks | `all`, a list of built-in aliases/names, or custom framework mappings. Use this to distribute total `num_designs` round-robin across frameworks. |

For scFv and VHH campaigns, omit both `framework` and `frameworks` to sweep all
bundled frameworks. Use exactly one of `framework` or `frameworks` when you want
to restrict the panel. `num_designs` is always the total candidate count. For example, three
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
| `steps` | no | `150` | Number of gradient optimization steps. Set a smaller value only for smoke tests. |

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
| `binder_target_contact_mode` | `legacy` for miniproteins; `mosaic_cdr` for scFv/VHH | `legacy` keeps the original whole-binder target attraction. `mosaic_cdr` is scFv/VHH-only and replaces that attraction with a CDR-scoped Mosaic-style contact loss. |
| `mosaic_cdr_contact_weight` | `0.5` | Weight for the Mosaic-style CDR-to-target entropy contact loss. Used only with `binder_target_contact_mode: mosaic_cdr`. |
| `mosaic_cdr_contact_cutoff_angstrom` | `22.0` | Design-time distogram cutoff for the Mosaic-style CDR contact loss. |
| `mosaic_cdr_num_target_contacts` | `3` | Number of target contacts averaged per CDR residue in the Mosaic-style contact loss and framework penalty diagnostics. |
| `mosaic_framework_contact_penalty_weight` | `0.0` | Optional penalty for framework-to-target contacts in Mosaic CDR mode. The default keeps the penalty off; `1.0` is a reasonable starting value when enabling it. |
| `mosaic_framework_contact_penalty_cutoff_angstrom` | `22.0` | Distogram cutoff used for the optional framework contact penalty. |
| `mosaic_framework_contact_probability_threshold` | `0.2` | Framework contact probability threshold below which no penalty is applied. |
| `mosaic_framework_contact_penalty_scope` | `auto` | One of `auto`, `hotspot`, `target_all`; controls which target residues are penalized for framework contact. |
| `target_geometry_drift.enabled` | `true` for `target.structure`, otherwise `false` | When true, add a soft hinge penalty for target-target distance drift from the input structure. Requires `target.structure`. Set `false` to disable it for a structure-backed target. |
| `target_geometry_drift.weight` | `2.5` | Multiplier for the target geometry drift hinge loss. |
| `target_geometry_drift.tolerance_angstrom` | `0.1` | No drift penalty is applied below this target-target distance RMSE tolerance. |
| `target_geometry_drift.stiffness_angstrom` | `0.1` | Violation scale for the linear hinge: `relu((distance_rmse - tolerance_angstrom) / stiffness_angstrom)`. Lower values make the penalty steeper. |
| `target_geometry_drift.regions` | all selected target residues | Optional chain-keyed residue selectors limiting the drift penalty region. Omit or use `{}` for the whole selected target. |

`hotspot_contact_cutoff_angstrom` is accepted as a compatibility alias for
`hotspot_critic_contact_cutoff_angstrom`.

In `legacy` mode, `entropy_hotspot` is the recommended default when hotspots are
supplied. It adds a hotspot-masked entropy contact objective on top of the
original ESMFold2 structure losses.

For scFv and VHH campaigns, the default
`binder_target_contact_mode: mosaic_cdr` replaces the original whole-binder
target attraction with a Mosaic-style entropy contact loss whose binder rows are
restricted to CDR residues. If `target.hotspots` are omitted, the CDRs are
encouraged to contact any target residue. If `target.hotspots` are configured,
the CDRs are encouraged to contact those hotspot residues. The Mosaic CDR
contact loss does not use
`hotspot_contact_probability_target`; that field only applies to the legacy
`probability_hinge` hotspot mode.

In `mosaic_cdr` mode, `target.hotspots` is the target-side mask for the CDR
attraction. The legacy design-time hotspot knobs
(`hotspot_loss_mode`, `hotspot_contact_weight`,
`hotspot_distogram_contact_cutoff_angstrom`, and `hotspot_num_contacts`) do not
add a second hotspot attraction. `hotspot_critic_contact_cutoff_angstrom` still
controls final hotspot pass/fail reporting and selection when hotspot gating is
enabled.

The framework contact penalty is an optional extension to Mosaic CDR mode. It is
off by default. When enabled, framework residues are binder residues outside the
CDR set. For each framework residue, the loss looks at the top
`mosaic_cdr_num_target_contacts` contact probabilities below
`mosaic_framework_contact_penalty_cutoff_angstrom` and applies
`mean(relu(score - threshold)^2)`, scaled by
`mosaic_framework_contact_penalty_weight`. A practical first enabled value is
`1.0`; treat it as a sweep parameter if framework contacts remain common or CDR
binding becomes too constrained.

`mosaic_framework_contact_penalty_scope` controls only the optional framework
penalty target mask:

- `auto`: if hotspots exist, penalize framework-to-hotspot contact; otherwise,
  penalize framework-to-any-target contact.
- `hotspot`: penalize framework-to-hotspot contact and fail if the penalty is
  enabled without hotspots.
- `target_all`: penalize framework-to-any-target contact even when hotspots are
  configured.

Example Mosaic CDR VHH/scFv loss block:

```yaml
loss:
  binder_target_contact_mode: mosaic_cdr
  mosaic_cdr_contact_weight: 0.5
  mosaic_cdr_contact_cutoff_angstrom: 22.0
  mosaic_cdr_num_target_contacts: 3
  mosaic_framework_contact_penalty_weight: 1.0
  mosaic_framework_contact_penalty_scope: target_all
```

### Target geometry drift restriction

```yaml
loss:
  target_geometry_drift:
    enabled: true
    weight: 2.5
    tolerance_angstrom: 0.1
    stiffness_angstrom: 0.1
```

Structure-backed targets enable this penalty by default. With no `regions`, the
penalty applies to every selected target residue. To limit the penalty to
specific target parts, use the same residue selector style as `target.crop` and
`target.hotspots`:

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
    use_msa: true            # inferred when target MSA server/paths are set
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

When `msa.use_msa` is omitted, target MSA server or provided-MSA settings infer
`use_msa: true`. Set `use_msa: false` explicitly when you want to keep MSA use
off despite those settings.

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
