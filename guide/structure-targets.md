# Structure Targets & Hotspots

[← Back to README](../README.md) · [Documentation index](README.md)

## Target input modes

Target input has two modes.

**Direct sequence targets** supply the target protein sequence in YAML:

```yaml
target:
  name: custom_target
  sequence: ACDEFGHIKLMNPQRSTVWY
```

**Structure-derived configs** use a PDB/mmCIF file. The pipeline parses the
target, resolves chain/residue selectors, writes normalized target artifacts,
and can attach target-chain distogram conditioning during design and critic
folds.

```yaml
target:
  name: custom_structure_target
  structure: /path/to/target.cif
  chains: [A, C]
```

`target.sequence` and `target.structure` are mutually exclusive. A structure
template supplies its own sequence, so `target.sequence` is not required when
`target.structure` is set. Conversely, a direct sequence target does not require
`target.structure`.

## Full structure-target example

```yaml
target:
  name: il2rb_hotspot
  structure: /path/to/target.cif
  chains: [A]
  structure_indexing: auth_seq_id
  crop: ["1-130"]
  hotspots: "A:88,91"
  conditioning:
    mode: distogram

binder:
  scaffold: miniprotein
  length: 80-140

campaign:
  num_designs: 20
  inversion_model: cutoff2025
  critics:
    - cutoff2025
  steps: 150

loss:
  hotspot_loss_mode: entropy_hotspot
  hotspot_contact_weight: 2.0
  hotspot_distogram_contact_cutoff_angstrom: 20.0
  hotspot_critic_contact_cutoff_angstrom: 5.0
  hotspot_num_contacts: 1
  hotspot_contact_probability_target: 0.6

output: /path/to/runs/il2rb-hotspot-n20
```

## Hotspots

Hotspot syntax supports explicit mappings:

```yaml
target:
  hotspots:
    A: [88, 91]
```

and BindCraft-style strings:

```yaml
target:
  hotspots: "A:88,91"
```

## Structure indexing

Use `structure_indexing: auth_seq_id` when you want selectors to match author
numbering from a PDB/mmCIF file. Use `label_seq_id` for label numbering. With
`auto`, the parser accepts unambiguous selectors and fails if author and label
numbering disagree.

## Distogram conditioning

Structure-target conditioning uses the shared ESMFold2 fold wrapper, so it
applies to miniprotein, scFv, and VHH campaigns. Same-chain target distograms
can condition each selected target chain, and `conditioning.assembly: true`
additionally conditions selected target-target chain-pair blocks in both the
design and critic folds. Target-binder geometry is left for the model to
predict.

Conditioning injects the raw target distances as model pair features before the
folding trunk, and is used for both design folds and critic re-evaluation. If
the selected ESMFold2 model does not expose the required distance-bin embedding,
the pipeline fails fast instead of silently running without structure
conditioning.

### Partial templates for unresolved residues

If the selected structure has unresolved residues but the full sequence register
can be recovered from mmCIF metadata, PDB `SEQRES`, or
`target.structure.sequences`, distogram conditioning automatically uses a
partial mask. Unresolved residues stay in the target sequence and folded output,
but their missing template distances are excluded from conditioning.

Structure-backed targets default to `conditioning.mode: distogram`, including
YAML-free `launch --target-structure` runs. Set `conditioning.mode: none` to opt
out, or set `conditioning.require_resolved: true` when you want strict
dense-template behavior that rejects any unresolved representative coordinates:

```yaml
target:
  conditioning:
    mode: distogram
    require_resolved: true
```

## PDB output limit

Selected target chains and the auto-assigned binder chain must be representable
as unique one-character PDB chain IDs. `check` fails early for multi-character or
otherwise incompatible chain IDs. mmCIF structure export is planned for that
edge case.
