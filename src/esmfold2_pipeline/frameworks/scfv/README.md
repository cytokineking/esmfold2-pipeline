# Bundled scFv Frameworks

This folder contains the bundled scFv framework YAML records loaded by the
framework registry. Each record is a VH-linker-VL template with six CDR
placeholders:

```text
{hcdr1}, {hcdr2}, {hcdr3}, {lcdr1}, {lcdr2}, {lcdr3}
```

The clinical panel is intended as a practical starting set for antibody
engineering campaigns where the CDRs will be diversified. Active YAML records
are limited to human or humanized variable-domain frameworks with clinical use
or approved-source provenance and public RCSB coordinate support. The source
therapeutic's target is not treated as transferable; these are framework
starting points, not target-specific recommendations.

Primary source for the clinical mAb-derived records:
[Thera-SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab/search/?all=true)
sequence/structure download, accessed 2026-06-17. The downloaded table reports
sequence release PL132 with clinical-stage field dated Feb '25.

## Selection Guidance

Use the core panel when the goal is a diverse clinical framework sweep:

```yaml
binder:
  scaffold: scfv
  frameworks:
    - pembrolizumab
    - belimumab
    - lebrikizumab
    - secukinumab
    - tezepelumab
    - panitumumab
    - guselkumab
    - avelumab
```

Use optional additions when you want a larger sweep or a specific light-chain or
germline family. The CLI value `all` loads every YAML in this directory.
Sequence-only records without public coordinates live under `legacy/` and are
not loaded by the registry.

## Core Clinical Panel

| Alias | Canonical name | Genetics | Light chain | Inferred V genes | Rationale |
| --- | --- | --- | --- | --- | --- |
| `pembrolizumab` | `pembrolizumab_framework_vhvl` | Humanized | Kappa | IGHV1-2 / IGKV3-11 | Approved, exact structural coverage, clinically familiar VH1/VK3 framework distinct from the existing trastuzumab-like VH3/VK1 baseline. |
| `belimumab` | `belimumab_framework_vhvl` | Genetically human | Lambda | IGHV1-69 / IGLV3-19 | Approved human lambda framework; adds VH1-69 and lambda3 diversity with exact structural coverage. |
| `lebrikizumab` | `lebrikizumab_framework_vhvl` | Humanized | Kappa | IGHV2-70 / IGKV4-1 | Approved, structurally covered VH2/VK4 combination; useful because VH2 and VK4 are not represented by the existing bundled records. |
| `secukinumab` | `secukinumab_framework_vhvl` | Genetically human | Kappa | IGHV3-7 / IGKV3-20 | Approved fully human framework with exact structures; provides a VH3 framework that is not simply the common IGHV3-23/3-66 family. |
| `tezepelumab` | `tezepelumab_framework_vhvl` | Genetically human | Lambda | IGHV3-33 / IGLV3-21 | Approved lambda framework with exact structural coverage; broadens lambda coverage while keeping a clinically successful VH3 scaffold. |
| `panitumumab` | `panitumumab_framework_vhvl` | Genetically human | Kappa | IGHV4-61 / IGKV1-33 | Approved fully human VH4/VK1 framework with exact structures; useful as a non-VH1/VH3 alternative. |
| `guselkumab` | `guselkumab_framework_vhvl` | Genetically human | Lambda | IGHV5-51 / IGLV1-40 | Approved fully human lambda framework; adds VH5 and lambda1 diversity with exact structural coverage. |
| `avelumab` | `avelumab_framework_vhvl` | Genetically human | Lambda | IGHV3-23 / IGLV2-14 | Approved, structurally covered lambda2 framework. It intentionally overlaps VH3-23 but adds a different light-chain class from trastuzumab/atezolizumab. |

## Optional Additions

| Alias | Canonical name | Genetics | Light chain | Inferred V genes | When to include |
| --- | --- | --- | --- | --- | --- |
| `daratumumab` | `daratumumab_framework_vhvl` | Genetically human | Kappa | IGHV3-23 / IGKV3-11 | Strong approved, structurally covered clinical scaffold. Optional because its heavy-chain family overlaps several common clinical VH3 frameworks. |
| `dupilumab` | `dupilumab_framework_vhvl` | Genetically human | Kappa | IGHV3-23 / IGKV2-28 | Adds VK2-28 and a long native HCDR3 window from an approved, structurally covered antibody. |
| `anifrolumab` | `anifrolumab_framework_vhvl` | Genetically human | Kappa | IGHV5-51 / IGKV3-20 | Adds a VH5 kappa counterpart to the lambda-heavy core coverage. |
| `tralokinumab` | `tralokinumab_framework_vhvl` | Genetically human | Lambda | IGHV1-18 / IGLV3-21 | Adds another approved VH1/lambda scaffold; optional because lambda3 is already represented in the core. |

## Existing Bundled Baselines

| Alias | Canonical name | Genetics | Light chain | Inferred V genes | Notes |
| --- | --- | --- | --- | --- | --- |
| `trastuzumab` | `trastuzumab_framework_vhvl` | Humanized | Kappa | IGHV3-66 / IGKV1-39 | Existing baseline and useful comparator for continuity with prior campaigns. |
| `atezolizumab` | `atezolizumab_framework_vhvl` | Humanized | Kappa | IGHV3-23 / IGKV1-12 | Existing baseline with approved-source provenance. |

## Coordinate Files

Processed framework-only mmCIF templates are vendored beside the active YAMLs.
The original RCSB downloads are retained under `reference_structures/` with the
same filenames. The processed templates use one synthetic chain whose sequence
is the YAML VH-linker-VL sequence with all six CDR placeholders removed. The
linker is present in the sequence record and intentionally has no atom rows.
CH1, CL, antigen chains, waters, ligands, tags, and CDR atom rows are removed.

When multiple candidate structures were available, the selected reference was
the best-resolution entry that matched the YAML framework sequence, with minor
terminal coordinate gaps allowed when no fully resolved exact framework was
available.

| Framework | Processed template and RCSB reference |
| --- | --- |
| `anifrolumab` | `anifrolumab_4QXG.cif` |
| `atezolizumab` | `atezolizumab_5X8L.cif` |
| `avelumab` | `avelumab_4NKI.cif` |
| `belimumab` | `belimumab_5Y9K.cif` |
| `daratumumab` | `daratumumab_7DUN.cif` |
| `dupilumab` | `dupilumab_6WG8.cif` |
| `guselkumab` | `guselkumab_4M6M.cif` |
| `lebrikizumab` | `lebrikizumab_4I77.cif` |
| `panitumumab` | `panitumumab_5SX5.cif` |
| `pembrolizumab` | `pembrolizumab_5GGS.cif` |
| `secukinumab` | `secukinumab_6WIO.cif` |
| `tezepelumab` | `tezepelumab_5J13.cif` |
| `tralokinumab` | `tralokinumab_5L6Y.cif` |
| `trastuzumab` | `trastuzumab_6BHZ.cif` |

Additional coordinate-backed clinical backfill reference structures are
vendored under `reference_structures/` for future curation, but are not registry
frameworks until their YAML templates have been sequence/CDR-validated:

| Candidate | Vendored coordinate |
| --- | --- |
| `omalizumab` | `reference_structures/omalizumab_4X7S.cif` |
| `natalizumab` | `reference_structures/natalizumab_6FG1.cif` |
| `tocilizumab` | `reference_structures/tocilizumab_8J6F.cif` |
| `nivolumab` | `reference_structures/nivolumab_5GGR.cif` |

## FMC63

FMC63 is best known as the murine anti-CD19 scFv used in several successful
CD19 CAR-T formats. That clinical success does not make the native FMC63
framework suitable for this panel because the inclusion rule here is human or
humanized only.

The legacy `humanized_fmc63_as` record uses a humanized FMC63-derived variant
from [WO2018200496A1](https://patents.google.com/patent/WO2018200496A1/en)
(SEQ ID NO: 17/18) and converts it into this repository's VH-linker-VL template
format. It is not loaded by the active registry because no exact public
coordinate-backed structure was identified for this humanized AS variant.

## Exclusions

Murine, chimeric, and mixed mouse/human frameworks are excluded from the core
panel even when the originating molecule is clinically approved. Examples
include native FMC63 CAR binders, blinatumomab, rituximab-derived frameworks,
cetuximab-derived frameworks, and other chimeric anti-CD19 antibodies. They can
be useful in their native therapeutic contexts, but they do not meet this
folder's human/humanized framework criterion.

Sequence-only human or humanized records without exact public coordinates are
kept under `legacy/` for provenance. They are not included in `--frameworks all`.
