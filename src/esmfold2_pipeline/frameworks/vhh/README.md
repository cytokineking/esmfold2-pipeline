# Bundled VHH Frameworks

This folder contains bundled VHH framework YAML records loaded by the
`esmfold2-pipeline` framework registry. They are available through
`binder.scaffold: vhh` and can be selected by alias, canonical name, or
`--frameworks all`.

Each YAML record is a single-domain framework template with IMGT-derived CDR
placeholders:

```text
{cdr1}, {cdr2}, {cdr3}
```

The core panel is selected for practical framework exploration rather than for
target-specific transfer. Active YAML records are limited to clinically exposed
VHH frameworks with public RCSB coordinate support. The source therapeutic
target is useful provenance, but users should assume the CDRs will be
diversified for a new target.

Primary sequence source:
[Thera-SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab/search/?all=true)
sequence/structure download, accessed 2026-06-17. The downloaded table reports
sequence release PL132 with clinical-stage field dated Feb '25.

## Core Panel

| Alias | Canonical name | Source therapeutic | Clinical provenance | Inferred annotation | Rationale |
| --- | --- | --- | --- | --- | --- |
| `caplacizumab_vhh` | `caplacizumab_framework_vhh` | Caplacizumab | Approved Nanobody, VWF | alpaca IGHV3-3 / IGHJ4 | First approved VHH/Nanobody medicine, exact structural coverage, long native CDR3. |
| `ozoralizumab_tnf_vhh` | `ozoralizumab_tnf_framework_vhh` | Ozoralizumab TNF-binding domain | Approved in Japan, TNF/albumin Nanobody | alpaca IGHV3-74 / IGHJ5 | Approved VHH therapeutic with exact structural coverage and short native CDR3. The YAML uses the TNF-binding domain, not the albumin-binding domain. |
| `vobarilizumab_il6r_vhh` | `vobarilizumab_il6r_framework_vhh` | Vobarilizumab IL-6R-binding domain | Phase II clinical Nanobody with exact structure | alpaca IGHV3S53 / IGHJ5 | Clinically exposed, structurally covered IL-6R domain with a distinct inferred V gene. Included despite discontinuation because it is useful framework diversity. |

## Coordinate Files

Processed framework-only mmCIF templates are vendored beside the active YAMLs.
The original RCSB downloads are retained under `reference_structures/` with the
same filenames. The processed templates use one synthetic chain whose sequence
is the YAML VHH sequence with `{cdr1}`, `{cdr2}`, and `{cdr3}` removed. Antigen
chains, additional VHH domains, waters, ligands, tags, and CDR atom rows are
removed.

When multiple candidate structures were available, the selected reference was
the best-resolution entry that matched the YAML framework sequence, with
coordinate completeness considered for retained framework residues.

| Framework | Processed template and RCSB reference |
| --- | --- |
| `caplacizumab_vhh` | `caplacizumab_vhh_7EOW.cif` |
| `ozoralizumab_tnf_vhh` | `ozoralizumab_tnf_vhh_8Z8M.cif` |
| `vobarilizumab_il6r_vhh` | `vobarilizumab_il6r_vhh_7XL0.cif` |

Additional coordinate-backed clinical backfill reference structures are
vendored under `reference_structures/` for future curation, but are not registry
frameworks until their YAML templates have been sequence/CDR-validated:

| Candidate | Vendored coordinate |
| --- | --- |
| `envafolimab_vhh` | `reference_structures/envafolimab_vhh_5JDS.cif` |
| `gefurulimab_c5_vhh` | `reference_structures/gefurulimab_c5_vhh_8COH.cif` |
| `ciltacel_bcma_nb1_vhh` | `reference_structures/ciltacel_bcma_nb1_vhh_8HXQ.cif` |
| `ciltacel_bcma_nb2_vhh` | `reference_structures/ciltacel_bcma_nb2_vhh_8HXR.cif` |

## YAML Shape

The VHH YAMLs intentionally differ from the scFv framework records:

```yaml
id: caplacizumab_vhh
canonical_name: caplacizumab_framework_vhh
modality: vhh
format: template
template: "FR1{cdr1}FR2{cdr2}FR3{cdr3}FR4"
cdr_lengths:
  cdr1: {min: 7, max: 9}
  cdr2: {min: 7, max: 9}
  cdr3: {min: 18, max: 24}
```

There is no linker and no light chain. The `template` field represents one VHH
domain only. The `cdr_lengths` windows are deterministic around the observed
source CDR lengths:

- CDR1 and CDR2: observed length +/- 1, with a lower bound of 5.
- CDR3: observed length +/- 3, with a lower bound of 5.

## Multispecific Records

Several clinical VHH products are multispecific and include an albumin-binding
VHH for half-life extension. The core records select the disease-targeting
domain from each source product:

- `ozoralizumab_tnf_vhh` uses the TNF-binding domain, not the albumin binder.
- `vobarilizumab_il6r_vhh` uses the IL-6R-binding domain, not the albumin binder.

Albumin-binding VHH frameworks may be useful later as optional half-life
extension scaffolds, but they should not dominate the core panel because the
same or related albumin-binding domains recur across multiple programs.

## Exclusions For The Core Panel

The following categories were intentionally left out of the core VHH set:

- VL-only single-domain antibodies such as lulizumab and placulumab.
- Conventional VH/VL antibodies with camelid-origin heavy chains.
- Veterinary-only VHHs.
- Records with unclear sequence provenance or TBC status.
- CAR-T VHH YAMLs until the individual VHH domains are curated and
  sequence/CDR-validated separately from patent, regulatory, or coordinate
  sequence listings.

Sequence-only VHH records without exact public coordinates are kept under
`legacy/` for provenance. They are not included in `--frameworks all`.

## Runtime Contract

VHH records are loaded through the VHH registry, separate from the scFv registry,
because VHH templates use `cdr1`, `cdr2`, and `cdr3` placeholders instead of
the scFv `hcdr*` and `lcdr*` placeholder set. During design, the sampler
generates one mutable run per VHH CDR in template order and keeps all framework
residues fixed.

Final CSV exports map the three designed VHH CDR sequences to heavy-chain
columns:

```text
hcdr1, hcdr2, hcdr3
```

VHH exports intentionally do not include light-chain CDR columns.
