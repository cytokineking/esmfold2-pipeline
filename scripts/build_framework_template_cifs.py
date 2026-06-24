#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from biotite.structure.io.pdbx import CIFFile


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = ROOT / "src" / "esmfold2_pipeline" / "frameworks"
SCFV_LINKER = "GGGSGGGSGGGSGGGS"
PLACEHOLDER_RE = re.compile(r"\{([^}]+)\}")

AA1_TO_3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}
AA3_TO_1 = {value: key for key, value in AA1_TO_3.items()}


@dataclass(frozen=True)
class TemplateResidue:
    aa: str
    new_seq_id: int
    source_chain: str | None
    source_chain_column: str | None
    source_seq_id: int | None


@dataclass(frozen=True)
class SourceMatch:
    chain_id: str
    chain_column: str
    match: re.Match[str]
    retained: tuple[tuple[int, str], ...]
    missing_source_seq_ids: tuple[int, ...]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build framework-only template mmCIF files from bundled YAMLs."
    )
    parser.add_argument(
        "--framework-root",
        type=Path,
        default=FRAMEWORK_ROOT,
        help="Path containing scfv/ and vhh/ framework folders.",
    )
    args = parser.parse_args()

    for modality in ("scfv", "vhh"):
        folder = args.framework_root / modality
        for yaml_path in sorted(folder.glob("*.yaml")):
            record = yaml.safe_load(yaml_path.read_text())
            reference_path = _single_reference_structure(folder, record["id"])
            output_path = folder / reference_path.name
            print(f"{record['id']}: {reference_path.name} -> {output_path.name}")
            build_template_cif(record, reference_path, output_path)


def build_template_cif(
    record: dict[str, Any],
    reference_path: Path,
    output_path: Path,
) -> None:
    cif = CIFFile.read(str(reference_path))
    block_name = next(iter(cif.keys()))
    block = cif[block_name]
    chain_sequences = _chain_sequences(block)
    atom_site = _atom_site_arrays(block)
    cdr_lengths = _parse_cdr_lengths(record["cdr_lengths"])

    if record["modality"] == "scfv":
        template_residues = _scfv_template_residues(
            record["template"],
            cdr_lengths,
            chain_sequences,
            atom_site,
        )
    elif record["modality"] == "vhh":
        template_residues = _single_domain_template_residues(
            record["template"],
            cdr_lengths,
            chain_sequences,
            atom_site,
        )
    else:
        raise ValueError(f"{record['id']}: unsupported modality {record['modality']!r}")

    _write_template_cif(
        output_path=output_path,
        record=record,
        source_pdb=block_name.upper(),
        atom_site=atom_site,
        template_residues=template_residues,
    )


def _single_reference_structure(folder: Path, framework_id: str) -> Path:
    matches = sorted((folder / "reference_structures").glob(f"{framework_id}_*.cif"))
    if len(matches) != 1:
        raise ValueError(
            f"{framework_id}: expected exactly one reference CIF, found {len(matches)}"
        )
    return matches[0]


def _scfv_template_residues(
    template: str,
    cdr_lengths: dict[str, tuple[int, int]],
    chain_sequences: dict[str, str],
    atom_site: dict[str, list[str]],
) -> list[TemplateResidue]:
    parts = template.split(SCFV_LINKER)
    if len(parts) != 2:
        raise ValueError("scFv template must contain exactly one canonical linker")
    vh_template, vl_template = parts
    vh_match = _find_source_match(vh_template, cdr_lengths, chain_sequences, atom_site)
    vl_match = _find_source_match(vl_template, cdr_lengths, chain_sequences, atom_site)

    residues: list[TemplateResidue] = []
    residues.extend(
        _template_residues_from_match(
            vh_template,
            cdr_lengths,
            chain_sequences[vh_match.chain_id],
            vh_match,
            start_seq_id=1,
        )
    )
    for aa in SCFV_LINKER:
        residues.append(
            TemplateResidue(
                aa=aa,
                new_seq_id=len(residues) + 1,
                source_chain=None,
                source_chain_column=None,
                source_seq_id=None,
            )
        )
    residues.extend(
        _template_residues_from_match(
            vl_template,
            cdr_lengths,
            chain_sequences[vl_match.chain_id],
            vl_match,
            start_seq_id=len(residues) + 1,
        )
    )
    return residues


def _single_domain_template_residues(
    template: str,
    cdr_lengths: dict[str, tuple[int, int]],
    chain_sequences: dict[str, str],
    atom_site: dict[str, list[str]],
) -> list[TemplateResidue]:
    match = _find_source_match(template, cdr_lengths, chain_sequences, atom_site)
    return _template_residues_from_match(
        template,
        cdr_lengths,
        chain_sequences[match.chain_id],
        match,
        start_seq_id=1,
    )


def _find_source_match(
    template: str,
    cdr_lengths: dict[str, tuple[int, int]],
    chain_sequences: dict[str, str],
    atom_site: dict[str, list[str]],
) -> SourceMatch:
    pattern = _template_pattern(template, cdr_lengths)
    failures: list[str] = []
    best: SourceMatch | None = None
    for chain_id, sequence in chain_sequences.items():
        match = pattern.search(sequence)
        if match is None:
            continue
        retained = tuple(_retained_source_residues(template, sequence, match))
        for chain_column in ("auth_asym_id", "label_asym_id"):
            missing = [
                source_seq_id
                for source_seq_id, _aa in retained
                if not _atom_rows_for_residue(
                    atom_site,
                    chain_column=chain_column,
                    chain_id=chain_id,
                    source_seq_id=source_seq_id,
                )
            ]
            candidate = SourceMatch(
                chain_id=chain_id,
                chain_column=chain_column,
                match=match,
                retained=retained,
                missing_source_seq_ids=tuple(missing),
            )
            if not missing:
                return candidate
            if best is None or len(missing) < len(best.missing_source_seq_ids):
                best = candidate
            failures.append(
                f"{chain_id}/{chain_column}: missing atom rows for {missing[:5]}"
            )
    if best is not None:
        print(
            "  warning: "
            f"{best.chain_id}/{best.chain_column} lacks atom rows for "
            f"{len(best.missing_source_seq_ids)} retained framework residues"
        )
        return best
    details = "; ".join(failures) if failures else "no sequence match"
    raise ValueError(f"could not map template to reference chain: {details}")


def _template_residues_from_match(
    template: str,
    cdr_lengths: dict[str, tuple[int, int]],
    source_sequence: str,
    source_match: SourceMatch,
    *,
    start_seq_id: int,
) -> list[TemplateResidue]:
    del cdr_lengths
    residues = []
    new_seq_id = start_seq_id
    for source_seq_id, aa in _retained_source_residues(
        template,
        source_sequence,
        source_match.match,
    ):
        residues.append(
            TemplateResidue(
                aa=aa,
                new_seq_id=new_seq_id,
                source_chain=source_match.chain_id,
                source_chain_column=source_match.chain_column,
                source_seq_id=source_seq_id,
            )
        )
        new_seq_id += 1
    return residues


def _template_pattern(
    template: str,
    cdr_lengths: dict[str, tuple[int, int]],
) -> re.Pattern[str]:
    regex = []
    for token_type, value in _template_tokens(template):
        if token_type == "text":
            regex.append(re.escape(value))
        else:
            low, high = cdr_lengths[value]
            regex.append(f"(?P<{value}>[A-Z]{{{low},{high}}})")
    return re.compile("".join(regex))


def _retained_source_residues(
    template: str,
    source_sequence: str,
    match: re.Match[str],
) -> list[tuple[int, str]]:
    retained: list[tuple[int, str]] = []
    source_index = match.start()
    for token_type, value in _template_tokens(template):
        if token_type == "text":
            observed = source_sequence[source_index : source_index + len(value)]
            if observed != value:
                raise ValueError(f"template fragment mismatch: {value!r} != {observed!r}")
            for offset, aa in enumerate(value):
                retained.append((source_index + offset + 1, aa))
            source_index += len(value)
        else:
            source_index = match.end(value)
    if source_index != match.end():
        raise ValueError("template token scan did not consume regex match")
    return retained


def _template_tokens(template: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    cursor = 0
    for match in PLACEHOLDER_RE.finditer(template):
        if match.start() > cursor:
            tokens.append(("text", template[cursor : match.start()]))
        tokens.append(("cdr", match.group(1)))
        cursor = match.end()
    if cursor < len(template):
        tokens.append(("text", template[cursor:]))
    return tokens


def _parse_cdr_lengths(value: dict[str, Any]) -> dict[str, tuple[int, int]]:
    parsed = {}
    for name, raw in value.items():
        if isinstance(raw, int):
            parsed[name] = (raw, raw)
        else:
            parsed[name] = (raw["min"], raw["max"])
    return parsed


def _chain_sequences(block: Any) -> dict[str, str]:
    category = block["entity_poly"]
    sequences = _column(category, "pdbx_seq_one_letter_code_can")
    chain_groups = _column(category, "pdbx_strand_id")
    chain_sequences: dict[str, str] = {}
    for sequence, chain_group in zip(sequences, chain_groups, strict=True):
        clean_sequence = sequence.replace("\n", "").replace(" ", "")
        for chain_id in chain_group.split(","):
            chain_id = chain_id.strip()
            if chain_id:
                chain_sequences[chain_id] = clean_sequence
    return chain_sequences


def _atom_site_arrays(block: Any) -> dict[str, list[str]]:
    atom_site = block["atom_site"]
    return {key: [str(value) for value in atom_site[key].as_array()] for key in atom_site}


def _column(category: Any, name: str) -> list[str]:
    return [str(value) for value in category[name].as_array()]


def _atom_rows_for_residue(
    atom_site: dict[str, list[str]],
    *,
    chain_column: str,
    chain_id: str,
    source_seq_id: int,
) -> list[int]:
    rows = []
    for index, (group, model, row_chain, seq_id) in enumerate(
        zip(
            atom_site["group_PDB"],
            atom_site.get("pdbx_PDB_model_num", ["1"] * len(atom_site["group_PDB"])),
            atom_site[chain_column],
            atom_site["label_seq_id"],
            strict=True,
        )
    ):
        if group != "ATOM" or model != "1" or row_chain != chain_id:
            continue
        if seq_id in {".", "?"}:
            continue
        if int(seq_id) == source_seq_id:
            rows.append(index)
    return rows


def _write_template_cif(
    *,
    output_path: Path,
    record: dict[str, Any],
    source_pdb: str,
    atom_site: dict[str, list[str]],
    template_residues: list[TemplateResidue],
) -> None:
    sequence = "".join(residue.aa for residue in template_residues)
    data_id = f"{record['id']}_framework_template"
    title = f"Framework-only template for {record['id']} derived from RCSB {source_pdb}"
    lines = [
        f"data_{data_id}",
        "#",
        f"_entry.id {data_id}",
        f"_struct.title {_quote(title)}",
        "#",
        "_entity.id 1",
        "_entity.type polymer",
        f"_entity.pdbx_description {_quote(record['canonical_name'])}",
        "#",
        "_entity_poly.entity_id 1",
        "_entity_poly.type 'polypeptide(L)'",
        "_entity_poly.nstd_linkage no",
        "_entity_poly.nstd_monomer no",
        "_entity_poly.pdbx_seq_one_letter_code",
        _multiline(sequence),
        "_entity_poly.pdbx_seq_one_letter_code_can",
        _multiline(sequence),
        "_entity_poly.pdbx_strand_id A",
        "_entity_poly.pdbx_target_identifier ?",
        "#",
        "loop_",
        "_entity_poly_seq.entity_id",
        "_entity_poly_seq.num",
        "_entity_poly_seq.mon_id",
        "_entity_poly_seq.hetero",
    ]
    for residue in template_residues:
        lines.append(
            f"1 {residue.new_seq_id} {AA1_TO_3[residue.aa]} n"
        )
    lines.extend(
        [
            "#",
            "_struct_asym.id A",
            "_struct_asym.entity_id 1",
            "_struct_asym.details ?",
            "#",
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.pdbx_formal_charge",
            "_atom_site.auth_seq_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_atom_id",
            "_atom_site.pdbx_PDB_model_num",
        ]
    )
    atom_id = 1
    for residue in template_residues:
        if residue.source_chain is None:
            continue
        rows = _atom_rows_for_residue(
            atom_site,
            chain_column=residue.source_chain_column or "auth_asym_id",
            chain_id=residue.source_chain,
            source_seq_id=residue.source_seq_id or 0,
        )
        if not rows:
            continue
        expected_comp = AA1_TO_3[residue.aa]
        for row in rows:
            observed_comp = atom_site["label_comp_id"][row]
            if AA3_TO_1.get(observed_comp) != residue.aa:
                raise ValueError(
                    f"{record['id']}: source residue {residue.source_chain}:"
                    f"{residue.source_seq_id} is {observed_comp}, expected {expected_comp}"
                )
            lines.append(
                " ".join(
                    [
                        "ATOM",
                        str(atom_id),
                        _cif_atom(atom_site["type_symbol"][row]),
                        _cif_atom(atom_site["label_atom_id"][row]),
                        _cif_atom(atom_site.get("label_alt_id", ["."])[row]),
                        expected_comp,
                        "A",
                        "1",
                        str(residue.new_seq_id),
                        "?",
                        atom_site["Cartn_x"][row],
                        atom_site["Cartn_y"][row],
                        atom_site["Cartn_z"][row],
                        atom_site.get("occupancy", ["1.00"])[row],
                        atom_site.get("B_iso_or_equiv", ["0.00"])[row],
                        "?",
                        str(residue.new_seq_id),
                        expected_comp,
                        "A",
                        _cif_atom(atom_site.get("auth_atom_id", atom_site["label_atom_id"])[row]),
                        "1",
                    ]
                )
            )
            atom_id += 1
    lines.extend(["#", ""])
    output_path.write_text("\n".join(lines))


def _multiline(sequence: str) -> str:
    wrapped = "\n".join(sequence[index : index + 80] for index in range(0, len(sequence), 80))
    return f";{wrapped}\n;"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _cif_atom(value: str) -> str:
    if value in {".", "?"}:
        return value
    if re.fullmatch(r"[A-Za-z0-9_+-]+", value):
        return value
    return _quote(value)


if __name__ == "__main__":
    main()
