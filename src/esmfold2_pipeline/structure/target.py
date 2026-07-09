from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
import struct
from typing import Iterable

import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx

from esmfold2_pipeline.artifacts import write_bytes_atomic, write_json_atomic, write_text_atomic


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA1_TO_3 = {value: key for key, value in AA3_TO_1.items()}
MODIFIED_AA3_TO_STANDARD = {
    "MSE": "MET",
}

ATOM37_ORDER = (
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)

MMCIF_EXTRA_FIELDS = [
    "label_seq_id",
    "auth_seq_id",
    "label_asym_id",
    "label_entity_id",
    "auth_asym_id",
    "pdbx_PDB_ins_code",
    "auth_comp_id",
    "label_comp_id",
    "auth_atom_id",
    "label_atom_id",
]


class StructureTargetError(ValueError):
    """Raised when a structure-derived target cannot be represented safely."""


@dataclass(frozen=True)
class StructureTargetConfig:
    path: Path
    chains: tuple[str, ...]
    structure_indexing: str = "auto"
    sequences: dict[str, str] | None = None
    crop: dict[str, tuple[str, ...]] | None = None
    hotspots: dict[str, tuple[str, ...]] | None = None
    conditioning_mode: str = "none"
    conditioning_assembly: bool = False
    conditioning_assembly_auto: bool = False
    conditioning_chain_pairs: tuple[tuple[str, str], ...] | None = None
    partial_conditioning: bool = True
    representative_atom: str = "esmfold2_default"
    require_resolved: bool = False


@dataclass(frozen=True)
class AtomRecord:
    atom_name: str
    res_name: str
    auth_asym_id: str
    auth_seq_id: str
    pdbx_pdb_ins_code: str
    label_asym_id: str
    label_entity_id: str
    label_seq_id: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class TargetResidue:
    input_chain: str
    input_residue: str
    input_insertion_code: str
    auth_asym_id: str
    auth_seq_id: str
    pdbx_pdb_ins_code: str
    label_asym_id: str
    label_seq_id: str
    canonical_chain_id: str
    model_residue_index_0: int
    res_name: str
    sequence_1letter: str
    atoms: dict[str, tuple[float, float, float]]
    atom_order: tuple[str, ...]
    representative_atom: str | None
    representative_coord: tuple[float, float, float] | None
    is_observed: bool
    sequence_source: str


@dataclass(frozen=True)
class TargetChain:
    canonical_chain_id: str
    auth_asym_id: str
    label_asym_id: str
    sequence: str
    residues: tuple[TargetResidue, ...]
    hotspot_indices: tuple[int, ...]
    distogram: tuple[tuple[float, ...], ...]
    distogram_mask: tuple[tuple[bool, ...], ...]
    representative_coord_mask: tuple[bool, ...]
    sequence_source: str


@dataclass(frozen=True)
class PreparedTarget:
    source_path: Path
    input_format: str
    chains: tuple[TargetChain, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetArtifactResult:
    target_dir: Path
    normalized_target: Path
    residue_map_csv: Path
    chain_summary_json: Path
    conditioning_files: tuple[Path, ...]
    assembly_conditioning_json: Path | None = None


def parse_structure_target(config: StructureTargetConfig) -> PreparedTarget:
    atoms, input_format = _read_structure_atoms(config.path)
    source_chains = _build_source_chains(atoms)
    selected = _select_source_chains(source_chains, config.chains)
    sequence_metadata = _read_sequence_metadata(config.path, input_format)

    canonical_counts: dict[str, int] = {}
    target_chains: list[TargetChain] = []
    warnings: list[str] = []
    for source_chain in selected:
        canonical_chain_id = _canonical_chain_id(
            source_chain.input_id,
            canonical_counts,
        )
        full_residues, register_warnings = _full_register_residues_for_chain(
            source_chain,
            canonical_chain_id=canonical_chain_id,
            config=config,
            input_format=input_format,
            sequence_metadata=sequence_metadata,
        )
        warnings.extend(register_warnings)
        residues = _crop_residues(
            full_residues,
            canonical_chain_id=canonical_chain_id,
            config=config,
            selected_chain_count=len(selected),
        )
        if not residues:
            raise StructureTargetError(
                f"target chain {source_chain.input_id} has no residues after crop"
            )

        rewritten = _rewrite_residues_for_chain(residues, canonical_chain_id)
        hotspot_indices = _resolve_hotspot_indices(
            rewritten,
            canonical_chain_id=canonical_chain_id,
            auth_asym_id=source_chain.auth_asym_id,
            label_asym_id=source_chain.label_asym_id,
            config=config,
            selected_chain_count=len(selected),
        )
        _validate_hotspots_resolved(rewritten, hotspot_indices, canonical_chain_id)
        distogram, distogram_mask = _compute_distogram(rewritten)
        unresolved = [
            residue
            for residue in rewritten
            if residue.representative_coord is None
        ]
        if unresolved and config.require_resolved:
            formatted = ", ".join(_format_residue_id(residue) for residue in unresolved)
            raise StructureTargetError(
                f"target chain {canonical_chain_id} has unresolved representative "
                f"coordinates: {formatted}"
            )
        if unresolved:
            warnings.append(
                f"target chain {canonical_chain_id} has {len(unresolved)} unresolved "
                "representative coordinates"
            )
            if config.conditioning_mode == "distogram" and not config.partial_conditioning:
                formatted = ", ".join(_format_residue_id(residue) for residue in unresolved)
                raise StructureTargetError(
                    f"target chain {canonical_chain_id} has unresolved representative "
                    "coordinates but target.conditioning.partial is false: "
                    f"{formatted}"
                )

        target_chains.append(
            TargetChain(
                canonical_chain_id=canonical_chain_id,
                auth_asym_id=source_chain.auth_asym_id,
                label_asym_id=source_chain.label_asym_id,
                sequence="".join(residue.sequence_1letter for residue in rewritten),
                residues=tuple(rewritten),
                hotspot_indices=tuple(sorted(hotspot_indices)),
                distogram=distogram,
                distogram_mask=distogram_mask,
                representative_coord_mask=tuple(
                    residue.representative_coord is not None for residue in rewritten
                ),
                sequence_source=_chain_sequence_source(rewritten),
            )
        )

    return PreparedTarget(
        source_path=config.path,
        input_format=input_format,
        chains=tuple(target_chains),
        warnings=tuple(warnings),
    )


def write_target_artifacts(
    prepared: PreparedTarget,
    target_dir: str | Path,
    *,
    conditioning_mode: str = "none",
    conditioning_assembly: bool = False,
    conditioning_chain_pairs: tuple[tuple[str, str], ...] | None = None,
    partial_conditioning: bool = True,
    representative_atom: str = "esmfold2_default",
    require_resolved: bool = False,
) -> TargetArtifactResult:
    root = Path(target_dir)
    conditioning_dir = root / "conditioning"
    normalized_target = _write_normalized_cif(prepared, root / "normalized_target.cif")
    residue_map_csv = _write_residue_map(prepared, root / "residue_map.csv")

    conditioning_files: list[Path] = []
    assembly_conditioning_json: Path | None = None
    if conditioning_mode == "distogram":
        for chain in prepared.chains:
            prefix = conditioning_dir / f"chain_{_safe_name(chain.canonical_chain_id)}"
            coords = [
                residue.representative_coord or (0.0, 0.0, 0.0)
                for residue in chain.residues
            ]
            if (
                not partial_conditioning
                and len(coords) != sum(chain.representative_coord_mask)
            ):
                raise StructureTargetError(
                    f"target chain {chain.canonical_chain_id} has unresolved "
                    "representative coordinates"
                )
            coords_path = _write_float32_npy(
                prefix.with_name(f"{prefix.name}_rep_coords.npy"),
                coords,
                (len(coords), 3),
            )
            distogram_path = _write_float32_npy(
                prefix.with_name(f"{prefix.name}_distogram.npy"),
                chain.distogram,
                (len(chain.residues), len(chain.residues)),
            )
            coord_mask_path = _write_bool_npy(
                prefix.with_name(f"{prefix.name}_rep_coord_mask.npy"),
                chain.representative_coord_mask,
                (len(chain.residues),),
            )
            distogram_mask_path = _write_bool_npy(
                prefix.with_name(f"{prefix.name}_distogram_mask.npy"),
                chain.distogram_mask,
                (len(chain.residues), len(chain.residues)),
            )
            conditioning_files.extend(
                [coords_path, coord_mask_path, distogram_path, distogram_mask_path]
            )
        if conditioning_assembly:
            pair_files, assembly_conditioning_json = _write_assembly_conditioning(
                prepared,
                conditioning_dir=conditioning_dir,
                target_dir=root,
                conditioning_chain_pairs=conditioning_chain_pairs,
                partial_conditioning=partial_conditioning,
                representative_atom=representative_atom,
                require_resolved=require_resolved,
            )
            conditioning_files.extend(pair_files)
    elif conditioning_assembly:
        raise StructureTargetError(
            "target.conditioning.assembly requires target.conditioning.mode: distogram"
        )

    chain_summary_json = write_json_atomic(
        root / "chain_summary.json",
        _chain_summary(
            prepared,
            conditioning_mode=conditioning_mode,
            conditioning_assembly=conditioning_assembly,
            conditioning_chain_pairs=conditioning_chain_pairs,
        ),
    )
    return TargetArtifactResult(
        target_dir=root,
        normalized_target=normalized_target,
        residue_map_csv=residue_map_csv,
        chain_summary_json=chain_summary_json,
        conditioning_files=tuple(conditioning_files),
        assembly_conditioning_json=assembly_conditioning_json,
    )


def resolve_target_geometry_drift_indices(
    prepared: PreparedTarget,
    regions: dict[str, tuple[str, ...]] | None,
    *,
    structure_indexing: str = "auto",
    field_name: str = "loss.target_geometry_drift.regions",
) -> tuple[int, ...]:
    selected: set[int] = set()
    if regions is None or not regions:
        selected.update(
            index
            for index, _residue in _iter_flat_target_residues(prepared)
        )
    else:
        _validate_drift_region_chain_keys(prepared, regions, field_name=field_name)

        for chain_start, chain in _iter_target_chains_with_start(prepared):
            selectors = _drift_selectors_for_chain(chain, regions)
            if not selectors:
                continue
            if any(selector.strip().lower() == "all" for selector in selectors):
                selected.update(range(chain_start, chain_start + len(chain.residues)))
                continue
            residue_builders = _residue_builders_from_target_residues(chain.residues)
            local_indices = _resolve_selectors(
                residue_builders,
                selectors,
                indexing=structure_indexing,
                field_name=f"{field_name}[{chain.canonical_chain_id}]",
            )
            selected.update(chain_start + index for index in local_indices)

    if not selected:
        raise StructureTargetError(f"{field_name} selected no target residues")
    resolved_selected = {
        index
        for index, residue in _iter_flat_target_residues(prepared)
        if index in selected and residue.representative_coord is not None
    }
    if len(resolved_selected) < 2:
        raise StructureTargetError(
            f"{field_name} must select at least two resolved residues"
        )
    return tuple(sorted(resolved_selected))


@dataclass(frozen=True)
class _ResidueBuilder:
    auth_asym_id: str
    label_asym_id: str
    label_entity_id: str
    auth_seq_id: str
    label_seq_id: str
    pdbx_pdb_ins_code: str
    res_name: str
    atoms: dict[str, tuple[float, float, float]]
    atom_order: tuple[str, ...]
    is_observed: bool = True
    sequence_source: str = "observed_atom"
    sequence_1letter: str = ""


@dataclass(frozen=True)
class _SourceChain:
    input_id: str
    auth_asym_id: str
    label_asym_id: str
    label_entity_id: str
    residues: tuple[_ResidueBuilder, ...]


@dataclass(frozen=True)
class _SequenceMetadata:
    mmcif_scheme_by_label_asym: dict[str, tuple[_ResidueBuilder, ...]]
    mmcif_entity_res_names: dict[str, tuple[str, ...]]
    pdb_seqres: dict[str, tuple[str, ...]]


def _read_structure_atoms(path: Path) -> tuple[tuple[AtomRecord, ...], str]:
    if not path.exists():
        raise StructureTargetError(f"target structure does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        atoms = _read_mmcif_atoms(path)
        input_format = "mmcif"
    elif suffix == ".pdb":
        atoms = _read_pdb_atoms(path)
        input_format = "pdb"
    else:
        raise StructureTargetError(
            f"target.structure must be a .pdb, .cif, or .mmcif file: {path}"
        )
    if not atoms:
        raise StructureTargetError(f"target structure contains no protein ATOM records: {path}")
    return atoms, input_format


def _read_pdb_atoms(path: Path) -> tuple[AtomRecord, ...]:
    pdb_file = pdb.PDBFile.read(path)
    atom_array = pdb.get_structure(pdb_file, model=1, altloc="first")
    atoms: list[AtomRecord] = []
    for index in range(atom_array.array_length()):
        if bool(atom_array.hetero[index]):
            continue
        res_name = _normalize_res_name(str(atom_array.res_name[index]).upper())
        if res_name not in AA3_TO_1:
            continue
        chain_id = _clean_missing(str(atom_array.chain_id[index])) or "_"
        auth_seq_id = str(int(atom_array.res_id[index]))
        ins_code = _clean_missing(str(atom_array.ins_code[index]))
        atom_name = str(atom_array.atom_name[index]).strip()
        x, y, z = atom_array.coord[index]
        atoms.append(
            AtomRecord(
                atom_name=atom_name,
                res_name=res_name,
                auth_asym_id=chain_id,
                auth_seq_id=auth_seq_id,
                pdbx_pdb_ins_code=ins_code,
                label_asym_id=chain_id,
                label_entity_id=chain_id,
                label_seq_id=auth_seq_id,
                x=float(x),
                y=float(y),
                z=float(z),
            )
        )
    return tuple(atoms)


def _read_mmcif_atoms(path: Path) -> tuple[AtomRecord, ...]:
    cif_file = pdbx.CIFFile.read(path)
    atom_array = pdbx.get_structure(
        cif_file,
        model=1,
        altloc="first",
        extra_fields=MMCIF_EXTRA_FIELDS,
        use_author_fields=True,
    )
    atoms: list[AtomRecord] = []
    for index in range(atom_array.array_length()):
        if bool(atom_array.hetero[index]):
            continue
        res_name = _normalize_res_name(_clean_missing(
            str(getattr(atom_array, "auth_comp_id", atom_array.res_name)[index])
        ).upper())
        if res_name not in AA3_TO_1:
            continue
        auth_asym_id = _clean_missing(str(atom_array.auth_asym_id[index])) or "_"
        label_asym_id = _clean_missing(str(atom_array.label_asym_id[index])) or auth_asym_id
        label_entity_values = getattr(atom_array, "label_entity_id", None)
        label_entity_id = (
            _clean_missing(str(label_entity_values[index]))
            if label_entity_values is not None
            else ""
        )
        auth_seq_id = _clean_missing(str(atom_array.auth_seq_id[index]))
        label_seq_id = _clean_missing(str(atom_array.label_seq_id[index])) or auth_seq_id
        ins_code = _clean_missing(str(atom_array.pdbx_PDB_ins_code[index]))
        atom_name = _clean_missing(str(atom_array.auth_atom_id[index])) or str(
            atom_array.atom_name[index]
        )
        x, y, z = atom_array.coord[index]
        atoms.append(
            AtomRecord(
                atom_name=atom_name,
                res_name=res_name,
                auth_asym_id=auth_asym_id,
                auth_seq_id=auth_seq_id,
                pdbx_pdb_ins_code=ins_code,
                label_asym_id=label_asym_id,
                label_entity_id=label_entity_id,
                label_seq_id=label_seq_id,
                x=float(x),
                y=float(y),
                z=float(z),
            )
        )
    return tuple(atoms)


def _build_source_chains(atoms: Iterable[AtomRecord]) -> tuple[_SourceChain, ...]:
    chain_order: list[tuple[str, str, str]] = []
    residue_order: dict[tuple[str, str, str], list[tuple[str, str, str, str]]] = {}
    residues: dict[tuple[str, str, str, str, str, str, str], _ResidueBuilder] = {}

    for atom in atoms:
        chain_key = (atom.auth_asym_id, atom.label_asym_id, atom.label_entity_id)
        if chain_key not in residue_order:
            chain_order.append(chain_key)
            residue_order[chain_key] = []
        residue_key = (
            atom.auth_seq_id,
            atom.pdbx_pdb_ins_code,
            atom.label_seq_id,
            atom.res_name,
        )
        full_key = (*chain_key, *residue_key)
        if full_key not in residues:
            residue_order[chain_key].append(residue_key)
            residues[full_key] = _ResidueBuilder(
                auth_asym_id=atom.auth_asym_id,
                label_asym_id=atom.label_asym_id,
                label_entity_id=atom.label_entity_id,
                auth_seq_id=atom.auth_seq_id,
                label_seq_id=atom.label_seq_id,
                pdbx_pdb_ins_code=atom.pdbx_pdb_ins_code,
                res_name=atom.res_name,
                atoms={},
                atom_order=(),
                sequence_1letter=AA3_TO_1[atom.res_name],
            )
        builder = residues[full_key]
        if atom.atom_name in builder.atoms:
            continue
        atoms_by_name = dict(builder.atoms)
        atoms_by_name[atom.atom_name] = (atom.x, atom.y, atom.z)
        residues[full_key] = _ResidueBuilder(
            auth_asym_id=builder.auth_asym_id,
            label_asym_id=builder.label_asym_id,
            label_entity_id=builder.label_entity_id,
            auth_seq_id=builder.auth_seq_id,
            label_seq_id=builder.label_seq_id,
            pdbx_pdb_ins_code=builder.pdbx_pdb_ins_code,
            res_name=builder.res_name,
            atoms=atoms_by_name,
            atom_order=(*builder.atom_order, atom.atom_name),
            is_observed=builder.is_observed,
            sequence_source=builder.sequence_source,
            sequence_1letter=builder.sequence_1letter,
        )

    chains: list[_SourceChain] = []
    for chain_key in chain_order:
        chain_residues = tuple(
            residues[(*chain_key, *residue_key)]
            for residue_key in residue_order[chain_key]
        )
        auth_asym_id, label_asym_id, label_entity_id = chain_key
        chains.append(
            _SourceChain(
                input_id=auth_asym_id or label_asym_id,
                auth_asym_id=auth_asym_id,
                label_asym_id=label_asym_id,
                label_entity_id=label_entity_id,
                residues=chain_residues,
            )
        )
    return tuple(chains)


def _select_source_chains(
    source_chains: tuple[_SourceChain, ...],
    requested_chains: tuple[str, ...],
) -> tuple[_SourceChain, ...]:
    if not requested_chains:
        return source_chains

    selected: list[_SourceChain] = []
    for requested in requested_chains:
        matches = [
            chain
            for chain in source_chains
            if requested in {chain.input_id, chain.auth_asym_id, chain.label_asym_id}
        ]
        if not matches:
            available = ", ".join(
                f"{chain.input_id}(auth={chain.auth_asym_id},label={chain.label_asym_id})"
                for chain in source_chains
            )
            raise StructureTargetError(
                f"requested target chain {requested} was not found; available: {available}"
            )
        if len(matches) > 1:
            raise StructureTargetError(
                f"requested target chain {requested} is ambiguous; match by a unique "
                "auth_asym_id or label_asym_id"
            )
        selected.append(matches[0])
    return tuple(selected)


def _read_sequence_metadata(path: Path, input_format: str) -> _SequenceMetadata:
    if input_format == "mmcif":
        return _read_mmcif_sequence_metadata(path)
    if input_format == "pdb":
        return _SequenceMetadata(
            mmcif_scheme_by_label_asym={},
            mmcif_entity_res_names={},
            pdb_seqres=_read_pdb_seqres(path),
        )
    return _SequenceMetadata(
        mmcif_scheme_by_label_asym={},
        mmcif_entity_res_names={},
        pdb_seqres={},
    )


def _read_mmcif_sequence_metadata(path: Path) -> _SequenceMetadata:
    cif_file = pdbx.CIFFile.read(path)
    scheme_rows = _read_mmcif_poly_seq_scheme(cif_file)
    entity_res_names = _read_mmcif_entity_poly_seq(cif_file)
    return _SequenceMetadata(
        mmcif_scheme_by_label_asym=scheme_rows,
        mmcif_entity_res_names=entity_res_names,
        pdb_seqres={},
    )


def _read_mmcif_poly_seq_scheme(
    cif_file: pdbx.CIFFile,
) -> dict[str, tuple[_ResidueBuilder, ...]]:
    columns = _cif_category_columns(cif_file, "pdbx_poly_seq_scheme")
    if not columns:
        return {}
    row_count = _cif_row_count(columns)
    by_chain: dict[str, list[_ResidueBuilder]] = {}
    for index in range(row_count):
        label_asym_id = _cif_value(columns, "asym_id", index)
        if not label_asym_id:
            continue
        raw_res_name = (
            _cif_value(columns, "mon_id", index)
            or _cif_value(columns, "pdb_mon_id", index)
            or _cif_value(columns, "auth_mon_id", index)
        )
        res_name = _normalize_res_name(raw_res_name)
        if res_name not in AA3_TO_1:
            continue
        label_seq_id = (
            _cif_value(columns, "seq_id", index)
            or _cif_value(columns, "ndb_seq_num", index)
            or str(len(by_chain.get(label_asym_id, ())) + 1)
        )
        auth_seq_id = (
            _cif_value(columns, "auth_seq_num", index)
            or _cif_value(columns, "pdb_seq_num", index)
            or label_seq_id
        )
        auth_asym_id = (
            _cif_value(columns, "pdb_strand_id", index)
            or _cif_value(columns, "auth_asym_id", index)
            or label_asym_id
        )
        by_chain.setdefault(label_asym_id, []).append(
            _ResidueBuilder(
                auth_asym_id=auth_asym_id,
                label_asym_id=label_asym_id,
                label_entity_id=_cif_value(columns, "entity_id", index),
                auth_seq_id=auth_seq_id,
                label_seq_id=label_seq_id,
                pdbx_pdb_ins_code=_cif_value(columns, "pdb_ins_code", index),
                res_name=res_name,
                atoms={},
                atom_order=(),
                is_observed=False,
                sequence_source="mmcif_pdbx_poly_seq_scheme",
                sequence_1letter=AA3_TO_1[res_name],
            )
        )
    return {
        chain_id: tuple(_sort_by_numeric_label_seq_id(residues))
        for chain_id, residues in by_chain.items()
    }


def _read_mmcif_entity_poly_seq(cif_file: pdbx.CIFFile) -> dict[str, tuple[str, ...]]:
    columns = _cif_category_columns(cif_file, "entity_poly_seq")
    if not columns:
        return {}
    rows: dict[str, list[tuple[int, str]]] = {}
    row_count = _cif_row_count(columns)
    for index in range(row_count):
        entity_id = _cif_value(columns, "entity_id", index)
        res_name = _normalize_res_name(_cif_value(columns, "mon_id", index))
        if not entity_id or res_name not in AA3_TO_1:
            continue
        seq_num = _int_or_none(_cif_value(columns, "num", index)) or index + 1
        rows.setdefault(entity_id, []).append((seq_num, res_name))
    return {
        entity_id: tuple(res_name for _seq_num, res_name in sorted(items))
        for entity_id, items in rows.items()
    }


def _cif_category_columns(
    cif_file: pdbx.CIFFile,
    category_name: str,
) -> dict[str, list[str]]:
    category = cif_file.block.get(category_name)
    if category is None:
        return {}
    return {
        key: [_clean_missing(str(value)) for value in category[key].as_array(str)]
        for key in category.keys()
    }


def _cif_row_count(columns: dict[str, list[str]]) -> int:
    if not columns:
        return 0
    return len(next(iter(columns.values())))


def _cif_value(columns: dict[str, list[str]], key: str, index: int) -> str:
    values = columns.get(key)
    if values is None or index >= len(values):
        return ""
    return values[index]


def _read_pdb_seqres(path: Path) -> dict[str, tuple[str, ...]]:
    sequences: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("SEQRES"):
            continue
        chain_id = _clean_missing(line[11:12]) or "_"
        for raw_res_name in line[19:].split():
            res_name = _normalize_res_name(raw_res_name)
            if res_name in AA3_TO_1:
                sequences.setdefault(chain_id, []).append(res_name)
    return {chain_id: tuple(res_names) for chain_id, res_names in sequences.items()}


def _full_register_residues_for_chain(
    source_chain: _SourceChain,
    *,
    canonical_chain_id: str,
    config: StructureTargetConfig,
    input_format: str,
    sequence_metadata: _SequenceMetadata,
) -> tuple[tuple[_ResidueBuilder, ...], list[str]]:
    warnings: list[str] = []
    user_sequence = _user_sequence_for_chain(
        config.sequences,
        source_chain=source_chain,
        canonical_chain_id=canonical_chain_id,
    )
    metadata_residues = _metadata_residues_for_chain(
        source_chain,
        input_format=input_format,
        sequence_metadata=sequence_metadata,
    )
    if metadata_residues is not None:
        residues = _overlay_observed_residues(metadata_residues, source_chain.residues)
        if user_sequence is not None:
            metadata_sequence = "".join(residue.sequence_1letter for residue in residues)
            if metadata_sequence != user_sequence:
                raise StructureTargetError(
                    "target.structure.sequences for chain "
                    f"{canonical_chain_id} does not match sequence metadata in "
                    f"{config.path.name}"
                )
        return residues, warnings

    if user_sequence is not None:
        return (
            _build_register_from_full_sequence(
                source_chain,
                user_sequence,
                sequence_source="user_sequence",
                field_name=f"target.structure.sequences[{canonical_chain_id}]",
            ),
            warnings,
        )

    _validate_observed_residue_register_contiguous(source_chain, input_format=input_format)
    warnings.append(
        f"target chain {canonical_chain_id} used observed ATOM records as sequence "
        "because no full sequence metadata or user sequence was available"
    )
    return (
        tuple(
            _replace_residue_sequence_fields(
                residue,
                sequence_source="observed_atom",
                is_observed=True,
            )
            for residue in source_chain.residues
        ),
        warnings,
    )


def _metadata_residues_for_chain(
    source_chain: _SourceChain,
    *,
    input_format: str,
    sequence_metadata: _SequenceMetadata,
) -> tuple[_ResidueBuilder, ...] | None:
    if input_format == "mmcif":
        scheme_residues = sequence_metadata.mmcif_scheme_by_label_asym.get(
            source_chain.label_asym_id
        )
        if scheme_residues:
            return scheme_residues
        entity_res_names = sequence_metadata.mmcif_entity_res_names.get(
            source_chain.label_entity_id
        )
        if entity_res_names:
            return _build_mmcif_entity_register(source_chain, entity_res_names)
    if input_format == "pdb":
        seqres = _sequence_for_source_chain(sequence_metadata.pdb_seqres, source_chain)
        if seqres:
            sequence = "".join(AA3_TO_1[res_name] for res_name in seqres)
            return _build_register_from_full_sequence(
                source_chain,
                sequence,
                sequence_source="pdb_seqres",
                field_name=f"SEQRES[{source_chain.input_id}]",
            )
    return None


def _build_mmcif_entity_register(
    source_chain: _SourceChain,
    entity_res_names: tuple[str, ...],
) -> tuple[_ResidueBuilder, ...]:
    return tuple(
        _ResidueBuilder(
            auth_asym_id=source_chain.auth_asym_id,
            label_asym_id=source_chain.label_asym_id,
            label_entity_id=source_chain.label_entity_id,
            auth_seq_id=str(index),
            label_seq_id=str(index),
            pdbx_pdb_ins_code="",
            res_name=res_name,
            atoms={},
            atom_order=(),
            is_observed=False,
            sequence_source="mmcif_entity_poly_seq",
            sequence_1letter=AA3_TO_1[res_name],
        )
        for index, res_name in enumerate(entity_res_names, start=1)
    )


def _overlay_observed_residues(
    template_residues: tuple[_ResidueBuilder, ...],
    observed_residues: tuple[_ResidueBuilder, ...],
) -> tuple[_ResidueBuilder, ...]:
    observed_by_label = {
        (residue.label_asym_id, residue.label_seq_id): residue
        for residue in observed_residues
        if residue.label_seq_id
    }
    observed_by_auth = {
        (
            residue.auth_asym_id,
            residue.auth_seq_id,
            residue.pdbx_pdb_ins_code,
        ): residue
        for residue in observed_residues
        if residue.auth_seq_id
    }
    overlaid: list[_ResidueBuilder] = []
    for template in template_residues:
        observed = observed_by_label.get((template.label_asym_id, template.label_seq_id))
        if observed is None:
            observed = observed_by_auth.get(
                (
                    template.auth_asym_id,
                    template.auth_seq_id,
                    template.pdbx_pdb_ins_code,
                )
            )
        if observed is None:
            overlaid.append(template)
            continue
        observed_letter = observed.sequence_1letter or AA3_TO_1[observed.res_name]
        if observed_letter != template.sequence_1letter:
            raise StructureTargetError(
                "observed residue identity does not match sequence metadata: "
                f"{observed.auth_asym_id}:{observed.auth_seq_id}"
                f"{observed.pdbx_pdb_ins_code} observed {observed.res_name}, "
                f"metadata {template.res_name}"
            )
        overlaid.append(
            _ResidueBuilder(
                auth_asym_id=observed.auth_asym_id or template.auth_asym_id,
                label_asym_id=template.label_asym_id,
                label_entity_id=template.label_entity_id or observed.label_entity_id,
                auth_seq_id=observed.auth_seq_id or template.auth_seq_id,
                label_seq_id=template.label_seq_id,
                pdbx_pdb_ins_code=observed.pdbx_pdb_ins_code
                or template.pdbx_pdb_ins_code,
                res_name=template.res_name,
                atoms=observed.atoms,
                atom_order=observed.atom_order,
                is_observed=True,
                sequence_source=template.sequence_source,
                sequence_1letter=template.sequence_1letter,
            )
        )
    return tuple(overlaid)


def _build_register_from_full_sequence(
    source_chain: _SourceChain,
    sequence: str,
    *,
    sequence_source: str,
    field_name: str,
) -> tuple[_ResidueBuilder, ...]:
    observed_positions = _map_observed_residues_to_sequence(
        source_chain.residues,
        sequence,
        field_name=field_name,
    )
    observed_by_position = {
        position: residue
        for position, residue in zip(observed_positions, source_chain.residues)
    }
    residues: list[_ResidueBuilder] = []
    for index, aa in enumerate(sequence, start=1):
        res_name = AA1_TO_3[aa]
        observed = observed_by_position.get(index - 1)
        if observed is None:
            residues.append(
                _ResidueBuilder(
                    auth_asym_id=source_chain.auth_asym_id,
                    label_asym_id=source_chain.label_asym_id,
                    label_entity_id=source_chain.label_entity_id,
                    auth_seq_id=str(index),
                    label_seq_id=str(index),
                    pdbx_pdb_ins_code="",
                    res_name=res_name,
                    atoms={},
                    atom_order=(),
                    is_observed=False,
                    sequence_source=sequence_source,
                    sequence_1letter=aa,
                )
            )
            continue
        residues.append(
            _ResidueBuilder(
                auth_asym_id=observed.auth_asym_id,
                label_asym_id=observed.label_asym_id,
                label_entity_id=observed.label_entity_id,
                auth_seq_id=observed.auth_seq_id,
                label_seq_id=str(index),
                pdbx_pdb_ins_code=observed.pdbx_pdb_ins_code,
                res_name=res_name,
                atoms=observed.atoms,
                atom_order=observed.atom_order,
                is_observed=True,
                sequence_source=sequence_source,
                sequence_1letter=aa,
            )
        )
    return tuple(residues)


def _map_observed_residues_to_sequence(
    observed_residues: tuple[_ResidueBuilder, ...],
    sequence: str,
    *,
    field_name: str,
) -> tuple[int, ...]:
    observed_sequence = "".join(
        residue.sequence_1letter or AA3_TO_1[residue.res_name]
        for residue in observed_residues
    )
    direct_positions = _direct_auth_sequence_positions(observed_residues, sequence)
    if direct_positions is not None:
        return direct_positions
    subsequence_positions = _unique_subsequence_positions(sequence, observed_sequence)
    if subsequence_positions is None:
        raise StructureTargetError(
            f"{field_name} cannot be aligned unambiguously to observed ATOM "
            "records; provide an mmCIF file with sequence metadata or a "
            "non-ambiguous full chain sequence"
        )
    return subsequence_positions


def _direct_auth_sequence_positions(
    observed_residues: tuple[_ResidueBuilder, ...],
    sequence: str,
) -> tuple[int, ...] | None:
    positions: list[int] = []
    seen: set[int] = set()
    for residue in observed_residues:
        auth_int = _int_or_none(residue.auth_seq_id)
        if auth_int is None:
            return None
        position = auth_int - 1
        if position < 0 or position >= len(sequence) or position in seen:
            return None
        residue_letter = residue.sequence_1letter or AA3_TO_1[residue.res_name]
        if sequence[position] != residue_letter:
            return None
        seen.add(position)
        positions.append(position)
    return tuple(positions)


def _unique_subsequence_positions(
    sequence: str,
    observed_sequence: str,
) -> tuple[int, ...] | None:
    n = len(sequence)
    m = len(observed_sequence)
    if m > n:
        return None
    counts = [[0] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        counts[m][j] = 1
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            value = counts[i][j + 1]
            if sequence[j] == observed_sequence[i]:
                value += counts[i + 1][j + 1]
            counts[i][j] = min(value, 2)
    if counts[0][0] != 1:
        return None
    positions: list[int] = []
    i = 0
    j = 0
    while i < m and j < n:
        skip_count = counts[i][j + 1] if j + 1 <= n else 0
        take_count = (
            counts[i + 1][j + 1]
            if sequence[j] == observed_sequence[i] and j + 1 <= n
            else 0
        )
        if take_count and not skip_count:
            positions.append(j)
            i += 1
            j += 1
        else:
            j += 1
    return tuple(positions) if len(positions) == m else None


def _user_sequence_for_chain(
    sequences: dict[str, str] | None,
    *,
    source_chain: _SourceChain,
    canonical_chain_id: str,
) -> str | None:
    if not sequences:
        return None
    matches = []
    for key in (canonical_chain_id, source_chain.input_id, source_chain.auth_asym_id, source_chain.label_asym_id):
        if key in sequences and key not in matches:
            matches.append(key)
    if not matches:
        return None
    values = {sequences[key] for key in matches}
    if len(values) != 1:
        raise StructureTargetError(
            f"target.structure.sequences provides conflicting entries for chain "
            f"{canonical_chain_id}: {', '.join(matches)}"
        )
    return sequences[matches[0]]


def _sequence_for_source_chain(
    sequences: dict[str, tuple[str, ...]],
    source_chain: _SourceChain,
) -> tuple[str, ...] | None:
    for key in (source_chain.input_id, source_chain.auth_asym_id, source_chain.label_asym_id):
        value = sequences.get(key)
        if value:
            return value
    return None


def _validate_observed_residue_register_contiguous(
    source_chain: _SourceChain,
    *,
    input_format: str,
) -> None:
    if len(source_chain.residues) < 2:
        return
    ids = [
        _int_or_none(residue.label_seq_id if input_format == "mmcif" else residue.auth_seq_id)
        for residue in source_chain.residues
    ]
    if any(value is None for value in ids):
        return
    assert all(value is not None for value in ids)
    gaps: list[tuple[int, int]] = []
    for left, right in zip(ids[:-1], ids[1:]):
        assert left is not None and right is not None
        if right not in {left, left + 1}:
            gaps.append((left, right))
    if gaps:
        formatted = ", ".join(f"{left}->{right}" for left, right in gaps[:5])
        raise StructureTargetError(
            f"target chain {source_chain.input_id} has discontinuous observed "
            f"residue records ({formatted}) but no full sequence metadata; provide "
            "an mmCIF/SEQRES record or target.structure.sequences"
        )


def _replace_residue_sequence_fields(
    residue: _ResidueBuilder,
    *,
    sequence_source: str,
    is_observed: bool,
) -> _ResidueBuilder:
    return _ResidueBuilder(
        auth_asym_id=residue.auth_asym_id,
        label_asym_id=residue.label_asym_id,
        label_entity_id=residue.label_entity_id,
        auth_seq_id=residue.auth_seq_id,
        label_seq_id=residue.label_seq_id,
        pdbx_pdb_ins_code=residue.pdbx_pdb_ins_code,
        res_name=residue.res_name,
        atoms=residue.atoms,
        atom_order=residue.atom_order,
        is_observed=is_observed,
        sequence_source=sequence_source,
        sequence_1letter=residue.sequence_1letter or AA3_TO_1[residue.res_name],
    )


def _sort_by_numeric_label_seq_id(
    residues: list[_ResidueBuilder],
) -> tuple[_ResidueBuilder, ...]:
    if all(_int_or_none(residue.label_seq_id) is not None for residue in residues):
        return tuple(sorted(residues, key=lambda residue: int(residue.label_seq_id)))
    return tuple(residues)


def _canonical_chain_id(input_id: str, counts: dict[str, int]) -> str:
    base = input_id or "chain"
    counts[base] = counts.get(base, 0) + 1
    if counts[base] == 1:
        return base
    return f"{base}_{counts[base]}"


def _crop_residues(
    residues: tuple[_ResidueBuilder, ...],
    *,
    canonical_chain_id: str,
    config: StructureTargetConfig,
    selected_chain_count: int,
) -> tuple[_ResidueBuilder, ...]:
    selectors = _selectors_for_chain(
        config.crop or {},
        canonical_chain_id=canonical_chain_id,
        auth_asym_id=residues[0].auth_asym_id,
        label_asym_id=residues[0].label_asym_id,
        field_name="target.crop",
        allow_global=True,
        selected_chain_count=selected_chain_count,
    )
    if not selectors:
        return residues
    indices = _resolve_selectors(
        residues,
        selectors,
        indexing=config.structure_indexing,
        field_name=f"target.crop[{canonical_chain_id}]",
    )
    return tuple(residue for index, residue in enumerate(residues) if index in indices)


def _rewrite_residues_for_chain(
    residues: tuple[_ResidueBuilder, ...],
    canonical_chain_id: str,
) -> tuple[TargetResidue, ...]:
    rewritten: list[TargetResidue] = []
    for index, residue in enumerate(residues):
        representative_atom = _representative_atom_name(residue)
        representative_coord = (
            residue.atoms[representative_atom]
            if representative_atom is not None
            else None
        )
        label_seq_id = residue.label_seq_id or str(index + 1)
        rewritten.append(
            TargetResidue(
                input_chain=residue.auth_asym_id or residue.label_asym_id,
                input_residue=residue.auth_seq_id,
                input_insertion_code=residue.pdbx_pdb_ins_code,
                auth_asym_id=residue.auth_asym_id,
                auth_seq_id=residue.auth_seq_id,
                pdbx_pdb_ins_code=residue.pdbx_pdb_ins_code,
                label_asym_id=residue.label_asym_id,
                label_seq_id=label_seq_id,
                canonical_chain_id=canonical_chain_id,
                model_residue_index_0=index,
                res_name=residue.res_name,
                sequence_1letter=residue.sequence_1letter or AA3_TO_1[residue.res_name],
                atoms=residue.atoms,
                atom_order=residue.atom_order,
                representative_atom=representative_atom,
                representative_coord=representative_coord,
                is_observed=residue.is_observed,
                sequence_source=residue.sequence_source,
            )
        )
    return tuple(rewritten)


def _resolve_hotspot_indices(
    residues: tuple[TargetResidue, ...],
    *,
    canonical_chain_id: str,
    auth_asym_id: str,
    label_asym_id: str,
    config: StructureTargetConfig,
    selected_chain_count: int,
) -> tuple[int, ...]:
    selectors = _selectors_for_chain(
        config.hotspots or {},
        canonical_chain_id=canonical_chain_id,
        auth_asym_id=auth_asym_id,
        label_asym_id=label_asym_id,
        field_name="target.hotspots",
        allow_global=False,
        selected_chain_count=selected_chain_count,
    )
    if not selectors:
        return ()
    residue_builders = tuple(
            _ResidueBuilder(
                auth_asym_id=residue.auth_asym_id,
                label_asym_id=residue.label_asym_id,
                label_entity_id="",
                auth_seq_id=residue.auth_seq_id,
                label_seq_id=residue.label_seq_id,
                pdbx_pdb_ins_code=residue.pdbx_pdb_ins_code,
                res_name=residue.res_name,
                atoms=residue.atoms,
                atom_order=residue.atom_order,
                is_observed=residue.is_observed,
                sequence_source=residue.sequence_source,
                sequence_1letter=residue.sequence_1letter,
            )
            for residue in residues
        )
    return tuple(
        _resolve_selectors(
            residue_builders,
            selectors,
            indexing=config.structure_indexing,
            field_name=f"target.hotspots[{canonical_chain_id}]",
        )
    )


def _selectors_for_chain(
    selector_map: dict[str, tuple[str, ...]],
    *,
    canonical_chain_id: str,
    auth_asym_id: str,
    label_asym_id: str,
    field_name: str,
    allow_global: bool,
    selected_chain_count: int,
) -> tuple[str, ...]:
    for key in (canonical_chain_id, auth_asym_id, label_asym_id):
        if key in selector_map:
            return selector_map[key]
    if "*" not in selector_map:
        return ()
    if not allow_global:
        raise StructureTargetError(
            f"{field_name} must be keyed by chain id; wildcard is not allowed"
        )
    if selected_chain_count != 1:
        raise StructureTargetError(
            f"{field_name} uses an unqualified selector but multiple chains are selected"
        )
    return selector_map["*"]


def _resolve_selectors(
    residues: tuple[_ResidueBuilder, ...],
    selectors: tuple[str, ...],
    *,
    indexing: str,
    field_name: str,
) -> tuple[int, ...]:
    selected: set[int] = set()
    for selector in selectors:
        matches = _resolve_one_selector(residues, selector, indexing=indexing)
        if not matches:
            raise StructureTargetError(f"{field_name} selector {selector!r} matched no residues")
        selected.update(matches)
    return tuple(sorted(selected))


def _resolve_one_selector(
    residues: tuple[_ResidueBuilder, ...],
    selector: str,
    *,
    indexing: str,
) -> set[int]:
    text = str(selector).strip()
    if not text:
        raise StructureTargetError("empty residue selector")
    if "-" in text and not text.startswith("-"):
        start_text, end_text = text.split("-", 1)
        start = int(start_text.strip())
        end = int(end_text.strip())
        if end < start:
            raise StructureTargetError(f"residue range {selector!r} has end before start")
        return _resolve_numeric_range(residues, start, end, indexing=indexing)
    return _resolve_exact_residue(residues, text, indexing=indexing)


def _resolve_exact_residue(
    residues: tuple[_ResidueBuilder, ...],
    text: str,
    *,
    indexing: str,
) -> set[int]:
    if indexing == "auth_seq_id":
        return {
            index
            for index, residue in enumerate(residues)
            if _residue_selector_text(residue.auth_seq_id, residue.pdbx_pdb_ins_code) == text
        }
    if indexing == "label_seq_id":
        return {
            index
            for index, residue in enumerate(residues)
            if residue.label_seq_id == text
        }
    auth_matches = _resolve_exact_residue(residues, text, indexing="auth_seq_id")
    label_matches = _resolve_exact_residue(residues, text, indexing="label_seq_id")
    return _auto_matches(auth_matches, label_matches, text)


def _resolve_numeric_range(
    residues: tuple[_ResidueBuilder, ...],
    start: int,
    end: int,
    *,
    indexing: str,
) -> set[int]:
    if indexing == "auth_seq_id":
        return {
            index
            for index, residue in enumerate(residues)
            if _int_or_none(residue.auth_seq_id) is not None
            and start <= int(residue.auth_seq_id) <= end
        }
    if indexing == "label_seq_id":
        return {
            index
            for index, residue in enumerate(residues)
            if _int_or_none(residue.label_seq_id) is not None
            and start <= int(residue.label_seq_id) <= end
        }
    auth_matches = _resolve_numeric_range(residues, start, end, indexing="auth_seq_id")
    label_matches = _resolve_numeric_range(residues, start, end, indexing="label_seq_id")
    return _auto_matches(auth_matches, label_matches, f"{start}-{end}")


def _auto_matches(auth_matches: set[int], label_matches: set[int], selector: str) -> set[int]:
    if auth_matches and label_matches and auth_matches != label_matches:
        raise StructureTargetError(
            f"selector {selector!r} is ambiguous under structure_indexing=auto; "
            "set structure_indexing to auth_seq_id or label_seq_id"
        )
    return auth_matches or label_matches


def _validate_hotspots_resolved(
    residues: tuple[TargetResidue, ...],
    hotspot_indices: tuple[int, ...],
    canonical_chain_id: str,
) -> None:
    unresolved = [
        _format_residue_id(residues[index])
        for index in hotspot_indices
        if residues[index].representative_coord is None
    ]
    if unresolved:
        formatted = ", ".join(unresolved)
        raise StructureTargetError(
            f"target.hotspots for chain {canonical_chain_id} selected unresolved "
            f"representative coordinates: {formatted}"
        )


def _compute_distogram(
    residues: tuple[TargetResidue, ...],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[bool, ...], ...]]:
    coords = [residue.representative_coord for residue in residues]
    rows: list[tuple[float, ...]] = []
    mask_rows: list[tuple[bool, ...]] = []
    for coord_a in coords:
        row: list[float] = []
        mask_row: list[bool] = []
        for coord_b in coords:
            if coord_a is None or coord_b is None:
                row.append(0.0)
                mask_row.append(False)
                continue
            dx = coord_a[0] - coord_b[0]
            dy = coord_a[1] - coord_b[1]
            dz = coord_a[2] - coord_b[2]
            row.append(math.sqrt(dx * dx + dy * dy + dz * dz))
            mask_row.append(True)
        rows.append(tuple(row))
        mask_rows.append(tuple(mask_row))
    return tuple(rows), tuple(mask_rows)


def _chain_sequence_source(residues: tuple[TargetResidue, ...]) -> str:
    sources = tuple(dict.fromkeys(residue.sequence_source for residue in residues))
    if not sources:
        return ""
    if len(sources) == 1:
        return sources[0]
    return "mixed:" + ",".join(sources)


def _representative_atom_name(residue: _ResidueBuilder) -> str | None:
    if residue.res_name == "GLY":
        return "CA" if "CA" in residue.atoms else None
    if "CB" in residue.atoms:
        return "CB"
    if "CA" in residue.atoms:
        return "CA"
    return None


def _write_normalized_cif(prepared: PreparedTarget, path: Path) -> Path:
    lines = [
        f"data_{_safe_name(prepared.source_path.stem) or 'target'}",
        "#",
    ]
    _append_normalized_cif_sequence_metadata(lines, prepared)
    lines.extend(
        [
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
    entity_ids = {
        chain.canonical_chain_id: str(index)
        for index, chain in enumerate(prepared.chains, start=1)
    }
    atom_id = 1
    for chain in prepared.chains:
        entity_id = entity_ids[chain.canonical_chain_id]
        for residue in chain.residues:
            ordered_atoms = [
                atom_name
                for atom_name in residue.atom_order
                if atom_name in residue.atoms
            ]
            for atom_name in ATOM37_ORDER:
                if atom_name in residue.atoms and atom_name not in ordered_atoms:
                    ordered_atoms.append(atom_name)
            for atom_name in ordered_atoms:
                x, y, z = residue.atoms[atom_name]
                ins_code = residue.pdbx_pdb_ins_code or "?"
                type_symbol = _atom_type_symbol(atom_name)
                lines.append(
                    " ".join(
                        [
                            "ATOM",
                            str(atom_id),
                            type_symbol,
                            atom_name,
                            ".",
                            residue.res_name,
                            residue.canonical_chain_id,
                            entity_id,
                            str(residue.model_residue_index_0 + 1),
                            ins_code,
                            f"{x:.3f}",
                            f"{y:.3f}",
                            f"{z:.3f}",
                            "1.00",
                            "0.00",
                            "?",
                            residue.auth_seq_id or str(residue.model_residue_index_0 + 1),
                            residue.res_name,
                            residue.auth_asym_id or residue.canonical_chain_id,
                            atom_name,
                            "1",
                        ]
                    )
                )
                atom_id += 1
    lines.append("#")
    return write_text_atomic(path, "\n".join(lines) + "\n")


def _append_normalized_cif_sequence_metadata(
    lines: list[str],
    prepared: PreparedTarget,
) -> None:
    chem_comp_ids = sorted(
        {
            residue.res_name
            for chain in prepared.chains
            for residue in chain.residues
        }
    )
    if chem_comp_ids:
        lines.extend(
            [
                "loop_",
                "_chem_comp.id",
                "_chem_comp.type",
            ]
        )
        for res_name in chem_comp_ids:
            lines.append(f"{_cif_scalar(res_name)} {_cif_scalar('L-peptide linking')}")
        lines.append("#")

    lines.extend(
        [
            "loop_",
            "_entity_poly.entity_id",
            "_entity_poly.type",
            "_entity_poly.nstd_linkage",
            "_entity_poly.nstd_monomer",
            "_entity_poly.pdbx_seq_one_letter_code",
            "_entity_poly.pdbx_seq_one_letter_code_can",
            "_entity_poly.pdbx_strand_id",
            "_entity_poly.pdbx_target_identifier",
        ]
    )
    for entity_id, chain in enumerate(prepared.chains, start=1):
        sequence = "".join(residue.sequence_1letter for residue in chain.residues)
        chain_id = chain.canonical_chain_id
        lines.append(
            " ".join(
                [
                    str(entity_id),
                    _cif_scalar("polypeptide(L)"),
                    "no",
                    "no",
                    _cif_scalar(sequence),
                    _cif_scalar(sequence),
                    _cif_scalar(chain_id),
                    "?",
                ]
            )
        )
    lines.append("#")

    lines.extend(
        [
            "loop_",
            "_entity_poly_seq.entity_id",
            "_entity_poly_seq.num",
            "_entity_poly_seq.mon_id",
            "_entity_poly_seq.hetero",
        ]
    )
    for entity_id, chain in enumerate(prepared.chains, start=1):
        for seq_id, residue in enumerate(chain.residues, start=1):
            lines.append(
                " ".join(
                    [
                        str(entity_id),
                        str(seq_id),
                        _cif_scalar(residue.res_name),
                        "n",
                    ]
                )
            )
    lines.append("#")

    lines.extend(
        [
            "loop_",
            "_struct_asym.id",
            "_struct_asym.entity_id",
        ]
    )
    for entity_id, chain in enumerate(prepared.chains, start=1):
        lines.append(f"{_cif_scalar(chain.canonical_chain_id)} {entity_id}")
    lines.append("#")

    lines.extend(
        [
            "loop_",
            "_pdbx_poly_seq_scheme.asym_id",
            "_pdbx_poly_seq_scheme.entity_id",
            "_pdbx_poly_seq_scheme.seq_id",
            "_pdbx_poly_seq_scheme.mon_id",
            "_pdbx_poly_seq_scheme.ndb_seq_num",
            "_pdbx_poly_seq_scheme.pdb_seq_num",
            "_pdbx_poly_seq_scheme.auth_seq_num",
            "_pdbx_poly_seq_scheme.pdb_mon_id",
            "_pdbx_poly_seq_scheme.auth_mon_id",
            "_pdbx_poly_seq_scheme.pdb_strand_id",
            "_pdbx_poly_seq_scheme.pdb_ins_code",
            "_pdbx_poly_seq_scheme.hetero",
        ]
    )
    for entity_id, chain in enumerate(prepared.chains, start=1):
        chain_id = chain.canonical_chain_id
        for seq_id, residue in enumerate(chain.residues, start=1):
            auth_seq_id = residue.auth_seq_id or str(seq_id)
            auth_asym_id = residue.auth_asym_id or chain_id
            ins_code = residue.pdbx_pdb_ins_code or "?"
            lines.append(
                " ".join(
                    [
                        _cif_scalar(chain_id),
                        str(entity_id),
                        str(seq_id),
                        _cif_scalar(residue.res_name),
                        str(seq_id),
                        _cif_scalar(auth_seq_id),
                        _cif_scalar(auth_seq_id),
                        _cif_scalar(residue.res_name),
                        _cif_scalar(residue.res_name),
                        _cif_scalar(auth_asym_id),
                        _cif_scalar(ins_code),
                        "n",
                    ]
                )
            )
    lines.append("#")


def _cif_scalar(value: object) -> str:
    text = str(value)
    if text == "":
        return "?"
    if text in {".", "?"}:
        return text
    if any(ch.isspace() for ch in text) or text[0] in {"_", "#", "$", ";", "'", '"'}:
        return "'" + text.replace("'", "''") + "'"
    return text


def _write_residue_map(prepared: PreparedTarget, path: Path) -> Path:
    columns = [
        "canonical_chain_id",
        "model_residue_index_0",
        "sequence_1letter",
        "res_name",
        "input_chain",
        "input_residue",
        "input_insertion_code",
        "auth_asym_id",
        "auth_seq_id",
        "pdbx_PDB_ins_code",
        "label_asym_id",
        "label_seq_id",
        "is_observed",
        "sequence_source",
        "representative_atom",
        "resolved_representative_coord",
        "rep_x",
        "rep_y",
        "rep_z",
    ]
    rows: list[dict[str, object]] = []
    for chain in prepared.chains:
        for residue in chain.residues:
            coord = residue.representative_coord
            rows.append(
                {
                    "canonical_chain_id": residue.canonical_chain_id,
                    "model_residue_index_0": residue.model_residue_index_0,
                    "sequence_1letter": residue.sequence_1letter,
                    "res_name": residue.res_name,
                    "input_chain": residue.input_chain,
                    "input_residue": residue.input_residue,
                    "input_insertion_code": residue.input_insertion_code,
                    "auth_asym_id": residue.auth_asym_id,
                    "auth_seq_id": residue.auth_seq_id,
                    "pdbx_PDB_ins_code": residue.pdbx_pdb_ins_code,
                    "label_asym_id": residue.label_asym_id,
                    "label_seq_id": residue.label_seq_id,
                    "is_observed": str(residue.is_observed).lower(),
                    "sequence_source": residue.sequence_source,
                    "representative_atom": residue.representative_atom or "",
                    "resolved_representative_coord": str(coord is not None).lower(),
                    "rep_x": "" if coord is None else f"{coord[0]:.3f}",
                    "rep_y": "" if coord is None else f"{coord[1]:.3f}",
                    "rep_z": "" if coord is None else f"{coord[2]:.3f}",
                }
            )

    output = []
    writer_buffer = _CsvBuffer(output)
    writer = csv.DictWriter(writer_buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return write_text_atomic(path, "".join(output))


def _write_assembly_conditioning(
    prepared: PreparedTarget,
    *,
    conditioning_dir: Path,
    target_dir: Path,
    conditioning_chain_pairs: tuple[tuple[str, str], ...] | None,
    partial_conditioning: bool,
    representative_atom: str,
    require_resolved: bool,
) -> tuple[list[Path], Path]:
    if len(prepared.chains) < 2:
        raise StructureTargetError(
            "target.conditioning.assembly requires at least two selected target chains"
        )
    pairs = _resolve_conditioning_chain_pairs(prepared, conditioning_chain_pairs)
    if not pairs:
        raise StructureTargetError(
            "target.conditioning.assembly requires at least one target chain pair"
        )
    pair_files: list[Path] = []
    pair_summaries: list[dict[str, object]] = []
    for chain_a, chain_b in pairs:
        distogram, distogram_mask = _compute_pair_distogram(chain_a, chain_b)
        if not partial_conditioning and not all(value for row in distogram_mask for value in row):
            raise StructureTargetError(
                f"target chains {chain_a.canonical_chain_id}-{chain_b.canonical_chain_id} "
                "have unresolved representative coordinates but "
                "target.conditioning.partial is false"
            )
        safe_a = _safe_name(chain_a.canonical_chain_id)
        safe_b = _safe_name(chain_b.canonical_chain_id)
        pair_path = conditioning_dir / f"pair_{safe_a}_{safe_b}_distogram.npy"
        pair_mask_path = conditioning_dir / f"pair_{safe_a}_{safe_b}_distogram_mask.npy"
        pair_files.append(
            _write_float32_npy(
                pair_path,
                distogram,
                (len(chain_a.residues), len(chain_b.residues)),
            )
        )
        pair_files.append(
            _write_bool_npy(
                pair_mask_path,
                distogram_mask,
                (len(chain_a.residues), len(chain_b.residues)),
            )
        )
        pair_summaries.append(
            {
                "chain_id_1": chain_a.canonical_chain_id,
                "chain_id_2": chain_b.canonical_chain_id,
                "shape": [len(chain_a.residues), len(chain_b.residues)],
                "distogram_path": str(pair_path.relative_to(target_dir)),
                "distogram_mask_path": str(pair_mask_path.relative_to(target_dir)),
                "distogram_mask_true": sum(value for row in distogram_mask for value in row),
            }
        )

    metadata = {
        "assembly": True,
        "conditioning_mode": "distogram",
        "chain_order": [chain.canonical_chain_id for chain in prepared.chains],
        "target_chain_spans": _target_chain_spans(prepared),
        "chain_pairs": pair_summaries,
        "representative_atom": representative_atom,
        "partial_conditioning": partial_conditioning,
        "require_resolved": require_resolved,
        "warnings": list(prepared.warnings),
    }
    metadata_path = write_json_atomic(
        conditioning_dir / "assembly_conditioning.json",
        metadata,
    )
    return pair_files, metadata_path


def _resolve_conditioning_chain_pairs(
    prepared: PreparedTarget,
    requested_pairs: tuple[tuple[str, str], ...] | None,
) -> tuple[tuple[TargetChain, TargetChain], ...]:
    if requested_pairs is None:
        pairs: list[tuple[TargetChain, TargetChain]] = []
        chains = list(prepared.chains)
        for left_index, chain_a in enumerate(chains):
            for chain_b in chains[left_index + 1 :]:
                pairs.append((chain_a, chain_b))
        return tuple(pairs)

    pairs = []
    seen: set[tuple[str, str]] = set()
    for chain_id_1, chain_id_2 in requested_pairs:
        chain_a = _resolve_conditioning_chain(prepared, chain_id_1)
        chain_b = _resolve_conditioning_chain(prepared, chain_id_2)
        if chain_a.canonical_chain_id == chain_b.canonical_chain_id:
            raise StructureTargetError(
                "target.conditioning.chain_pairs cannot pair a chain with itself: "
                f"{chain_id_1}, {chain_id_2}"
            )
        key = tuple(sorted((chain_a.canonical_chain_id, chain_b.canonical_chain_id)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((chain_a, chain_b))
    return tuple(pairs)


def _resolve_conditioning_chain(prepared: PreparedTarget, requested: str) -> TargetChain:
    matches = [
        chain
        for chain in prepared.chains
        if requested
        in {
            chain.canonical_chain_id,
            chain.auth_asym_id,
            chain.label_asym_id,
        }
    ]
    if not matches:
        available = ", ".join(
            f"{chain.canonical_chain_id}(auth={chain.auth_asym_id},label={chain.label_asym_id})"
            for chain in prepared.chains
        )
        raise StructureTargetError(
            f"target.conditioning.chain_pairs references unknown chain {requested}; "
            f"available: {available}"
        )
    if len(matches) > 1:
        raise StructureTargetError(
            f"target.conditioning.chain_pairs chain {requested} is ambiguous; "
            "use a canonical chain id from check output"
        )
    return matches[0]


def _iter_target_chains_with_start(
    prepared: PreparedTarget,
) -> Iterable[tuple[int, TargetChain]]:
    start = 0
    for chain in prepared.chains:
        yield start, chain
        start += len(chain.residues)


def _iter_flat_target_residues(
    prepared: PreparedTarget,
) -> Iterable[tuple[int, TargetResidue]]:
    for chain_start, chain in _iter_target_chains_with_start(prepared):
        for local_index, residue in enumerate(chain.residues):
            yield chain_start + local_index, residue


def _validate_drift_region_chain_keys(
    prepared: PreparedTarget,
    regions: dict[str, tuple[str, ...]],
    *,
    field_name: str,
) -> None:
    for requested in regions:
        if requested == "*":
            if len(prepared.chains) != 1:
                raise StructureTargetError(
                    f"{field_name} uses an unqualified selector but multiple "
                    "chains are selected"
                )
            continue
        matches = [
            chain
            for chain in prepared.chains
            if requested
            in {
                chain.canonical_chain_id,
                chain.auth_asym_id,
                chain.label_asym_id,
            }
        ]
        if not matches:
            available = ", ".join(
                f"{chain.canonical_chain_id}(auth={chain.auth_asym_id},label={chain.label_asym_id})"
                for chain in prepared.chains
            )
            raise StructureTargetError(
                f"{field_name} references unknown chain {requested}; "
                f"available: {available}"
            )
        if len(matches) > 1:
            raise StructureTargetError(
                f"{field_name} chain {requested} is ambiguous; use a canonical "
                "chain id from check output"
            )


def _drift_selectors_for_chain(
    chain: TargetChain,
    regions: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    selectors: list[str] = []
    seen_keys: set[str] = set()
    for key in (chain.canonical_chain_id, chain.auth_asym_id, chain.label_asym_id):
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selectors.extend(regions.get(key, ()))
    selectors.extend(regions.get("*", ()))
    return tuple(selectors)


def _residue_builders_from_target_residues(
    residues: tuple[TargetResidue, ...],
) -> tuple[_ResidueBuilder, ...]:
    return tuple(
        _ResidueBuilder(
            auth_asym_id=residue.auth_asym_id,
            label_asym_id=residue.label_asym_id,
            label_entity_id="",
            auth_seq_id=residue.auth_seq_id,
            label_seq_id=residue.label_seq_id,
            pdbx_pdb_ins_code=residue.pdbx_pdb_ins_code,
            res_name=residue.res_name,
            atoms=residue.atoms,
            atom_order=residue.atom_order,
            is_observed=residue.is_observed,
            sequence_source=residue.sequence_source,
            sequence_1letter=residue.sequence_1letter,
        )
        for residue in residues
    )


def _compute_pair_distogram(
    chain_a: TargetChain,
    chain_b: TargetChain,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[bool, ...], ...]]:
    coords_a = [residue.representative_coord for residue in chain_a.residues]
    coords_b = [residue.representative_coord for residue in chain_b.residues]

    rows: list[tuple[float, ...]] = []
    mask_rows: list[tuple[bool, ...]] = []
    for coord_a in coords_a:
        row: list[float] = []
        mask_row: list[bool] = []
        for coord_b in coords_b:
            if coord_a is None or coord_b is None:
                row.append(0.0)
                mask_row.append(False)
                continue
            dx = coord_a[0] - coord_b[0]
            dy = coord_a[1] - coord_b[1]
            dz = coord_a[2] - coord_b[2]
            row.append(math.sqrt(dx * dx + dy * dy + dz * dz))
            mask_row.append(True)
        rows.append(tuple(row))
        mask_rows.append(tuple(mask_row))
    return tuple(rows), tuple(mask_rows)


def _target_chain_spans(prepared: PreparedTarget) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    start = 0
    for chain in prepared.chains:
        end = start + len(chain.residues)
        spans.append(
            {
                "chain_id": chain.canonical_chain_id,
                "auth_asym_id": chain.auth_asym_id,
                "label_asym_id": chain.label_asym_id,
                "start": start,
                "end": end,
                "length": len(chain.residues),
            }
        )
        start = end
    return spans


def _chain_summary(
    prepared: PreparedTarget,
    *,
    conditioning_mode: str,
    conditioning_assembly: bool = False,
    conditioning_chain_pairs: tuple[tuple[str, str], ...] | None = None,
) -> dict[str, object]:
    return {
        "source_path": str(prepared.source_path),
        "input_format": prepared.input_format,
        "conditioning_mode": conditioning_mode,
        "conditioning_assembly": conditioning_assembly,
        "conditioning_chain_pairs": (
            "auto"
            if conditioning_chain_pairs is None
            else [list(pair) for pair in conditioning_chain_pairs]
        ),
        "target_chain_spans": _target_chain_spans(prepared),
        "chains": [
            {
                "canonical_chain_id": chain.canonical_chain_id,
                "auth_asym_id": chain.auth_asym_id,
                "label_asym_id": chain.label_asym_id,
                "length": len(chain.residues),
                "sequence": chain.sequence,
                "sequence_source": chain.sequence_source,
                "hotspot_indices": list(chain.hotspot_indices),
                "representative_coords_resolved": sum(
                    residue.representative_coord is not None for residue in chain.residues
                ),
                "distogram_mask_true": sum(
                    value for row in chain.distogram_mask for value in row
                ),
                "has_partial_conditioning": not all(chain.representative_coord_mask),
                "distogram_shape": [
                    len(chain.distogram),
                    len(chain.distogram[0]) if chain.distogram else 0,
                ],
                "distogram_mask_shape": [
                    len(chain.distogram_mask),
                    len(chain.distogram_mask[0]) if chain.distogram_mask else 0,
                ],
            }
            for chain in prepared.chains
        ],
        "warnings": list(prepared.warnings),
    }


def _write_float32_npy(
    path: Path,
    rows: Iterable[Iterable[float | int]],
    shape: tuple[int, ...],
) -> Path:
    flattened = [float(value) for row in rows for value in row]
    return _write_npy(path, flattened, shape, descr="<f4")


def _write_bool_npy(
    path: Path,
    values: Iterable[bool] | Iterable[Iterable[bool]],
    shape: tuple[int, ...],
) -> Path:
    flattened: list[bool] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flattened.extend(bool(item) for item in value)
        else:
            flattened.append(bool(value))
    return _write_npy(path, flattened, shape, descr="|b1")


def _write_npy(
    path: Path,
    flattened: list[float] | list[bool],
    shape: tuple[int, ...],
    *,
    descr: str,
) -> Path:
    if len(shape) == 1:
        expected = shape[0]
    elif len(shape) == 2:
        expected = shape[0] * shape[1]
    else:
        raise StructureTargetError(f"unsupported npy shape for {path.name}: {shape}")
    if len(flattened) != expected:
        raise StructureTargetError(
            f"array for {path.name} has {len(flattened)} values but shape {shape} "
            f"requires {expected}"
        )
    header = {
        "descr": descr,
        "fortran_order": False,
        "shape": shape,
    }
    header_text = repr(header)
    header_text = header_text.replace("False", "False")
    header_bytes = header_text.encode("latin1")
    prefix_len = 10
    padding = 16 - ((prefix_len + len(header_bytes) + 1) % 16)
    header_bytes = header_bytes + b" " * padding + b"\n"
    if len(header_bytes) > 65535:
        raise StructureTargetError(f"npy header too large for {path}")
    data = bytearray()
    data.extend(b"\x93NUMPY")
    data.extend(b"\x01\x00")
    data.extend(struct.pack("<H", len(header_bytes)))
    data.extend(header_bytes)
    if descr == "<f4":
        for value in flattened:
            data.extend(struct.pack("<f", float(value)))
    elif descr == "|b1":
        for value in flattened:
            data.extend(b"\x01" if bool(value) else b"\x00")
    else:
        raise StructureTargetError(f"unsupported npy dtype for {path.name}: {descr}")
    return write_bytes_atomic(path, bytes(data))


class _CsvBuffer:
    def __init__(self, output: list[str]) -> None:
        self._output = output

    def write(self, value: str) -> int:
        self._output.append(value)
        return len(value)


def _format_residue_id(residue: TargetResidue) -> str:
    ins_code = residue.pdbx_pdb_ins_code
    suffix = ins_code if ins_code else ""
    return f"{residue.canonical_chain_id}:{residue.auth_seq_id}{suffix}"


def _residue_selector_text(seq_id: str, ins_code: str) -> str:
    return f"{seq_id}{ins_code}" if ins_code else seq_id


def _clean_missing(value: str) -> str:
    text = value.strip()
    return "" if text in {"", ".", "?"} else text


def _normalize_res_name(value: str) -> str:
    text = _clean_missing(value).upper()
    return MODIFIED_AA3_TO_STANDARD.get(text, text)


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _atom_type_symbol(atom_name: str) -> str:
    for char in atom_name:
        if char.isalpha():
            return char.upper()
    return "?"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
