from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


_SCFV_PACKAGE = "esmfold2_pipeline.frameworks.scfv"
_SCFV_CDR_NAMES = ("hcdr1", "hcdr2", "hcdr3", "lcdr1", "lcdr2", "lcdr3")
_VHH_PACKAGE = "esmfold2_pipeline.frameworks.vhh"
_VHH_CDR_NAMES = ("cdr1", "cdr2", "cdr3")


@dataclass(frozen=True)
class FrameworkRecord:
    id: str
    canonical_name: str
    aliases: tuple[str, ...]
    modality: str
    template: str
    cdr_lengths: dict[str, tuple[int, int]]

    @property
    def accepted_names(self) -> tuple[str, ...]:
        return _unique_names((self.id, self.canonical_name, *self.aliases))


def all_scfv_framework_names() -> tuple[str, ...]:
    return tuple(sorted(_records_by_name(_SCFV_PACKAGE, "scfv", _SCFV_CDR_NAMES)))


def scfv_framework_alias_choices() -> tuple[str, ...]:
    return tuple(sorted(_alias_index(_SCFV_PACKAGE, "scfv", _SCFV_CDR_NAMES)))


def resolve_scfv_framework_name(name: str) -> str | None:
    return _alias_index(_SCFV_PACKAGE, "scfv", _SCFV_CDR_NAMES).get(_normalize_name(name))


def is_scfv_framework_alias(name: str) -> bool:
    return resolve_scfv_framework_name(name) is not None


def get_scfv_framework_record(name: str) -> FrameworkRecord:
    canonical = resolve_scfv_framework_name(name) or name
    records = _records_by_name(_SCFV_PACKAGE, "scfv", _SCFV_CDR_NAMES)
    try:
        return records[canonical]
    except KeyError as exc:
        choices = ", ".join(scfv_framework_alias_choices())
        raise KeyError(f"unknown scFv framework {name!r}; choices: {choices}") from exc


def get_scfv_framework_template_cif(name: str) -> Path | None:
    return _framework_template_cif(
        _SCFV_PACKAGE,
        "scfv",
        _SCFV_CDR_NAMES,
        name,
    )


def all_vhh_framework_names() -> tuple[str, ...]:
    return tuple(sorted(_records_by_name(_VHH_PACKAGE, "vhh", _VHH_CDR_NAMES)))


def vhh_framework_alias_choices() -> tuple[str, ...]:
    return tuple(sorted(_alias_index(_VHH_PACKAGE, "vhh", _VHH_CDR_NAMES)))


def resolve_vhh_framework_name(name: str) -> str | None:
    return _alias_index(_VHH_PACKAGE, "vhh", _VHH_CDR_NAMES).get(_normalize_name(name))


def is_vhh_framework_alias(name: str) -> bool:
    return resolve_vhh_framework_name(name) is not None


def get_vhh_framework_record(name: str) -> FrameworkRecord:
    canonical = resolve_vhh_framework_name(name) or name
    records = _records_by_name(_VHH_PACKAGE, "vhh", _VHH_CDR_NAMES)
    try:
        return records[canonical]
    except KeyError as exc:
        choices = ", ".join(vhh_framework_alias_choices())
        raise KeyError(f"unknown VHH framework {name!r}; choices: {choices}") from exc


def get_vhh_framework_template_cif(name: str) -> Path | None:
    return _framework_template_cif(
        _VHH_PACKAGE,
        "vhh",
        _VHH_CDR_NAMES,
        name,
    )


@cache
def _records_by_name(
    package: str,
    modality: str,
    cdr_names: tuple[str, ...],
) -> dict[str, FrameworkRecord]:
    records: dict[str, FrameworkRecord] = {}
    for file in sorted(resources.files(package).iterdir(), key=lambda item: item.name):
        if file.name.startswith("_") or file.suffix not in {".yaml", ".yml"}:
            continue
        data = yaml.safe_load(file.read_text()) or {}
        record = _record_from_yaml(data, file.name, cdr_names=cdr_names)
        if record.modality != modality:
            raise ValueError(f"{file.name}: modality must be {modality}")
        if record.canonical_name in records:
            raise ValueError(f"duplicate {modality} framework: {record.canonical_name}")
        records[record.canonical_name] = record
    if not records:
        raise ValueError(f"no bundled {modality} frameworks found")
    return records


@cache
def _alias_index(
    package: str,
    modality: str,
    cdr_names: tuple[str, ...],
) -> dict[str, str]:
    index: dict[str, str] = {}
    for record in _records_by_name(package, modality, cdr_names).values():
        for accepted_name in record.accepted_names:
            key = _normalize_name(accepted_name)
            existing = index.get(key)
            if existing is not None and existing != record.canonical_name:
                raise ValueError(
                    f"duplicate {modality} framework alias {accepted_name!r}: "
                    f"{existing}, {record.canonical_name}"
                )
            index[key] = record.canonical_name
    return index


@cache
def _framework_template_cif(
    package: str,
    modality: str,
    cdr_names: tuple[str, ...],
    name: str,
) -> Path | None:
    record = _records_by_name(package, modality, cdr_names).get(
        _alias_index(package, modality, cdr_names).get(_normalize_name(name), name)
    )
    if record is None:
        return None
    prefix = f"{record.id}_"
    matches = [
        file
        for file in resources.files(package).iterdir()
        if file.is_file()
        and file.suffix.lower() == ".cif"
        and (file.stem == record.id or file.name.startswith(prefix))
    ]
    if not matches:
        return None
    if len(matches) > 1:
        names = ", ".join(sorted(file.name for file in matches))
        raise ValueError(
            f"multiple bundled {modality} framework CIFs match {record.canonical_name}: {names}"
        )
    return Path(str(matches[0]))


def _record_from_yaml(
    data: Any,
    filename: str,
    *,
    cdr_names: tuple[str, ...],
) -> FrameworkRecord:
    if not isinstance(data, dict):
        raise ValueError(f"{filename}: framework record must be a mapping")
    framework_id = _required_string(data, "id", filename)
    canonical_name = _required_string(data, "canonical_name", filename)
    modality = _required_string(data, "modality", filename)
    template = _required_string(data, "template", filename)
    aliases = _string_list(data.get("aliases", []), "aliases", filename)
    cdr_lengths = _parse_cdr_lengths(data.get("cdr_lengths"), filename, cdr_names)
    return FrameworkRecord(
        id=framework_id,
        canonical_name=canonical_name,
        aliases=aliases,
        modality=modality,
        template=template,
        cdr_lengths=cdr_lengths,
    )


def _parse_cdr_lengths(
    value: Any,
    filename: str,
    cdr_names: tuple[str, ...],
) -> dict[str, tuple[int, int]]:
    if not isinstance(value, dict):
        raise ValueError(f"{filename}: cdr_lengths must be a mapping")
    unknown = sorted(set(value) - set(cdr_names))
    if unknown:
        raise ValueError(
            f"{filename}: cdr_lengths contains unsupported CDRs: {', '.join(unknown)}"
        )
    missing = [name for name in cdr_names if name not in value]
    if missing:
        raise ValueError(
            f"{filename}: cdr_lengths is missing CDRs: {', '.join(missing)}"
        )
    return {
        name: _parse_length(value[name], f"{filename}: cdr_lengths.{name}")
        for name in cdr_names
    }


def _parse_length(value: Any, field_name: str) -> tuple[int, int]:
    if isinstance(value, int):
        low = high = value
    elif isinstance(value, dict):
        low = value.get("min")
        high = value.get("max")
    else:
        raise ValueError(f"{field_name} must be an integer or {{min, max}} mapping")
    if not isinstance(low, int) or not isinstance(high, int):
        raise ValueError(f"{field_name}.min and .max must be integers")
    if low <= 0 or high <= 0:
        raise ValueError(f"{field_name} values must be positive")
    if high < low:
        raise ValueError(f"{field_name}.max must be >= min")
    return low, high


def _required_string(data: dict[str, Any], key: str, filename: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{filename}: {key} must be a non-empty string")
    return value


def _string_list(value: Any, field_name: str, filename: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{filename}: {field_name} must be a list")
    parsed: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"{filename}: {field_name}[{index}] must be a non-empty string"
            )
        parsed.append(item)
    return tuple(parsed)


def _unique_names(names: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        key = _normalize_name(name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(name)
    return tuple(unique)


def _normalize_name(name: str) -> str:
    return name.strip().lower()
