from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from esmfold2_pipeline.artifacts.writers import write_text_atomic


@dataclass(frozen=True)
class FastaRecord:
    """One FASTA record."""

    identifier: str
    sequence: str
    description: str = ""


def write_fasta(
    path: str | Path,
    records: Iterable[FastaRecord | tuple[str, str] | tuple[str, str, str]],
    *,
    overwrite: bool = True,
    line_width: int = 80,
) -> Path:
    """Write FASTA records atomically and return the final path."""

    if line_width <= 0:
        raise ValueError("line_width must be positive")

    normalized = [_normalize_record(record) for record in records]
    if not normalized:
        raise ValueError("at least one FASTA record is required")

    lines: list[str] = []
    for record in normalized:
        identifier = _clean_identifier(record.identifier)
        sequence = _clean_sequence(record.sequence)
        description = record.description.strip()

        header = f">{identifier}"
        if description:
            header = f"{header} {description}"
        lines.append(header)

        for start in range(0, len(sequence), line_width):
            lines.append(sequence[start : start + line_width])

    return write_text_atomic(path, "\n".join(lines) + "\n", overwrite=overwrite)


def _normalize_record(
    record: FastaRecord | tuple[str, str] | tuple[str, str, str],
) -> FastaRecord:
    if isinstance(record, FastaRecord):
        return record
    if len(record) == 2:
        identifier, sequence = record
        return FastaRecord(identifier=identifier, sequence=sequence)
    if len(record) == 3:
        identifier, sequence, description = record
        return FastaRecord(
            identifier=identifier,
            sequence=sequence,
            description=description,
        )
    raise ValueError("FASTA records must have 2 or 3 fields")


def _clean_identifier(identifier: str) -> str:
    value = identifier.strip()
    if not value:
        raise ValueError("FASTA identifier cannot be empty")
    if any(char.isspace() for char in value):
        raise ValueError("FASTA identifier cannot contain whitespace")
    if value.startswith(">"):
        raise ValueError("FASTA identifier cannot start with '>'")
    return value


def _clean_sequence(sequence: str) -> str:
    value = "".join(sequence.split()).upper()
    if not value:
        raise ValueError("FASTA sequence cannot be empty")
    if ">" in value:
        raise ValueError("FASTA sequence cannot contain '>'")
    return value

