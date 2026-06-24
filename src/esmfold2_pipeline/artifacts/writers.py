from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ArtifactExistsError(FileExistsError):
    """Raised when an artifact exists and overwrite was disabled."""


def write_text_atomic(
    path: str | Path,
    text: str,
    *,
    overwrite: bool = True,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write text to a deterministic artifact path."""

    return write_bytes_atomic(
        path,
        text.encode(encoding),
        overwrite=overwrite,
    )


def write_json_atomic(
    path: str | Path,
    data: Any,
    *,
    overwrite: bool = True,
    indent: int = 2,
) -> Path:
    """Atomically write JSON with stable key ordering."""

    text = json.dumps(data, sort_keys=True, indent=indent)
    return write_text_atomic(path, f"{text}\n", overwrite=overwrite)


def write_bytes_atomic(
    path: str | Path,
    data: bytes,
    *,
    overwrite: bool = True,
) -> Path:
    """Write bytes through a same-directory temp file and atomic publish step."""

    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and final_path.exists():
        raise ArtifactExistsError(f"artifact already exists: {final_path}")

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        dir=final_path.parent,
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        if overwrite:
            os.replace(tmp_path, final_path)
        else:
            try:
                os.link(tmp_path, final_path)
            except FileExistsError as exc:
                raise ArtifactExistsError(
                    f"artifact already exists: {final_path}"
                ) from exc
            finally:
                _unlink_if_present(tmp_path)

        _fsync_directory(final_path.parent)
        return final_path
    except Exception:
        _unlink_if_present(tmp_path)
        raise


def _unlink_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return

    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return

    try:
        os.fsync(fd)
    finally:
        os.close(fd)

