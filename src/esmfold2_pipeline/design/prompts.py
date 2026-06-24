from __future__ import annotations

import random
from typing import Any, Mapping

from esmfold2_pipeline.design.spec import BinderPromptPlan
from esmfold2_pipeline.planning import binder_code


ANTIBODY_BINDER_CODES = {"scfv", "vhh"}
SCFV_CDR_RUN_NAMES = ("hcdr1", "hcdr2", "hcdr3", "lcdr1", "lcdr2", "lcdr3")
VHH_CDR_REPORT_NAMES = ("hcdr1", "hcdr2", "hcdr3")
MINIPROTEIN_DEFAULT_LENGTH_RANGE = (60, 200)


def prepare_binder_prompt_plan(
    *,
    binder_name: str,
    binder_scaffold: str | None,
    binder_framework_name: str | None,
    binder_framework_source: str | None,
    binder_framework_template: str | None,
    binder_framework_cdr_lengths: dict[str, tuple[int, int]] | None,
    binder_framework_sequence: str | None,
    binder_framework_cdr_indices: tuple[int, ...] | None,
    seed: int,
    is_antibody: bool | None,
    binder_length_range: tuple[int, int] | None = None,
    local_miniprotein: bool = False,
    binder_prompt_factories: Mapping[str, Any] | None = None,
) -> BinderPromptPlan:
    resolved_binder_code = binder_code(binder_scaffold or binder_name)
    if resolved_binder_code not in ANTIBODY_BINDER_CODES:
        if local_miniprotein:
            return BinderPromptPlan(
                binder_name=None,
                binder_sequence=sample_miniprotein_prompt(
                    binder_length_range,
                    seed=seed,
                ),
                is_antibody=False if is_antibody is None else is_antibody,
                cdr_indices=(),
                cdr_lengths={},
                cdr_report_names=(),
            )
        return BinderPromptPlan(
            binder_name=binder_name,
            binder_sequence=None,
            is_antibody=is_antibody,
            cdr_indices=(),
            cdr_lengths={},
            cdr_report_names=(),
        )

    cdr_report_names = antibody_cdr_report_names(resolved_binder_code)
    source = binder_framework_source or "builtin"
    if source == "builtin":
        if (
            binder_framework_template is not None
            and binder_framework_cdr_lengths is not None
        ):
            prompt = sample_antibody_template(
                binder_framework_template,
                binder_framework_cdr_lengths,
                seed=seed,
            )
        else:
            if resolved_binder_code == "vhh":
                raise ValueError(
                    f"VHH framework {binder_framework_name or binder_name} "
                    "requires a bundled template"
                )
            if (
                binder_prompt_factories is None
                or binder_name not in binder_prompt_factories
            ):
                raise ValueError(
                    f"scFv framework not found in ESM binder factories: {binder_name}"
                )
            prompt = binder_prompt_factories[binder_name].sample(seed=seed)
    elif source == "template":
        if binder_framework_template is None or binder_framework_cdr_lengths is None:
            raise ValueError(
                f"custom {resolved_binder_code} template framework requires "
                "template and CDR lengths"
            )
        prompt = sample_antibody_template(
            binder_framework_template,
            binder_framework_cdr_lengths,
            seed=seed,
        )
    elif source == "sequence":
        if binder_framework_sequence is None:
            raise ValueError(
                f"custom {resolved_binder_code} sequence framework requires sequence"
            )
        if binder_framework_cdr_indices is None:
            raise ValueError(
                f"custom {resolved_binder_code} sequence framework requires "
                "explicit CDR indices"
            )
        prompt = cdr_prompt_from_indices(
            binder_framework_sequence,
            binder_framework_cdr_indices,
        )
    else:
        raise ValueError(f"unsupported {resolved_binder_code} framework source: {source}")

    cdr_indices = tuple(index for index, residue in enumerate(prompt) if residue == "#")
    if not cdr_indices:
        raise ValueError(
            f"{resolved_binder_code} framework {binder_framework_name or binder_name} "
            "has no mutable CDR positions"
        )
    return BinderPromptPlan(
        binder_name=None,
        binder_sequence=prompt,
        is_antibody=True,
        cdr_indices=cdr_indices,
        cdr_lengths=mutable_run_lengths(prompt, cdr_names=cdr_report_names),
        cdr_report_names=tuple(
            cdr_report_names[: len(contiguous_mutable_runs(prompt))]
        ),
    )


def sample_miniprotein_prompt(
    length_range: tuple[int, int] | None,
    *,
    seed: int,
) -> str:
    low, high = length_range or MINIPROTEIN_DEFAULT_LENGTH_RANGE
    rng = random.Random(seed)
    return "#" * rng.randint(low, high)


def sample_antibody_template(
    template: str,
    cdr_lengths: dict[str, tuple[int, int]],
    *,
    seed: int,
) -> str:
    rng = random.Random(seed)
    replacements = {
        name: "#" * rng.randint(low, high)
        for name, (low, high) in cdr_lengths.items()
    }
    return template.format(**replacements)


def sample_scfv_template(
    template: str,
    cdr_lengths: dict[str, tuple[int, int]],
    *,
    seed: int,
) -> str:
    return sample_antibody_template(template, cdr_lengths, seed=seed)


def antibody_cdr_report_names(binder_type: str) -> tuple[str, ...]:
    if binder_type == "scfv":
        return SCFV_CDR_RUN_NAMES
    if binder_type == "vhh":
        return VHH_CDR_REPORT_NAMES
    return ()


def mutable_run_lengths(
    prompt: str,
    *,
    cdr_names: tuple[str, ...] = SCFV_CDR_RUN_NAMES,
) -> dict[str, int]:
    lengths = [end - start for start, end in contiguous_mutable_runs(prompt)]
    return {
        name: length
        for name, length in zip(cdr_names, lengths)
    }


def mutable_run_sequences(
    sequence: str,
    cdr_indices: tuple[int, ...],
    *,
    cdr_names: tuple[str, ...] = SCFV_CDR_RUN_NAMES,
) -> dict[str, str]:
    if not cdr_indices:
        return {}
    runs = contiguous_index_runs(cdr_indices)
    return {
        name: sequence[start:end]
        for name, (start, end) in zip(cdr_names, runs)
        if end <= len(sequence)
    }


def contiguous_mutable_runs(prompt: str) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(prompt):
        if prompt[index] != "#":
            index += 1
            continue
        start = index
        while index < len(prompt) and prompt[index] == "#":
            index += 1
        runs.append((start, index))
    return runs


def contiguous_index_runs(indices: tuple[int, ...]) -> list[tuple[int, int]]:
    if not indices:
        return []
    sorted_indices = sorted(set(indices))
    runs: list[tuple[int, int]] = []
    start = sorted_indices[0]
    previous = start
    for index in sorted_indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        runs.append((start, previous + 1))
        start = previous = index
    runs.append((start, previous + 1))
    return runs


def cdr_prompt_from_indices(
    sequence: str,
    cdr_indices: tuple[int, ...],
) -> str:
    cdr_index_set = set(cdr_indices)
    return "".join(
        "#" if index in cdr_index_set else residue
        for index, residue in enumerate(sequence)
    )


def scfv_cdr_prompt_from_indices(
    sequence: str,
    cdr_indices: tuple[int, ...],
) -> str:
    return cdr_prompt_from_indices(sequence, cdr_indices)
