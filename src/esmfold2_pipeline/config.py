from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Literal

import yaml

from esmfold2_pipeline.frameworks import (
    all_scfv_framework_names,
    all_vhh_framework_names,
    get_scfv_framework_record,
    get_vhh_framework_record,
    is_scfv_framework_alias,
    is_vhh_framework_alias,
    resolve_scfv_framework_name,
    resolve_vhh_framework_name,
    scfv_framework_alias_choices,
    vhh_framework_alias_choices,
)
from esmfold2_pipeline.structure import (
    PreparedTarget,
    StructureTargetConfig,
    StructureTargetError,
    parse_structure_target,
    resolve_target_geometry_drift_indices,
)


DEFAULT_ESMFOLD2_INVERSION_MODEL = "ESMFold2-Experimental-Cutoff2025"
DEFAULT_ESMFOLD2_CRITIC_MODEL = "ESMFold2-Experimental-Cutoff2025"
DEFAULT_ESMFOLD2_MODEL = DEFAULT_ESMFOLD2_CRITIC_MODEL
ESMFOLD2_MODEL_ALIASES = {
    "cutoff2025": "ESMFold2-Experimental-Cutoff2025",
    "fast-cutoff2025": "ESMFold2-Experimental-Fast-Cutoff2025",
    "experimental": "ESMFold2-Experimental",
    "fast": "ESMFold2-Experimental-Fast",
}
DEFAULT_HOTSPOT_LOSS_MODE = "entropy_hotspot"
DEFAULT_HOTSPOT_CONTACT_WEIGHT = 2.0
DEFAULT_HOTSPOT_DISTOGRAM_CONTACT_CUTOFF_ANGSTROM = 20.0
DEFAULT_HOTSPOT_CRITIC_CONTACT_CUTOFF_ANGSTROM = 5.0
DEFAULT_TARGET_GEOMETRY_DRIFT_WEIGHT = 2.5
DEFAULT_TARGET_GEOMETRY_DRIFT_TOLERANCE_ANGSTROM = 0.1
DEFAULT_TARGET_GEOMETRY_DRIFT_STIFFNESS_ANGSTROM = 0.1
DEFAULT_ANALYSIS_TOP_K = 25
HOTSPOT_LOSS_MODES = {"entropy_hotspot", "probability_hinge"}
DEFAULT_MINIPROTEIN_LENGTH_RANGE = (60, 200)
MINIPROTEIN_SCAFFOLD = "miniprotein"
SCFV_SCAFFOLD = "scfv"
VHH_SCAFFOLD = "vhh"
SUPPORTED_SCFV_FRAMEWORKS = set(all_scfv_framework_names())
SUPPORTED_VHH_FRAMEWORKS = set(all_vhh_framework_names())
SCFV_CDR_NAMES = ("hcdr1", "hcdr2", "hcdr3", "lcdr1", "lcdr2", "lcdr3")
VHH_CDR_NAMES = ("cdr1", "cdr2", "cdr3")
VHH_CDR_REPORT_NAMES = ("hcdr1", "hcdr2", "hcdr3")
SCFV_FRAMEWORK_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
PDB_CHAIN_ID_CANDIDATES = tuple(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)


@dataclass(frozen=True)
class AntibodyFrameworkSpec:
    name: str
    source: Literal["builtin", "template", "sequence"]
    cdr_names: tuple[str, ...]
    cdr_report_names: tuple[str, ...]
    esm_binder_name: str | None = None
    template: str | None = None
    cdr_lengths: dict[str, tuple[int, int]] | None = None
    sequence: str | None = None
    cdr_ranges: dict[str, tuple[int, int]] | None = None

    @property
    def key(self) -> str:
        return f"framework={self.name}:source={self.source}"

    def to_resolved_value(self) -> str | dict[str, Any]:
        if self.source == "builtin":
            return self.name
        resolved: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
        }
        if self.template is not None:
            resolved["template"] = self.template
        if self.cdr_lengths is not None:
            resolved["cdr_lengths"] = {
                name: (
                    low if low == high else {"min": low, "max": high}
                )
                for name, (low, high) in self.cdr_lengths.items()
            }
        if self.sequence is not None:
            resolved["sequence"] = self.sequence
            resolved["mutate"] = "cdrs"
        if self.cdr_ranges is not None:
            resolved["cdrs"] = {
                name: f"{start + 1}-{end}"
                for name, (start, end) in self.cdr_ranges.items()
            }
        return resolved

    @property
    def cdr_indices(self) -> tuple[int, ...] | None:
        if self.cdr_ranges is None:
            return None
        indices: list[int] = []
        for cdr_name in self.cdr_names:
            start, end = self.cdr_ranges[cdr_name]
            indices.extend(range(start, end))
        return tuple(indices)


ScfvFrameworkSpec = AntibodyFrameworkSpec


@dataclass(frozen=True)
class BinderConfig:
    scaffold: str
    frameworks: tuple[AntibodyFrameworkSpec, ...]
    length_range: tuple[int, int] | None
    esm_binder_name: str

    @property
    def framework(self) -> str | None:
        if len(self.frameworks) == 1:
            return self.frameworks[0].name
        return None

    @property
    def framework_names(self) -> tuple[str, ...]:
        return tuple(framework.name for framework in self.frameworks)

    @property
    def display_name(self) -> str:
        if self.frameworks:
            return f"{self.scaffold}:{','.join(self.framework_names)}"
        return self.scaffold

    @property
    def key(self) -> str:
        parts = [self.scaffold]
        if self.frameworks:
            parts.append(
                "frameworks=" + ",".join(framework.name for framework in self.frameworks)
            )
        if self.length_range is not None:
            low, high = self.length_range
            parts.append(f"length={low}-{high}")
        return "binder:" + ":".join(parts)

    def to_resolved_dict(self) -> dict[str, Any]:
        resolved: dict[str, Any] = {"scaffold": self.scaffold}
        if len(self.frameworks) == 1:
            resolved["framework"] = self.frameworks[0].to_resolved_value()
        elif self.frameworks:
            resolved["frameworks"] = [
                framework.to_resolved_value() for framework in self.frameworks
            ]
        if self.length_range is not None:
            low, high = self.length_range
            resolved["length"] = {"min": low, "max": high}
        return resolved


@dataclass(frozen=True)
class TargetGeometryDriftConfig:
    enabled: bool = False
    weight: float = DEFAULT_TARGET_GEOMETRY_DRIFT_WEIGHT
    tolerance_angstrom: float = DEFAULT_TARGET_GEOMETRY_DRIFT_TOLERANCE_ANGSTROM
    stiffness_angstrom: float = DEFAULT_TARGET_GEOMETRY_DRIFT_STIFFNESS_ANGSTROM
    regions: dict[str, tuple[str, ...]] | None = None

    def to_resolved_dict(self) -> dict[str, Any]:
        resolved: dict[str, Any] = {
            "enabled": self.enabled,
            "weight": self.weight,
            "tolerance_angstrom": self.tolerance_angstrom,
            "stiffness_angstrom": self.stiffness_angstrom,
        }
        if self.regions:
            resolved["regions"] = _selector_map_to_yaml(self.regions)
        return resolved


@dataclass(frozen=True)
class AnalysisConfig:
    top_k: int = DEFAULT_ANALYSIS_TOP_K

    def to_resolved_dict(self) -> dict[str, Any]:
        return {"top_k": self.top_k}


@dataclass(frozen=True)
class CampaignConfig:
    target_name: str
    target_structure: StructureTargetConfig | None
    target_sequence: str | None
    binder: BinderConfig
    seeds: tuple[int, ...]
    inversion_model_name: str
    critic_name: str
    steps: int
    hotspot_contact_weight: float
    hotspot_distogram_contact_cutoff_angstrom: float
    hotspot_critic_contact_cutoff_angstrom: float
    hotspot_num_contacts: int
    hotspot_contact_probability_target: float
    hotspot_loss_mode: str
    target_geometry_drift: TargetGeometryDriftConfig
    analysis: AnalysisConfig
    output: Path

    @property
    def binder_name(self) -> str:
        return self.binder_name_for_design_index(0)

    def binder_framework_for_design_index(self, batch_index: int) -> AntibodyFrameworkSpec | None:
        if batch_index < 0:
            raise ValueError("batch_index must be non-negative")
        if not self.binder.frameworks:
            return None
        return self.binder.frameworks[batch_index % len(self.binder.frameworks)]

    def binder_name_for_design_index(self, batch_index: int) -> str:
        framework = self.binder_framework_for_design_index(batch_index)
        if framework is None:
            return self.binder.esm_binder_name
        return framework.esm_binder_name or framework.name

    def binder_key_for_design_index(self, batch_index: int) -> str:
        framework = self.binder_framework_for_design_index(batch_index)
        if framework is None:
            return self.binder.key
        return f"{self.binder.key}:{framework.key}"

    @property
    def hotspot_contact_cutoff_angstrom(self) -> float:
        return self.hotspot_critic_contact_cutoff_angstrom

    def to_resolved_dict(self) -> dict[str, Any]:
        target: dict[str, Any] = {"name": self.target_name}
        if self.target_structure is not None:
            target.update(
                {
                    "structure": str(self.target_structure.path),
                    "chains": list(self.target_structure.chains),
                    "sequences": dict(self.target_structure.sequences or {}),
                    "structure_indexing": self.target_structure.structure_indexing,
                    "crop": _selector_map_to_yaml(self.target_structure.crop or {}),
                    "hotspots": _selector_map_to_yaml(
                        self.target_structure.hotspots or {}
                    ),
                    "conditioning": {
                        "mode": self.target_structure.conditioning_mode,
                        "assembly": self.target_structure.conditioning_assembly,
                        "chain_pairs": (
                            "auto"
                            if self.target_structure.conditioning_chain_pairs is None
                            else [
                                list(pair)
                                for pair in self.target_structure.conditioning_chain_pairs
                            ]
                        ),
                        "representative_atom": self.target_structure.representative_atom,
                        "partial": self.target_structure.partial_conditioning,
                        "require_resolved": self.target_structure.require_resolved,
                    },
                }
            )
        elif self.target_sequence is not None:
            target["sequence"] = self.target_sequence
        campaign: dict[str, Any] = {
            "num_designs": len(self.seeds),
            "inversion_model": self.inversion_model_name,
            "critics": [self.critic_name],
            "steps": self.steps,
        }
        if self.binder.frameworks:
            campaign["framework_schedule"] = [
                self.binder_framework_for_design_index(index).name  # type: ignore[union-attr]
                for index in range(len(self.seeds))
            ]
        if self.seeds[0] != 0:
            campaign["seed_start"] = self.seeds[0]
        return {
            "target": target,
            "binder": self.binder.to_resolved_dict(),
            "campaign": campaign,
            "loss": {
                "hotspot_contact_weight": self.hotspot_contact_weight,
                "hotspot_distogram_contact_cutoff_angstrom": (
                    self.hotspot_distogram_contact_cutoff_angstrom
                ),
                "hotspot_critic_contact_cutoff_angstrom": (
                    self.hotspot_critic_contact_cutoff_angstrom
                ),
                "hotspot_num_contacts": self.hotspot_num_contacts,
                "hotspot_contact_probability_target": (
                    self.hotspot_contact_probability_target
                ),
                "hotspot_loss_mode": self.hotspot_loss_mode,
                "target_geometry_drift": (
                    self.target_geometry_drift.to_resolved_dict()
                ),
            },
            "analysis": self.analysis.to_resolved_dict(),
            "output": str(self.output),
        }


@dataclass(frozen=True)
class ConfigCheckResult:
    ok: bool
    config_path: Path
    config: CampaignConfig | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    prepared_target: PreparedTarget | None = None


def resolve_esmfold2_model_name(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("model name must be a non-empty string")
    return ESMFOLD2_MODEL_ALIASES.get(text.lower(), text)


def check_campaign_config(
    path: str | Path,
    *,
    output_override: str | Path | None = None,
) -> ConfigCheckResult:
    config_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        config = load_campaign_config(config_path, output_override=output_override)
    except Exception as exc:
        return ConfigCheckResult(
            ok=False,
            config_path=config_path,
            config=None,
            errors=(str(exc),),
        )

    errors.extend(_custom_scfv_sequence_errors(config))
    if config.output.exists() and not config.output.is_dir():
        errors.append(f"output path exists and is not a directory: {config.output}")
    output_writable_error = _output_writable_error(config.output)
    if output_writable_error is not None:
        errors.append(output_writable_error)
    if len(config.seeds) > 100_000:
        warnings.append(
            "campaign expands to more than 100000 designs; planning may take time"
        )
    if config.output.is_dir() and any(config.output.iterdir()):
        warnings.append(f"output directory already exists and is not empty: {config.output}")
    prepared_target: PreparedTarget | None = None
    if config.target_structure is not None:
        try:
            prepared_target = parse_structure_target(config.target_structure)
            warnings.extend(prepared_target.warnings)
            if config.target_geometry_drift.enabled:
                resolve_target_geometry_drift_indices(
                    prepared_target,
                    config.target_geometry_drift.regions,
                    structure_indexing=config.target_structure.structure_indexing,
                    field_name="loss.target_geometry_drift.regions",
                )
            pdb_chain_error = _pdb_export_chain_id_error(prepared_target)
            if pdb_chain_error is not None:
                errors.append(pdb_chain_error)
        except StructureTargetError as exc:
            errors.append(str(exc))

    return ConfigCheckResult(
        ok=not errors,
        config_path=config_path,
        config=config if not errors else None,
        errors=tuple(errors),
        warnings=tuple(warnings),
        prepared_target=prepared_target if not errors else None,
    )


def _custom_scfv_sequence_errors(config: CampaignConfig) -> list[str]:
    errors: list[str] = []
    for framework in config.binder.frameworks:
        if framework.source != "sequence":
            continue
        if not framework.cdr_indices:
            errors.append(
                f"binder framework {framework.name} sequence must define explicit CDR ranges"
            )
    return errors


def load_campaign_config(
    path: str | Path,
    *,
    output_override: str | Path | None = None,
) -> CampaignConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping")

    target = _mapping(raw, "target")
    binder = _mapping(raw, "binder")
    campaign = _mapping(raw, "campaign")
    loss = raw.get("loss", {})
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError("loss must be a mapping")
    analysis_raw = raw.get("analysis", {})
    if analysis_raw is None:
        analysis_raw = {}
    if not isinstance(analysis_raw, dict):
        raise ValueError("analysis must be a mapping")

    target_structure = _parse_target_structure(target, config_path.parent)
    target_sequence = _parse_target_sequence(target.get("sequence"))
    if target_structure is not None and target_sequence is not None:
        raise ValueError("target.structure and target.sequence are mutually exclusive")
    if target_structure is None:
        _reject_structure_only_target_fields(target)
    target_name = _optional_str(target, "name")
    if target_name is None:
        if target_structure is not None:
            target_name = target_structure.path.stem
        elif target_sequence is not None:
            target_name = "sequence_target"
        else:
            raise ValueError(
                "target.name or target.sequence is required unless target.structure is set"
            )
    binder_config = _parse_binder_config(binder)
    seeds = _parse_campaign_seeds(campaign)
    inversion_model_name = _optional_str_with_default(
        campaign,
        "campaign.inversion_model",
        "inversion_model",
        default=DEFAULT_ESMFOLD2_INVERSION_MODEL,
    )
    inversion_model_name = resolve_esmfold2_model_name(inversion_model_name)
    critics = campaign.get("critics", [DEFAULT_ESMFOLD2_CRITIC_MODEL])
    if not isinstance(critics, list) or not critics or not all(isinstance(v, str) and v for v in critics):
        raise ValueError("campaign.critics must be a non-empty list of strings")
    critics = [resolve_esmfold2_model_name(value) for value in critics]
    if len(critics) != 1:
        raise ValueError("single-worker campaign runner currently supports exactly one critic")

    steps = int(campaign.get("steps", 2))
    if steps <= 0:
        raise ValueError("campaign.steps must be positive")

    output_value = output_override if output_override is not None else raw.get("output")
    if output_value is None:
        raise ValueError("output is required unless --out is provided")
    hotspot_contact_weight = _optional_float(
        loss,
        "loss.hotspot_contact_weight",
        "hotspot_contact_weight",
        default=DEFAULT_HOTSPOT_CONTACT_WEIGHT,
    )
    if hotspot_contact_weight < 0:
        raise ValueError("loss.hotspot_contact_weight must be non-negative")
    hotspot_distogram_contact_cutoff_angstrom = _optional_float(
        loss,
        "loss.hotspot_distogram_contact_cutoff_angstrom",
        "hotspot_distogram_contact_cutoff_angstrom",
        default=DEFAULT_HOTSPOT_DISTOGRAM_CONTACT_CUTOFF_ANGSTROM,
    )
    if hotspot_distogram_contact_cutoff_angstrom <= 0:
        raise ValueError(
            "loss.hotspot_distogram_contact_cutoff_angstrom must be positive"
        )
    hotspot_critic_contact_cutoff_angstrom = _optional_float_with_alias(
        loss,
        "loss.hotspot_critic_contact_cutoff_angstrom",
        "hotspot_critic_contact_cutoff_angstrom",
        default=DEFAULT_HOTSPOT_CRITIC_CONTACT_CUTOFF_ANGSTROM,
        aliases=("hotspot_contact_cutoff_angstrom",),
    )
    if hotspot_critic_contact_cutoff_angstrom <= 0:
        raise ValueError("loss.hotspot_critic_contact_cutoff_angstrom must be positive")
    hotspot_num_contacts = _optional_int(
        loss,
        "loss.hotspot_num_contacts",
        "hotspot_num_contacts",
        default=1,
    )
    if hotspot_num_contacts <= 0:
        raise ValueError("loss.hotspot_num_contacts must be positive")
    hotspot_contact_probability_target = _optional_float(
        loss,
        "loss.hotspot_contact_probability_target",
        "hotspot_contact_probability_target",
        default=0.6,
    )
    if not 0 < hotspot_contact_probability_target <= 1:
        raise ValueError(
            "loss.hotspot_contact_probability_target must be greater than 0 and at most 1"
        )
    hotspot_loss_mode = _parse_choice(
        loss.get("hotspot_loss_mode", DEFAULT_HOTSPOT_LOSS_MODE),
        "loss.hotspot_loss_mode",
        HOTSPOT_LOSS_MODES,
    )
    target_geometry_drift = _parse_target_geometry_drift(loss)
    if target_geometry_drift.enabled and target_structure is None:
        raise ValueError("loss.target_geometry_drift requires target.structure")
    analysis = _parse_analysis_config(analysis_raw)

    return CampaignConfig(
        target_name=target_name,
        target_structure=target_structure,
        target_sequence=target_sequence,
        binder=binder_config,
        seeds=seeds,
        inversion_model_name=inversion_model_name,
        critic_name=critics[0],
        steps=steps,
        hotspot_contact_weight=hotspot_contact_weight,
        hotspot_distogram_contact_cutoff_angstrom=(
            hotspot_distogram_contact_cutoff_angstrom
        ),
        hotspot_critic_contact_cutoff_angstrom=hotspot_critic_contact_cutoff_angstrom,
        hotspot_num_contacts=hotspot_num_contacts,
        hotspot_contact_probability_target=hotspot_contact_probability_target,
        hotspot_loss_mode=hotspot_loss_mode,
        target_geometry_drift=target_geometry_drift,
        analysis=analysis,
        output=Path(output_value).expanduser(),
    )


def _mapping(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_str(raw: dict[str, Any], dotted_name: str, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{dotted_name} must be a non-empty string")
    return value


def _parse_binder_config(raw: dict[str, Any]) -> BinderConfig:
    if "scaffold" not in raw:
        raise ValueError("binder.scaffold is required")

    scaffold = _parse_scaffold(raw["scaffold"], "binder.scaffold")
    framework = raw.get("framework")
    frameworks = raw.get("frameworks")
    length_range = _parse_binder_length(
        raw.get("length"),
        default=DEFAULT_MINIPROTEIN_LENGTH_RANGE,
    )
    if scaffold == MINIPROTEIN_SCAFFOLD:
        if framework is not None or frameworks is not None:
            raise ValueError("binder.framework/frameworks are only valid for antibody scaffolds")
        return BinderConfig(
            scaffold=MINIPROTEIN_SCAFFOLD,
            frameworks=(),
            length_range=length_range,
            esm_binder_name="minibinder",
        )

    if scaffold in {SCFV_SCAFFOLD, VHH_SCAFFOLD}:
        if raw.get("length") is not None:
            raise ValueError("binder.length is currently only valid for miniproteins")
        if framework is not None and frameworks is not None:
            raise ValueError("use binder.framework or binder.frameworks, not both")
        parsed_frameworks = _parse_antibody_frameworks(scaffold, framework, frameworks)
        return BinderConfig(
            scaffold=scaffold,
            frameworks=parsed_frameworks,
            length_range=None,
            esm_binder_name=parsed_frameworks[0].esm_binder_name or parsed_frameworks[0].name,
        )

    raise AssertionError(f"unhandled binder scaffold: {scaffold}")


def _parse_antibody_frameworks(
    scaffold: str,
    framework: Any,
    frameworks: Any,
) -> tuple[AntibodyFrameworkSpec, ...]:
    if framework is None and frameworks is None:
        raise ValueError(
            "binder.framework or binder.frameworks is required when "
            f"binder.scaffold is {scaffold}"
        )
    if frameworks is not None:
        if not isinstance(frameworks, list) or not frameworks:
            raise ValueError("binder.frameworks must be a non-empty list")
        parsed = tuple(
            _parse_antibody_framework_spec(scaffold, item, f"binder.frameworks[{index}]")
            for index, item in enumerate(frameworks)
        )
    else:
        parsed = (_parse_antibody_framework_spec(scaffold, framework, "binder.framework"),)

    names = [item.name for item in parsed]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"binder.frameworks contains duplicate framework names: {joined}")
    return parsed


def _parse_antibody_framework_spec(
    scaffold: str,
    value: Any,
    field_name: str,
) -> AntibodyFrameworkSpec:
    cdr_names = _framework_cdr_names(scaffold)
    cdr_report_names = _framework_cdr_report_names(scaffold)
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{field_name} must be a non-empty framework name")
        canonical_name = _resolve_framework_name(scaffold, value)
        if canonical_name is None:
            choices = ", ".join(_framework_alias_choices(scaffold))
            raise ValueError(f"{field_name} must be one of: {choices}")
        record = _get_framework_record(scaffold, canonical_name)
        return AntibodyFrameworkSpec(
            name=record.canonical_name,
            source="builtin",
            cdr_names=cdr_names,
            cdr_report_names=cdr_report_names,
            esm_binder_name=record.canonical_name,
            template=record.template,
            cdr_lengths=dict(record.cdr_lengths),
        )

    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a framework name or mapping")
    name = _framework_name(value.get("name"), f"{field_name}.name")
    if _is_framework_alias(scaffold, name):
        raise ValueError(f"{field_name}.name must not collide with a built-in framework")

    has_template = value.get("template") is not None
    has_sequence = value.get("sequence") is not None
    if has_template == has_sequence:
        raise ValueError(f"{field_name} must define exactly one of template or sequence")

    if has_template:
        template = _parse_antibody_template(
            value.get("template"),
            f"{field_name}.template",
            cdr_names=cdr_names,
        )
        cdr_lengths = _parse_cdr_lengths(
            value.get("cdr_lengths"),
            f"{field_name}.cdr_lengths",
            cdr_names=cdr_names,
        )
        return AntibodyFrameworkSpec(
            name=name,
            source="template",
            cdr_names=cdr_names,
            cdr_report_names=cdr_report_names,
            esm_binder_name=None,
            template=template,
            cdr_lengths=cdr_lengths,
        )

    mutate = value.get("mutate", "cdrs")
    if mutate != "cdrs":
        raise ValueError(f"{field_name}.mutate must be cdrs")
    sequence = _parse_protein_sequence(
        value.get("sequence"),
        f"{field_name}.sequence",
        allow_chain_breaks=False,
    )
    cdr_ranges = _parse_cdr_ranges(
        value.get("cdrs"),
        f"{field_name}.cdrs",
        sequence_length=len(sequence),
        cdr_names=cdr_names,
        scaffold=scaffold,
    )
    return AntibodyFrameworkSpec(
        name=name,
        source="sequence",
        cdr_names=cdr_names,
        cdr_report_names=cdr_report_names,
        esm_binder_name=None,
        sequence=sequence,
        cdr_ranges=cdr_ranges,
    )


def _framework_name(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if not SCFV_FRAMEWORK_NAME_RE.match(value):
        raise ValueError(
            f"{field_name} must start with a letter and contain only letters, numbers, '_' or '-'"
        )
    return value


def _framework_cdr_names(scaffold: str) -> tuple[str, ...]:
    if scaffold == SCFV_SCAFFOLD:
        return SCFV_CDR_NAMES
    if scaffold == VHH_SCAFFOLD:
        return VHH_CDR_NAMES
    raise AssertionError(f"unhandled antibody scaffold: {scaffold}")


def _framework_cdr_report_names(scaffold: str) -> tuple[str, ...]:
    if scaffold == SCFV_SCAFFOLD:
        return SCFV_CDR_NAMES
    if scaffold == VHH_SCAFFOLD:
        return VHH_CDR_REPORT_NAMES
    raise AssertionError(f"unhandled antibody scaffold: {scaffold}")


def _resolve_framework_name(scaffold: str, name: str) -> str | None:
    if scaffold == SCFV_SCAFFOLD:
        return resolve_scfv_framework_name(name)
    if scaffold == VHH_SCAFFOLD:
        return resolve_vhh_framework_name(name)
    raise AssertionError(f"unhandled antibody scaffold: {scaffold}")


def _is_framework_alias(scaffold: str, name: str) -> bool:
    if scaffold == SCFV_SCAFFOLD:
        return is_scfv_framework_alias(name)
    if scaffold == VHH_SCAFFOLD:
        return is_vhh_framework_alias(name)
    raise AssertionError(f"unhandled antibody scaffold: {scaffold}")


def _framework_alias_choices(scaffold: str) -> tuple[str, ...]:
    if scaffold == SCFV_SCAFFOLD:
        return scfv_framework_alias_choices()
    if scaffold == VHH_SCAFFOLD:
        return vhh_framework_alias_choices()
    raise AssertionError(f"unhandled antibody scaffold: {scaffold}")


def _get_framework_record(scaffold: str, name: str):
    if scaffold == SCFV_SCAFFOLD:
        return get_scfv_framework_record(name)
    if scaffold == VHH_SCAFFOLD:
        return get_vhh_framework_record(name)
    raise AssertionError(f"unhandled antibody scaffold: {scaffold}")


def _parse_antibody_template(
    value: Any,
    field_name: str,
    *,
    cdr_names: tuple[str, ...],
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    placeholders = re.findall(r"{([^{}]+)}", value)
    unknown = sorted(set(placeholders) - set(cdr_names))
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"{field_name} contains unsupported placeholders: {joined}")
    for cdr_name in cdr_names:
        count = placeholders.count(cdr_name)
        if count != 1:
            raise ValueError(
                f"{field_name} must contain placeholder {{{cdr_name}}} exactly once"
            )
    fixed_sequence = value
    for cdr_name in cdr_names:
        fixed_sequence = fixed_sequence.replace(f"{{{cdr_name}}}", "")
    if "{" in fixed_sequence or "}" in fixed_sequence:
        raise ValueError(f"{field_name} contains malformed placeholders")
    _parse_protein_sequence(fixed_sequence, field_name, allow_chain_breaks=False)
    return value


def _parse_cdr_lengths(
    value: Any,
    field_name: str,
    *,
    cdr_names: tuple[str, ...],
) -> dict[str, tuple[int, int]]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    parsed: dict[str, tuple[int, int]] = {}
    missing = [cdr_name for cdr_name in cdr_names if cdr_name not in value]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{field_name} is missing CDR lengths: {joined}")
    unknown = sorted(set(value) - set(cdr_names))
    if unknown:
        joined = ", ".join(str(item) for item in unknown)
        raise ValueError(f"{field_name} contains unsupported CDRs: {joined}")
    for cdr_name in cdr_names:
        parsed[cdr_name] = _parse_binder_length(
            value[cdr_name],
            default=(1, 1),
            field_name=f"{field_name}.{cdr_name}",
        )
    return parsed


def _parse_cdr_ranges(
    value: Any,
    field_name: str,
    *,
    sequence_length: int,
    cdr_names: tuple[str, ...],
    scaffold: str,
) -> dict[str, tuple[int, int]]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping of 1-based inclusive ranges")
    parsed: dict[str, tuple[int, int]] = {}
    missing = [cdr_name for cdr_name in cdr_names if cdr_name not in value]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{field_name} is missing CDR ranges: {joined}")
    unknown = sorted(set(value) - set(cdr_names))
    if unknown:
        joined = ", ".join(str(item) for item in unknown)
        raise ValueError(f"{field_name} contains unsupported CDRs: {joined}")
    for cdr_name in cdr_names:
        parsed[cdr_name] = _parse_cdr_range(
            value[cdr_name],
            f"{field_name}.{cdr_name}",
            sequence_length=sequence_length,
        )

    starts = [parsed[cdr_name][0] for cdr_name in cdr_names]
    if starts != sorted(starts):
        raise ValueError(
            f"{field_name} ranges must appear in {scaffold} sequence order"
        )

    occupied: dict[int, str] = {}
    for cdr_name in cdr_names:
        start, end = parsed[cdr_name]
        for index in range(start, end):
            existing = occupied.get(index)
            if existing is not None:
                raise ValueError(
                    f"{field_name}.{cdr_name} overlaps {existing} at position {index + 1}"
                )
            occupied[index] = cdr_name
    return parsed


def _parse_cdr_range(
    value: Any,
    field_name: str,
    *,
    sequence_length: int,
) -> tuple[int, int]:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a 1-based inclusive range")
    if isinstance(value, int):
        start = end = value
    elif isinstance(value, str):
        start, end = _parse_length_string(value, field_name=field_name)
    elif isinstance(value, list) and len(value) == 2:
        start = _coerce_length_int(value[0], f"{field_name}[0]")
        end = _coerce_length_int(value[1], f"{field_name}[1]")
    elif isinstance(value, dict):
        start = _coerce_length_int(value.get("start"), f"{field_name}.start")
        end = _coerce_length_int(value.get("end"), f"{field_name}.end")
    else:
        raise ValueError(
            f"{field_name} must be a range string, [start, end], or {{start, end}} mapping"
        )
    if start <= 0 or end <= 0:
        raise ValueError(f"{field_name} positions must be positive")
    if end < start:
        raise ValueError(f"{field_name} end must be >= start")
    if end > sequence_length:
        raise ValueError(
            f"{field_name} end {end} is outside sequence length {sequence_length}"
        )
    return (start - 1, end)


def _parse_scaffold(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in {MINIPROTEIN_SCAFFOLD, SCFV_SCAFFOLD, VHH_SCAFFOLD}:
        raise ValueError(
            f"{field_name} must be one of: "
            f"{MINIPROTEIN_SCAFFOLD}, {SCFV_SCAFFOLD}, {VHH_SCAFFOLD}"
        )
    return normalized


def _parse_binder_length(
    value: Any,
    *,
    default: tuple[int, int],
    field_name: str = "binder.length",
) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer or range")
    if isinstance(value, int):
        low = high = value
    elif isinstance(value, str):
        low, high = _parse_length_string(value, field_name=field_name)
    elif isinstance(value, list) and len(value) == 2:
        low, high = _coerce_length_int(value[0], f"{field_name}[0]"), _coerce_length_int(
            value[1],
            f"{field_name}[1]",
        )
    elif isinstance(value, dict):
        low = _coerce_length_int(value.get("min"), f"{field_name}.min")
        high = _coerce_length_int(value.get("max"), f"{field_name}.max")
    else:
        raise ValueError(
            f"{field_name} must be an integer, range string, [min, max], or mapping"
        )
    if low <= 0 or high <= 0:
        raise ValueError(f"{field_name} values must be positive")
    if high < low:
        raise ValueError(f"{field_name} max must be >= min")
    return (low, high)


def _parse_length_string(value: str, *, field_name: str) -> tuple[int, int]:
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    if "-" not in text:
        length = _coerce_length_int(text, field_name)
        return (length, length)
    low_text, high_text = text.split("-", 1)
    low = _coerce_length_int(low_text.strip(), f"{field_name} min")
    high = _coerce_length_int(high_text.strip(), f"{field_name} max")
    return (low, high)


def _coerce_length_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _optional_str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"target.{key} must be a non-empty string")
    return value


def _optional_str_with_default(
    raw: dict[str, Any],
    dotted_name: str,
    key: str,
    *,
    default: str,
) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{dotted_name} must be a non-empty string")
    return value


def _optional_float(
    raw: dict[str, Any],
    dotted_name: str,
    key: str,
    *,
    default: float,
) -> float:
    value = raw.get(key, default)
    return _coerce_float(value, dotted_name)


def _optional_float_with_alias(
    raw: dict[str, Any],
    dotted_name: str,
    key: str,
    *,
    default: float,
    aliases: tuple[str, ...],
) -> float:
    if key in raw:
        return _coerce_float(raw[key], dotted_name)
    for alias in aliases:
        if alias in raw:
            return _coerce_float(raw[alias], f"loss.{alias}")
    return float(default)


def _parse_target_geometry_drift(raw_loss: dict[str, Any]) -> TargetGeometryDriftConfig:
    raw = raw_loss.get("target_geometry_drift")
    if raw is None:
        return TargetGeometryDriftConfig()
    if not isinstance(raw, dict):
        raise ValueError("loss.target_geometry_drift must be a mapping")

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("loss.target_geometry_drift.enabled must be true or false")

    weight = _coerce_float(
        raw.get("weight", DEFAULT_TARGET_GEOMETRY_DRIFT_WEIGHT),
        "loss.target_geometry_drift.weight",
    )
    if weight < 0:
        raise ValueError("loss.target_geometry_drift.weight must be non-negative")

    tolerance_angstrom = _coerce_float(
        raw.get(
            "tolerance_angstrom",
            DEFAULT_TARGET_GEOMETRY_DRIFT_TOLERANCE_ANGSTROM,
        ),
        "loss.target_geometry_drift.tolerance_angstrom",
    )
    if tolerance_angstrom <= 0:
        raise ValueError(
            "loss.target_geometry_drift.tolerance_angstrom must be positive"
        )

    stiffness_angstrom = _coerce_float(
        raw.get(
            "stiffness_angstrom",
            DEFAULT_TARGET_GEOMETRY_DRIFT_STIFFNESS_ANGSTROM,
        ),
        "loss.target_geometry_drift.stiffness_angstrom",
    )
    if stiffness_angstrom <= 0:
        raise ValueError(
            "loss.target_geometry_drift.stiffness_angstrom must be positive"
        )

    regions = raw.get("regions")
    if regions is None or (isinstance(regions, dict) and not regions):
        parsed_regions = None
    else:
        parsed_regions = _parse_selector_map(
            regions,
            "loss.target_geometry_drift.regions",
        )

    return TargetGeometryDriftConfig(
        enabled=enabled,
        weight=weight,
        tolerance_angstrom=tolerance_angstrom,
        stiffness_angstrom=stiffness_angstrom,
        regions=parsed_regions,
    )


def _parse_analysis_config(raw: dict[str, Any]) -> AnalysisConfig:
    top_k = _optional_int(
        raw,
        "analysis.top_k",
        "top_k",
        default=DEFAULT_ANALYSIS_TOP_K,
    )
    if top_k <= 0:
        raise ValueError("analysis.top_k must be positive")
    return AnalysisConfig(top_k=top_k)


def _coerce_float(value: Any, dotted_name: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise ValueError(f"{dotted_name} must be a number")
    return float(value)


def _optional_int(
    raw: dict[str, Any],
    dotted_name: str,
    key: str,
    *,
    default: int,
) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{dotted_name} must be an integer")
    return value


def _parse_target_structure(
    target: dict[str, Any],
    config_dir: Path,
) -> StructureTargetConfig | None:
    structure = target.get("structure")
    if structure is None:
        return None

    if isinstance(structure, str):
        structure_path = structure
        structure_options: dict[str, Any] = {}
    elif isinstance(structure, dict):
        structure_options = structure
        path_value = structure.get("path") or structure.get("file")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("target.structure.path must be a non-empty string")
        structure_path = path_value
    else:
        raise ValueError("target.structure must be a path string or mapping")

    path = Path(structure_path).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    path = path.resolve()

    chains = _parse_str_tuple(
        target.get("chains", structure_options.get("chains")),
        "target.chains",
        allow_empty=True,
    )
    structure_indexing = _parse_choice(
        target.get("structure_indexing", structure_options.get("structure_indexing", "auto")),
        "target.structure_indexing",
        {"auto", "label_seq_id", "auth_seq_id"},
    )
    conditioning = target.get("conditioning", structure_options.get("conditioning", {}))
    if conditioning is None:
        conditioning = {}
    if not isinstance(conditioning, dict):
        raise ValueError("target.conditioning must be a mapping")
    conditioning_mode = _parse_choice(
        conditioning.get("mode", "none"),
        "target.conditioning.mode",
        {"none", "distogram"},
    )
    conditioning_assembly = conditioning.get(
        "assembly",
        conditioning_mode == "distogram" and len(chains) > 1,
    )
    if not isinstance(conditioning_assembly, bool):
        raise ValueError("target.conditioning.assembly must be true or false")
    if conditioning_assembly and conditioning_mode != "distogram":
        raise ValueError(
            "target.conditioning.assembly requires target.conditioning.mode: distogram"
        )
    conditioning_chain_pairs = _parse_conditioning_chain_pairs(
        conditioning.get("chain_pairs", "auto"),
        "target.conditioning.chain_pairs",
    )
    representative_atom = _parse_choice(
        conditioning.get("representative_atom", "esmfold2_default"),
        "target.conditioning.representative_atom",
        {"esmfold2_default"},
    )
    partial_conditioning = conditioning.get("partial", True)
    if not isinstance(partial_conditioning, bool):
        raise ValueError("target.conditioning.partial must be true or false")
    require_resolved = conditioning.get("require_resolved", False)
    if not isinstance(require_resolved, bool):
        raise ValueError("target.conditioning.require_resolved must be true or false")

    return StructureTargetConfig(
        path=path,
        chains=chains,
        structure_indexing=structure_indexing,
        sequences=_parse_structure_sequence_map(
            structure_options.get("sequences", target.get("structure_sequences")),
            "target.structure.sequences",
        ),
        crop=_parse_selector_map(target.get("crop", structure_options.get("crop")), "target.crop"),
        hotspots=_parse_selector_map(
            target.get("hotspots", structure_options.get("hotspots")),
            "target.hotspots",
            allow_inline_chain_selectors=True,
        ),
        conditioning_mode=conditioning_mode,
        conditioning_assembly=conditioning_assembly,
        conditioning_chain_pairs=conditioning_chain_pairs,
        partial_conditioning=partial_conditioning,
        representative_atom=representative_atom,
        require_resolved=require_resolved,
    )


def _reject_structure_only_target_fields(target: dict[str, Any]) -> None:
    forbidden = [
        "chains",
        "structure_indexing",
        "crop",
        "hotspots",
        "conditioning",
        "structure_sequences",
    ]
    present = [key for key in forbidden if key in target]
    if present:
        joined = ", ".join(f"target.{key}" for key in present)
        raise ValueError(f"{joined} require target.structure")


def _parse_target_sequence(value: Any) -> str | None:
    if value is None:
        return None
    return _parse_protein_sequence(
        value,
        "target.sequence",
        allow_chain_breaks=False,
        chain_break_error=(
            "target.sequence currently supports one target chain; use target.structure "
            "for multichain targets"
        ),
    )


def _parse_structure_sequence_map(
    value: Any,
    field_name: str,
) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping of chain id to sequence")
    parsed: dict[str, str] = {}
    for key, raw_sequence in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty chain ids")
        parsed[key] = _parse_protein_sequence(
            raw_sequence,
            f"{field_name}.{key}",
            allow_chain_breaks=False,
        )
    return parsed


def _parse_protein_sequence(
    value: Any,
    field_name: str,
    *,
    allow_chain_breaks: bool,
    chain_break_error: str | None = None,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a protein sequence string")
    sequence = "".join(value.split()).upper()
    if not sequence:
        raise ValueError(f"{field_name} must be a non-empty protein sequence")
    if "|" in sequence and not allow_chain_breaks:
        raise ValueError(chain_break_error or f"{field_name} must not contain chain breaks")
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    if allow_chain_breaks:
        allowed.add("|")
    invalid = sorted(set(sequence) - allowed)
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"{field_name} contains unsupported residues: {joined}")
    return sequence


def _output_writable_error(path: Path) -> str | None:
    candidate = path if path.exists() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        return f"output parent does not exist: {path.parent}"
    if not candidate.is_dir():
        return f"output parent is not a directory: {candidate}"
    if not os.access(candidate, os.W_OK | os.X_OK):
        return f"output parent is not writable: {candidate}"
    return None


def _pdb_export_chain_id_error(prepared_target: PreparedTarget) -> str | None:
    target_chain_ids = [chain.canonical_chain_id for chain in prepared_target.chains]
    binder_chain_id = _first_available_pdb_chain_id(target_chain_ids)
    if binder_chain_id is None:
        joined = ", ".join(target_chain_ids)
        return (
            "PDB structure export requires room for a unique one-character binder "
            f"chain ID; selected target chains use all available IDs: {joined}"
        )

    chain_ids = [*target_chain_ids, binder_chain_id]
    invalid = [chain_id for chain_id in chain_ids if not _is_pdb_chain_id(chain_id)]
    if invalid:
        joined = ", ".join(chain_ids)
        return (
            "PDB structure export currently requires selected target chain IDs and "
            "the auto-assigned binder chain ID to be unique one-character IDs; "
            f"got: {joined}"
        )
    if len(set(chain_ids)) != len(chain_ids):
        joined = ", ".join(chain_ids)
        return (
            "PDB structure export requires unique selected target and binder chain "
            f"IDs; got: {joined}"
        )
    return None


def _first_available_pdb_chain_id(target_chain_ids: list[str]) -> str | None:
    used = set(target_chain_ids)
    for candidate in PDB_CHAIN_ID_CANDIDATES:
        if candidate not in used:
            return candidate
    return None


def _is_pdb_chain_id(chain_id: str) -> bool:
    return len(chain_id) == 1 and not chain_id.isspace()


def _parse_choice(value: Any, field_name: str, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {choices}")
    return value


def _parse_str_tuple(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if value is None:
        if allow_empty:
            return ()
        raise ValueError(f"{field_name} is required")
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    return tuple(value)


def _parse_conditioning_chain_pairs(
    value: Any,
    field_name: str,
) -> tuple[tuple[str, str], ...] | None:
    if value in (None, "auto"):
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be 'auto' or a list of chain pairs")
    pairs: list[tuple[str, str]] = []
    for index, raw_pair in enumerate(value):
        item_name = f"{field_name}[{index}]"
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
            or not all(isinstance(chain_id, str) and chain_id for chain_id in raw_pair)
        ):
            raise ValueError(f"{item_name} must be a two-item list of chain ids")
        pairs.append((raw_pair[0], raw_pair[1]))
    return tuple(pairs)


def _parse_selector_map(
    value: Any,
    field_name: str,
    *,
    allow_inline_chain_selectors: bool = False,
) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if allow_inline_chain_selectors and isinstance(value, str):
        return _parse_inline_chain_selector_map(
            _parse_selector_values(value, field_name),
            field_name,
        )
    if isinstance(value, list):
        selectors = _parse_selector_values(value, field_name)
        if allow_inline_chain_selectors and _has_inline_chain_selector(selectors):
            return _parse_inline_chain_selector_map(selectors, field_name)
        return {"*": selectors}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a list or mapping")
    parsed: dict[str, tuple[str, ...]] = {}
    for key, raw_values in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty chain ids")
        parsed[key] = _parse_selector_values(raw_values, f"{field_name}.{key}")
    return parsed


def _parse_selector_values(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, int)):
        raw_values = (str(value),)
    elif isinstance(value, list) and value:
        raw_values = tuple(str(item) for item in value)
    else:
        raise ValueError(f"{field_name} must be a non-empty selector or list")

    selectors: list[str] = []
    for raw_value in raw_values:
        for part in raw_value.replace(";", ",").split(","):
            selector = part.strip()
            if not selector:
                raise ValueError(f"{field_name} contains an empty selector")
            selectors.append(selector)
    return tuple(selectors)


def _has_inline_chain_selector(selectors: tuple[str, ...]) -> bool:
    return any(_looks_like_inline_chain_selector(selector) for selector in selectors)


def _looks_like_inline_chain_selector(selector: str) -> bool:
    text = selector.strip()
    if ":" in text:
        chain, residue_selector = text.split(":", 1)
        return bool(chain.strip() and residue_selector.strip())
    return len(text) > 1 and text[0].isalpha()


def _parse_inline_chain_selector_map(
    selectors: tuple[str, ...],
    field_name: str,
) -> dict[str, tuple[str, ...]]:
    parsed: dict[str, list[str]] = {}
    current_chain: str | None = None
    for selector in selectors:
        chain_id, residue_selector, chain_was_explicit = _split_inline_chain_selector(
            selector,
            current_chain=current_chain,
            field_name=field_name,
        )
        if chain_was_explicit:
            current_chain = chain_id
        parsed.setdefault(chain_id, []).append(residue_selector)

    return {chain_id: tuple(values) for chain_id, values in parsed.items()}


def _split_inline_chain_selector(
    selector: str,
    *,
    current_chain: str | None,
    field_name: str,
) -> tuple[str, str, bool]:
    text = selector.strip()
    if ":" in text:
        chain_text, residue_text = text.split(":", 1)
        chain_id = chain_text.strip()
        residue_selector = residue_text.strip()
        if not chain_id or not residue_selector:
            raise ValueError(
                f"{field_name} selector {selector!r} must include both chain and residue"
            )
        return chain_id, residue_selector, True

    if text and text[0].isalpha():
        residue_selector = text[1:].strip()
        if not residue_selector:
            raise ValueError(
                f"{field_name} selector {selector!r} must include residue numbers"
            )
        return text[0], residue_selector, True

    if current_chain is None:
        raise ValueError(
            f"{field_name} string selectors must begin with a chain-qualified "
            "selector such as A88 or A:88"
        )
    return current_chain, text, False


def _selector_map_to_yaml(value: dict[str, tuple[str, ...]]) -> Any:
    if not value:
        return {}
    if set(value) == {"*"}:
        return list(value["*"])
    return {key: list(selectors) for key, selectors in value.items()}


def _parse_campaign_seeds(campaign: dict[str, Any]) -> tuple[int, ...]:
    has_num_designs = "num_designs" in campaign
    has_seed_start = "seed_start" in campaign
    if has_num_designs:
        num_designs = _coerce_non_negative_int(
            campaign["num_designs"],
            "campaign.num_designs",
        )
        if num_designs <= 0:
            raise ValueError("campaign.num_designs must be positive")
        seed_start = _coerce_non_negative_int(
            campaign.get("seed_start", 0),
            "campaign.seed_start",
        )
        return tuple(range(seed_start, seed_start + num_designs))
    if has_seed_start:
        raise ValueError("campaign.seed_start requires campaign.num_designs")
    raise ValueError("campaign.num_designs is required")


def _coerce_non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value
