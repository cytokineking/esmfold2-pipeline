"""Bundled binder framework registry."""

from esmfold2_pipeline.frameworks.registry import (
    FrameworkRecord,
    all_scfv_framework_names,
    all_vhh_framework_names,
    get_scfv_framework_record,
    get_scfv_framework_template_cif,
    get_vhh_framework_record,
    get_vhh_framework_template_cif,
    is_scfv_framework_alias,
    is_vhh_framework_alias,
    resolve_scfv_framework_name,
    resolve_vhh_framework_name,
    scfv_framework_alias_choices,
    vhh_framework_alias_choices,
)

__all__ = [
    "FrameworkRecord",
    "all_scfv_framework_names",
    "all_vhh_framework_names",
    "get_scfv_framework_record",
    "get_scfv_framework_template_cif",
    "get_vhh_framework_record",
    "get_vhh_framework_template_cif",
    "is_scfv_framework_alias",
    "is_vhh_framework_alias",
    "resolve_scfv_framework_name",
    "resolve_vhh_framework_name",
    "scfv_framework_alias_choices",
    "vhh_framework_alias_choices",
]
