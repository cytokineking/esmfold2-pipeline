from __future__ import annotations

import unittest
from importlib import resources

import yaml


VHH_PACKAGE = "esmfold2_pipeline.frameworks.vhh"
EXPECTED_VHH_FRAMEWORKS = (
    "caplacizumab_framework_vhh",
    "ozoralizumab_tnf_framework_vhh",
    "vobarilizumab_il6r_framework_vhh",
)
VHH_CDR_NAMES = ("cdr1", "cdr2", "cdr3")


class VhhFrameworkRecordsTest(unittest.TestCase):
    def test_core_vhh_yaml_records_are_valid_templates(self) -> None:
        records = _load_vhh_records()

        self.assertEqual(
            tuple(sorted(record["canonical_name"] for record in records)),
            EXPECTED_VHH_FRAMEWORKS,
        )
        for record in records:
            self.assertEqual(record["modality"], "vhh")
            self.assertEqual(record["format"], "template")
            self.assertEqual(record["panel"], "clinical_core")
            self.assertEqual(record["domain_role"], "disease_targeting")
            for cdr_name in VHH_CDR_NAMES:
                self.assertEqual(record["template"].count(f"{{{cdr_name}}}"), 1)
                self.assertIn(cdr_name, record["cdr_lengths"])
                self.assertIn(
                    cdr_name,
                    record["framework_annotation"]["observed_cdr_lengths"],
                )

    def test_multispecific_records_identify_selected_sequence_field(self) -> None:
        records = {
            record["id"]: record
            for record in _load_vhh_records()
        }

        self.assertEqual(
            records["ozoralizumab_tnf_vhh"]["therapeutic_source"][
                "selected_sequence_field"
            ],
            "HeavySequence",
        )
        self.assertTrue(
            records["ozoralizumab_tnf_vhh"]["multispecific_context"][
                "has_additional_heavy_domain"
            ]
        )


def _load_vhh_records() -> list[dict[str, object]]:
    records = []
    for file in sorted(resources.files(VHH_PACKAGE).iterdir(), key=lambda item: item.name):
        if file.suffix not in {".yaml", ".yml"}:
            continue
        records.append(yaml.safe_load(file.read_text()))
    return records


if __name__ == "__main__":
    unittest.main()
