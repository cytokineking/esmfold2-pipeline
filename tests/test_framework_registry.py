from __future__ import annotations

import unittest

from esmfold2_pipeline.frameworks import (
    all_scfv_framework_names,
    all_vhh_framework_names,
    get_scfv_framework_record,
    get_scfv_framework_template_cif,
    get_vhh_framework_record,
    get_vhh_framework_template_cif,
    resolve_scfv_framework_name,
    resolve_vhh_framework_name,
    scfv_framework_alias_choices,
    vhh_framework_alias_choices,
)

EXPECTED_SCFV_FRAMEWORKS = (
    "anifrolumab_framework_vhvl",
    "atezolizumab_framework_vhvl",
    "avelumab_framework_vhvl",
    "belimumab_framework_vhvl",
    "daratumumab_framework_vhvl",
    "dupilumab_framework_vhvl",
    "guselkumab_framework_vhvl",
    "lebrikizumab_framework_vhvl",
    "panitumumab_framework_vhvl",
    "pembrolizumab_framework_vhvl",
    "secukinumab_framework_vhvl",
    "tezepelumab_framework_vhvl",
    "tralokinumab_framework_vhvl",
    "trastuzumab_framework_vhvl",
)
EXPECTED_VHH_FRAMEWORKS = (
    "caplacizumab_framework_vhh",
    "ozoralizumab_tnf_framework_vhh",
    "vobarilizumab_il6r_framework_vhh",
)


class FrameworkRegistryTest(unittest.TestCase):
    def test_scfv_framework_aliases_resolve_to_canonical_names(self) -> None:
        self.assertEqual(all_scfv_framework_names(), EXPECTED_SCFV_FRAMEWORKS)
        self.assertEqual(
            resolve_scfv_framework_name("trastuzumab"),
            "trastuzumab_framework_vhvl",
        )
        self.assertEqual(
            resolve_scfv_framework_name("TRASTUZUMAB_FRAMEWORK_VHVL"),
            "trastuzumab_framework_vhvl",
        )
        self.assertEqual(resolve_scfv_framework_name("humanized_fmc63"), None)
        self.assertIn("atezolizumab", scfv_framework_alias_choices())

    def test_scfv_framework_records_include_template_and_cdr_lengths(self) -> None:
        record = get_scfv_framework_record("belimumab")

        self.assertEqual(record.id, "belimumab")
        self.assertEqual(record.canonical_name, "belimumab_framework_vhvl")
        self.assertIn("{hcdr3}", record.template)
        self.assertEqual(record.cdr_lengths["hcdr3"], (13, 19))
        self.assertEqual(record.cdr_lengths["lcdr2"], (3, 3))

    def test_vhh_framework_aliases_resolve_to_canonical_names(self) -> None:
        self.assertEqual(all_vhh_framework_names(), EXPECTED_VHH_FRAMEWORKS)
        self.assertEqual(
            resolve_vhh_framework_name("caplacizumab"),
            "caplacizumab_framework_vhh",
        )
        self.assertEqual(
            resolve_vhh_framework_name("CAPLACIZUMAB_FRAMEWORK_VHH"),
            "caplacizumab_framework_vhh",
        )
        self.assertEqual(
            resolve_vhh_framework_name("ozoralizumab_tnf"),
            "ozoralizumab_tnf_framework_vhh",
        )
        self.assertIn("vobarilizumab_il6r", vhh_framework_alias_choices())
        self.assertIsNone(resolve_scfv_framework_name("caplacizumab"))

    def test_vhh_framework_records_include_template_and_cdr_lengths(self) -> None:
        record = get_vhh_framework_record("caplacizumab")

        self.assertEqual(record.id, "caplacizumab_vhh")
        self.assertEqual(record.canonical_name, "caplacizumab_framework_vhh")
        self.assertIn("{cdr3}", record.template)
        self.assertEqual(record.cdr_lengths["cdr1"], (7, 9))
        self.assertEqual(record.cdr_lengths["cdr3"], (18, 24))

    def test_framework_template_cif_resolves_active_bundled_assets(self) -> None:
        scfv = get_scfv_framework_template_cif("anifrolumab_framework_vhvl")
        vhh = get_vhh_framework_template_cif("caplacizumab")

        self.assertIsNotNone(scfv)
        self.assertIsNotNone(vhh)
        assert scfv is not None
        assert vhh is not None
        self.assertEqual(scfv.name, "anifrolumab_4QXG.cif")
        self.assertEqual(vhh.name, "caplacizumab_vhh_7EOW.cif")
        self.assertTrue(scfv.exists())
        self.assertTrue(vhh.exists())


if __name__ == "__main__":
    unittest.main()
