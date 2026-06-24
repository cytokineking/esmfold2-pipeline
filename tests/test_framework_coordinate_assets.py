from __future__ import annotations

import unittest
from importlib import resources

import yaml
from biotite.structure.io.pdbx import CIFFile


SCFV_PACKAGE = "esmfold2_pipeline.frameworks.scfv"
VHH_PACKAGE = "esmfold2_pipeline.frameworks.vhh"
SCFV_LINKER = "GGGSGGGSGGGSGGGS"
PLACEHOLDER_RE = r"\{[^}]+\}"
AA3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

EXPECTED_COORDINATES = {
    "anifrolumab": "anifrolumab_4QXG.cif",
    "atezolizumab": "atezolizumab_5X8L.cif",
    "avelumab": "avelumab_4NKI.cif",
    "belimumab": "belimumab_5Y9K.cif",
    "daratumumab": "daratumumab_7DUN.cif",
    "dupilumab": "dupilumab_6WG8.cif",
    "guselkumab": "guselkumab_4M6M.cif",
    "lebrikizumab": "lebrikizumab_4I77.cif",
    "panitumumab": "panitumumab_5SX5.cif",
    "pembrolizumab": "pembrolizumab_5GGS.cif",
    "secukinumab": "secukinumab_6WIO.cif",
    "tezepelumab": "tezepelumab_5J13.cif",
    "tralokinumab": "tralokinumab_5L6Y.cif",
    "trastuzumab": "trastuzumab_6BHZ.cif",
    "caplacizumab_vhh": "caplacizumab_vhh_7EOW.cif",
    "ozoralizumab_tnf_vhh": "ozoralizumab_tnf_vhh_8Z8M.cif",
    "vobarilizumab_il6r_vhh": "vobarilizumab_il6r_vhh_7XL0.cif",
}


class FrameworkCoordinateAssetsTest(unittest.TestCase):
    def test_active_framework_yamls_have_matching_coordinate_assets(self) -> None:
        records = _load_records(SCFV_PACKAGE) + _load_records(VHH_PACKAGE)

        self.assertEqual(
            {record["id"] for record in records},
            set(EXPECTED_COORDINATES),
        )
        for record in records:
            package = SCFV_PACKAGE if record["modality"] == "scfv" else VHH_PACKAGE
            coordinate_name = EXPECTED_COORDINATES[record["id"]]
            expected_sequence = _framework_only_sequence(record["template"])
            polymers = _coordinate_polymer_sequences(package, coordinate_name)

            self.assertEqual(polymers, [expected_sequence])
            self.assertTrue(
                resources.files(package)
                .joinpath("reference_structures", coordinate_name)
                .is_file()
            )
            _assert_atom_site_is_framework_only(
                self,
                package,
                coordinate_name,
                expected_sequence,
            )
            if record["modality"] == "scfv":
                linker_start = len(
                    _framework_only_sequence(record["template"].split(SCFV_LINKER, 1)[0])
                ) + 1
                linker_positions = set(
                    range(linker_start, linker_start + len(SCFV_LINKER))
                )
                atom_positions = _atom_label_seq_ids(package, coordinate_name)
                self.assertTrue(atom_positions.isdisjoint(linker_positions))


def _load_records(package: str) -> list[dict[str, object]]:
    records = []
    for file in sorted(resources.files(package).iterdir(), key=lambda item: item.name):
        if file.suffix not in {".yaml", ".yml"}:
            continue
        records.append(yaml.safe_load(file.read_text()))
    return records


def _coordinate_polymer_sequences(package: str, filename: str) -> list[str]:
    block = _cif_block(package, filename)
    category = block["entity_poly"]
    sequences = category["pdbx_seq_one_letter_code_can"].as_array()
    return [str(sequence).replace("\n", "").replace(" ", "") for sequence in sequences]


def _framework_only_sequence(template: str) -> str:
    import re

    return re.sub(PLACEHOLDER_RE, "", template)


def _assert_atom_site_is_framework_only(
    test_case: unittest.TestCase,
    package: str,
    filename: str,
    expected_sequence: str,
) -> None:
    atom_site = _cif_block(package, filename)["atom_site"]
    chain_ids = {str(value) for value in atom_site["label_asym_id"].as_array()}
    entity_ids = {str(value) for value in atom_site["label_entity_id"].as_array()}
    group_ids = {str(value) for value in atom_site["group_PDB"].as_array()}
    test_case.assertEqual(chain_ids, {"A"})
    test_case.assertEqual(entity_ids, {"1"})
    test_case.assertEqual(group_ids, {"ATOM"})

    seq_ids = [str(value) for value in atom_site["label_seq_id"].as_array()]
    comp_ids = [str(value) for value in atom_site["label_comp_id"].as_array()]
    for raw_seq_id, comp_id in zip(seq_ids, comp_ids, strict=True):
        seq_id = int(raw_seq_id)
        test_case.assertGreaterEqual(seq_id, 1)
        test_case.assertLessEqual(seq_id, len(expected_sequence))
        test_case.assertEqual(AA3_TO_1[comp_id], expected_sequence[seq_id - 1])


def _atom_label_seq_ids(package: str, filename: str) -> set[int]:
    atom_site = _cif_block(package, filename)["atom_site"]
    return {int(str(value)) for value in atom_site["label_seq_id"].as_array()}


def _cif_block(package: str, filename: str):
    coordinate = resources.files(package).joinpath(filename)
    with resources.as_file(coordinate) as coordinate_path:
        cif = CIFFile.read(str(coordinate_path))
    return cif[next(iter(cif.keys()))]


if __name__ == "__main__":
    unittest.main()
