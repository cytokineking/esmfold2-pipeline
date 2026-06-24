from __future__ import annotations

import unittest

from esmfold2_pipeline.planning import binder_code, semantic_candidate_id


class PlanningIdsTest(unittest.TestCase):
    def test_semantic_candidate_id_uses_target_binder_type_and_seed(self) -> None:
        self.assertEqual(
            semantic_candidate_id(
                target_name="IL-2 Asn88/91",
                binder_scaffold="miniprotein",
                seed=12,
            ),
            "il_2_asn88_91_mp_seed12",
        )
        self.assertEqual(
            semantic_candidate_id(
                target_name="KRAS-G12C A11",
                binder_scaffold="scfv",
                seed=3,
                candidate_index=2,
            ),
            "kras_g12c_a11_scfv_seed3_c2",
        )
        self.assertEqual(
            semantic_candidate_id(
                target_name="CD45",
                binder_scaffold="scfv",
                seed=4,
            ),
            "cd45_scfv_seed4",
        )

    def test_binder_code_keeps_known_scaffolds_short(self) -> None:
        self.assertEqual(binder_code("minibinder"), "mp")
        self.assertEqual(binder_code("miniprotein"), "mp")
        self.assertEqual(binder_code("scFv"), "scfv")

if __name__ == "__main__":
    unittest.main()
