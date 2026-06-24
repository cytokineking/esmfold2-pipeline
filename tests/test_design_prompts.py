from __future__ import annotations

import unittest

from esmfold2_pipeline.design.prompts import (
    cdr_prompt_from_indices,
    contiguous_index_runs,
    mutable_run_sequences,
    prepare_binder_prompt_plan,
)


class DesignPromptTest(unittest.TestCase):
    def test_prepare_scfv_template_prompt_locally(self) -> None:
        plan = prepare_binder_prompt_plan(
            binder_name="custom_framework",
            binder_scaffold="scfv",
            binder_framework_name="custom_framework",
            binder_framework_source="template",
            binder_framework_template="EV{hcdr1}Q{hcdr2}SS",
            binder_framework_cdr_lengths={
                "hcdr1": (2, 2),
                "hcdr2": (3, 3),
            },
            binder_framework_sequence=None,
            binder_framework_cdr_indices=None,
            seed=11,
            is_antibody=None,
        )

        self.assertIsNone(plan.binder_name)
        self.assertEqual(plan.binder_sequence, "EV##Q###SS")
        self.assertTrue(plan.is_antibody)
        self.assertEqual(plan.cdr_indices, (2, 3, 5, 6, 7))
        self.assertEqual(plan.cdr_lengths, {"hcdr1": 2, "hcdr2": 3})

    def test_non_antibody_prompt_stays_factory_driven_for_tutorial_backend(self) -> None:
        plan = prepare_binder_prompt_plan(
            binder_name="minibinder",
            binder_scaffold="miniprotein",
            binder_framework_name=None,
            binder_framework_source=None,
            binder_framework_template=None,
            binder_framework_cdr_lengths=None,
            binder_framework_sequence=None,
            binder_framework_cdr_indices=None,
            seed=0,
            is_antibody=None,
        )

        self.assertEqual(plan.binder_name, "minibinder")
        self.assertIsNone(plan.binder_sequence)
        self.assertIsNone(plan.is_antibody)
        self.assertEqual(plan.cdr_indices, ())

    def test_miniprotein_prompt_can_be_owned_locally(self) -> None:
        plan = prepare_binder_prompt_plan(
            binder_name="minibinder",
            binder_scaffold="miniprotein",
            binder_framework_name=None,
            binder_framework_source=None,
            binder_framework_template=None,
            binder_framework_cdr_lengths=None,
            binder_framework_sequence=None,
            binder_framework_cdr_indices=None,
            seed=0,
            is_antibody=None,
            binder_length_range=(7, 7),
            local_miniprotein=True,
        )

        self.assertIsNone(plan.binder_name)
        self.assertEqual(plan.binder_sequence, "#######")
        self.assertFalse(plan.is_antibody)
        self.assertEqual(plan.cdr_indices, ())

    def test_cdr_prompt_and_runs_are_deterministic(self) -> None:
        prompt = cdr_prompt_from_indices("EVAAQBSS", (2, 3, 5))

        self.assertEqual(prompt, "EV##Q#SS")
        self.assertEqual(contiguous_index_runs((5, 2, 3)), [(2, 4), (5, 6)])
        self.assertEqual(
            mutable_run_sequences("EVAAQBSS", (2, 3, 5)),
            {"hcdr1": "AA", "hcdr2": "B"},
        )

    def test_custom_sequence_framework_uses_explicit_cdr_indices(self) -> None:
        plan = prepare_binder_prompt_plan(
            binder_name="lab_fixed",
            binder_scaffold="scfv",
            binder_framework_name="lab_fixed",
            binder_framework_source="sequence",
            binder_framework_template=None,
            binder_framework_cdr_lengths=None,
            binder_framework_sequence="EVAAQBSS",
            binder_framework_cdr_indices=(2, 3, 5),
            seed=0,
            is_antibody=None,
        )

        self.assertIsNone(plan.binder_name)
        self.assertEqual(plan.binder_sequence, "EV##Q#SS")
        self.assertTrue(plan.is_antibody)
        self.assertEqual(plan.cdr_indices, (2, 3, 5))
        self.assertEqual(plan.cdr_lengths, {"hcdr1": 2, "hcdr2": 1})
        self.assertEqual(plan.cdr_report_names, ("hcdr1", "hcdr2"))


if __name__ == "__main__":
    unittest.main()
