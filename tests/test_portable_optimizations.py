from __future__ import annotations

from contextlib import nullcontext
import os
import sys
import types
import unittest
from unittest.mock import patch

from esmfold2_pipeline.esm_adapter.portable import (
    LazySampleOutput,
    call_model,
    optimization_mode,
)


class _StructureHead:
    def __init__(self):
        self.calls = 0

    def sample(self):
        self.calls += 1
        return {"sample_atom_coords": (self.calls,)}


class _Model:
    def __init__(self):
        self.confidence_head = object()
        self.structure_head = _StructureHead()

    def __call__(self):
        result = {"distogram_logits": 7}
        result.update(self.structure_head.sample())
        if self.confidence_head is not None:
            result["iptm"] = 0.9
        return result


class PortableOptimizationsTest(unittest.TestCase):
    def test_setting_defaults_off_and_rejects_typos(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(optimization_mode(), "off")
        with patch.dict(os.environ, {"ESMFOLD2_PIPELINE_OPTIMIZATIONS": "fast"}):
            with self.assertRaisesRegex(ValueError, "must be"):
                optimization_mode()
    def test_early_fold_defers_real_coordinates_and_restores_modules(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.is_autocast_enabled = lambda device: False
        fake_torch.get_autocast_dtype = lambda device: "bf16"
        fake_torch.no_grad = nullcontext
        fake_torch.autocast = lambda *args, **kwargs: nullcontext()
        common = types.ModuleType(
            "transformers.models.esmfold2.modeling_esmfold2_common"
        )
        common._seed_context = lambda seed: nullcontext()
        model = _Model()
        original_head = model.confidence_head
        with patch.dict(os.environ, {"ESMFOLD2_PIPELINE_OPTIMIZATIONS": "portable"}), patch.dict(
            sys.modules,
            {"torch": fake_torch,
             "transformers.models.esmfold2.modeling_esmfold2_common": common},
        ):
            output = call_model(
                model, model, calculate_confidence=False,
                seed=13, num_sampling_steps=1,
            )
            self.assertIsInstance(output, LazySampleOutput)
            self.assertEqual(output["distogram_logits"], 7)
            self.assertEqual(model.structure_head.calls, 0)
            self.assertIs(model.confidence_head, original_head)
            self.assertNotIn("sample", vars(model.structure_head))
            self.assertEqual(output["sample_atom_coords"], (1,))
            self.assertEqual(output["sample_atom_coords"], (1,))
            self.assertEqual(model.structure_head.calls, 1)
            self.assertEqual(model._esmfold2_portable_stats,
                             {"confidence_skipped": 1, "sampling_deferred": 1})

            copied_output = call_model(
                model, model, calculate_confidence=False,
                seed=14, num_sampling_steps=1,
            )
            self.assertEqual(dict(copied_output)["sample_atom_coords"], (2,))
            self.assertEqual(model.structure_head.calls, 2)

    def test_late_confidence_fold_remains_eager(self):
        model = _Model()
        with patch.dict(os.environ, {"ESMFOLD2_PIPELINE_OPTIMIZATIONS": "portable"}):
            output = call_model(
                model, model, calculate_confidence=True,
                seed=13, num_sampling_steps=50,
            )
        self.assertEqual(model.structure_head.calls, 1)
        self.assertEqual(output["iptm"], 0.9)

    def test_exception_restores_model(self):
        model = _Model()
        head = model.confidence_head
        with patch.dict(os.environ, {"ESMFOLD2_PIPELINE_OPTIMIZATIONS": "portable"}):
            with self.assertRaisesRegex(RuntimeError, "broken"):
                call_model(
                    model, lambda: (_ for _ in ()).throw(RuntimeError("broken")),
                    calculate_confidence=False, seed=13, num_sampling_steps=1,
                )
        self.assertIs(model.confidence_head, head)
        self.assertNotIn("sample", vars(model.structure_head))


if __name__ == "__main__":
    unittest.main()
