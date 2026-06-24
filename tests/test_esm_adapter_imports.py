from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from esmfold2_pipeline.esm_adapter.imports import check_environment


class ESMAdapterImportsTest(unittest.TestCase):
    def test_local_runtime_check_does_not_require_tutorial_script(self) -> None:
        class FakeProteinChain:
            from_atomarray = object()

        class FakeProteinComplex:
            from_chains = object()

        def import_module(name: str):
            if name == "torch":
                return types.SimpleNamespace(
                    __version__="fake",
                    cuda=types.SimpleNamespace(
                        is_available=lambda: False,
                        device_count=lambda: 0,
                    ),
                )
            if name == "esm":
                return types.SimpleNamespace(__file__=None, __version__="fake")
            if name == "transformers.models.esmfold2.modeling_esmfold2_experimental":
                return types.SimpleNamespace(ESMFold2ExperimentalModel=object)
            if name == "transformers.models.esmc.modeling_esmc":
                return types.SimpleNamespace(ESMCForMaskedLM=object)
            if name == "transformers.models.esmc.tokenization_esmc":
                return types.SimpleNamespace(ESMCTokenizer=object)
            if name == "transformers.models.esmfold2.modeling_esmfold2_common":
                return types.SimpleNamespace(
                    _seed_context=object,
                    CUE_AVAILABLE=False,
                )
            if name == "esm.models.esmfold2":
                return types.SimpleNamespace(
                    ELEMENT_NUMBER_TO_SYMBOL={1: "H"},
                    ProteinInput=object,
                    StructurePredictionInput=object,
                    load_ccd=object,
                    prepare_esmfold2_input=object,
                )
            if name == "esm.models.esmfold2.constants":
                return types.SimpleNamespace(
                    PROTEIN_3TO1={},
                    RES_TYPE_TO_CCD={},
                    MOL_TYPE_NONPOLYMER=2,
                )
            if name == "esm.utils.structure.protein_chain":
                return types.SimpleNamespace(ProteinChain=FakeProteinChain)
            if name == "esm.utils.structure.protein_complex":
                return types.SimpleNamespace(ProteinComplex=FakeProteinComplex)
            if name == "biotite.structure":
                return types.SimpleNamespace(
                    Atom=object,
                    array=object,
                    chain_iter=object,
                )
            raise AssertionError(f"unexpected import: {name}")

        with patch(
            "esmfold2_pipeline.esm_adapter.imports.importlib.import_module",
            side_effect=import_module,
        ):
            result = check_environment(
                require_cuda=False,
                require_tutorial=False,
                require_local_runtime=True,
            )

        self.assertTrue(result.ok, result.errors)
        self.assertIsNone(result.checks["binder_design_py"])
        self.assertTrue(all(result.checks["local_runtime"].values()))


if __name__ == "__main__":
    unittest.main()
