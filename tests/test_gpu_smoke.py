from __future__ import annotations

import tempfile
import types
import unittest
from dataclasses import dataclass, replace
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

import esmfold2_pipeline.esm_adapter.binder_design as adapter_module
from esmfold2_pipeline.config import TargetGeometryDriftConfig
from esmfold2_pipeline.db import CampaignStore, connect_database
from esmfold2_pipeline.design import DesignRunResult, RuntimeModels
from esmfold2_pipeline.esm_adapter import preflight_models, run_binder_design_artifact
from esmfold2_pipeline.execution.gpu_smoke import (
    DEFAULT_CRITIC_NAME,
    DEFAULT_INVERSION_MODEL_NAME,
    plan_one_gpu_smoke_shard,
    run_one_gpu_smoke_shard,
)
from esmfold2_pipeline.reports import inspect_campaign
from esmfold2_pipeline.structure import StructureTargetConfig, parse_structure_target


class FakeComplex:
    def to_pdb_string(self) -> str:
        return "HEADER    FAKE GPU SMOKE COMPLEX\nEND\n"


class FakeDesignApp:
    last_design_kwargs = None

    def __init__(self) -> None:
        self.inversion_model_names = []
        self.hero_critic_hf_paths = []
        self.scaling_critic_hf_paths = []
        self.loaded = False

    def load(self, use_scaling_critics: bool) -> None:
        self.loaded = True
        assert use_scaling_critics is False
        self.inversion_models = {
            name: object() for name in self.inversion_model_names
        }
        self.hf_critic_models = {
            name: object() for name in self.hero_critic_hf_paths
        }
        self.esmc_model = object()

    def design(self, **kwargs):
        assert self.loaded
        FakeDesignApp.last_design_kwargs = kwargs
        return (
            ["TARGET|BINDER"],
            {},
            [
                {
                    "critic_name": DEFAULT_CRITIC_NAME,
                    "batch_idx": 0,
                    "designed_sequence": "TARGET|BINDER",
                    "complex": FakeComplex(),
                    "final_loss": 1.23,
                    "iptm": 0.91,
                    "distogram_iptm_proxy": 0.82,
                }
            ],
        )


class GPUSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        adapter_module._LOCAL_DESIGN_RUNTIME_CACHE.clear()

    def _runtime_cache_test_spec(self, *, gpu_id: str = "0"):
        return adapter_module._build_design_spec(
            campaign_dir=Path("/tmp/campaign"),
            candidate_id="candidate",
            shard_id="shard",
            seed=0,
            esm_repo="/tmp/esm",
            gpu_id=gpu_id,
            steps=1,
            target_name="target",
            binder_name="minibinder",
            critic_name="critic-model",
            binder_scaffold=None,
            binder_framework_name=None,
            binder_framework_source=None,
            binder_framework_template=None,
            binder_framework_cdr_lengths=None,
            binder_framework_sequence=None,
            binder_framework_cdr_indices=None,
            target_sequence=None,
            binder_length_range=(2, 2),
            is_antibody=False,
            inversion_model_name="inv-model",
            structure_target=None,
            target_structure_indexing="auto",
            conditioning_mode="none",
            conditioning_assembly=False,
            conditioning_chain_pairs=None,
            hotspot_contact_weight=0.0,
            hotspot_contact_cutoff_angstrom=None,
            hotspot_distogram_contact_cutoff_angstrom=20.0,
            hotspot_critic_contact_cutoff_angstrom=5.0,
            hotspot_num_contacts=1,
            hotspot_contact_probability_target=0.6,
            hotspot_loss_mode="entropy_hotspot",
            target_geometry_drift=None,
            artifact_stem=None,
            disable_hf_xet=True,
        )

    def test_gpu_smoke_keeps_heavy_result_inside_worker_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir)
            shard_id = plan_one_gpu_smoke_shard(campaign_dir)
            fake_module = types.SimpleNamespace(
                STEPS=150,
                LOG_INTERVAL=5,
                ESMFold2Design=FakeDesignApp,
            )

            with patch.dict(
                "os.environ",
                {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "tutorial"},
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
                return_value=fake_module,
            ):
                result = run_one_gpu_smoke_shard(
                    campaign_dir,
                    worker_id="gpu-test-worker",
                    gpu_id=None,
                    steps=2,
                )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.shard_id, shard_id)
            self.assertEqual(result.candidate_id, "cand_000000_0000")
            self.assertEqual(result.critic_name, DEFAULT_CRITIC_NAME)
            self.assertEqual(result.metrics["iptm"], 0.91)
            self.assertFalse(hasattr(result, "complex"))
            self.assertFalse(hasattr(result, "logits"))

            self.assertIsNone(result.sequence_path)
            self.assertEqual(result.structure_path, "esmfold2/structures/s000_seed000_c000.pdb")
            self.assertIn(
                "FAKE GPU SMOKE COMPLEX",
                (campaign_dir / result.structure_path).read_text(),
            )

            conn = connect_database(campaign_dir / "campaign.sqlite")
            store = CampaignStore(conn)
            shard = store.fetch_one("SELECT * FROM shards WHERE shard_id = ?", (shard_id,))
            self.assertEqual(shard["status"], "completed")
            candidate = store.fetch_one(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (result.candidate_id,),
            )
            self.assertEqual(candidate["designed_sequence"], "BINDER")
            self.assertIsNone(candidate["sequence_path"])
            critic = store.fetch_one(
                """
                SELECT *
                FROM critic_metrics
                WHERE candidate_id = ? AND critic_name = ?
                """,
                (result.candidate_id, DEFAULT_CRITIC_NAME),
            )
            self.assertEqual(critic["structure_path"], result.structure_path)
            self.assertEqual(critic["iptm"], 0.91)
            conn.close()

            status = inspect_campaign(campaign_dir)
            self.assertEqual(status.issues, [])

    def test_model_preflight_loads_separate_inversion_and_critic_models(self) -> None:
        fake_module = types.SimpleNamespace(
            STEPS=150,
            LOG_INTERVAL=5,
            ESMFold2Design=FakeDesignApp,
        )

        with patch.dict(
            "os.environ",
            {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "tutorial"},
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
            return_value=fake_module,
        ):
            result = preflight_models(
                esm_repo=None,
                gpu_id=None,
                inversion_model_name="ESMFold2-Experimental-Fast",
                critic_name=DEFAULT_CRITIC_NAME,
            )

        self.assertEqual(result.inversion_model_name, "ESMFold2-Experimental-Fast")
        self.assertEqual(result.critic_name, DEFAULT_CRITIC_NAME)
        self.assertEqual(result.loaded_inversion_models, ["ESMFold2-Experimental-Fast"])
        self.assertEqual(result.loaded_critic_models, [DEFAULT_CRITIC_NAME])
        self.assertTrue(result.esmc_loaded)

    def test_local_runtime_loader_uses_model_apis_without_design_app(self) -> None:
        fold_from_pretrained_calls: list[tuple[str, bool]] = []
        esmc_from_pretrained_calls: list[tuple[str, object]] = []
        fold_instances: list[object] = []

        class FakeFoldModel:
            def __init__(self, repo_id: str, load_esmc: bool) -> None:
                self.repo_id = repo_id
                self.load_esmc_flag = load_esmc
                self.config = types.SimpleNamespace(esmc_id=f"{repo_id}:esmc")
                self.loaded_esmc_ids: list[str] = []
                self.dropout_calls: list[tuple[float, bool]] = []
                self.kernel_backends: list[str | None] = []
                self.to_devices: list[str] = []
                self.eval_called = False
                self.requires_grad_calls: list[bool] = []
                self._esmc = None
                fold_instances.append(self)

            def load_esmc(self, esmc_id: str) -> None:
                self.loaded_esmc_ids.append(esmc_id)
                self._esmc = f"loaded:{esmc_id}"

            def configure_lm_dropout(
                self,
                lm_dropout: float,
                *,
                force_lm_dropout_during_inference: bool,
            ):
                self.dropout_calls.append(
                    (lm_dropout, force_lm_dropout_during_inference)
                )

            def set_kernel_backend(self, backend: str | None):
                self.kernel_backends.append(backend)

            def to(self, *, device: str):
                self.to_devices.append(device)
                return self

            def eval(self):
                self.eval_called = True
                return self

            def requires_grad_(self, requires_grad: bool):
                self.requires_grad_calls.append(requires_grad)
                return self

        class FakeFoldModelApi:
            @staticmethod
            def from_pretrained(repo_id: str, *, load_esmc: bool):
                fold_from_pretrained_calls.append((repo_id, load_esmc))
                return FakeFoldModel(repo_id, load_esmc)

        class FakeESMCModel:
            def __init__(self) -> None:
                self.cuda_called = False
                self.eval_called = False
                self.requires_grad_calls: list[bool] = []

            def cuda(self):
                self.cuda_called = True
                return self

            def eval(self):
                self.eval_called = True
                return self

            def requires_grad_(self, requires_grad: bool):
                self.requires_grad_calls.append(requires_grad)
                return self

        class FakeESMCForMaskedLM:
            @staticmethod
            def from_pretrained(repo_id: str, *, torch_dtype):
                esmc_from_pretrained_calls.append((repo_id, torch_dtype))
                return FakeESMCModel()

        fake_module = types.SimpleNamespace(
            ESMFold2ExperimentalModel=FakeFoldModelApi,
            ESMCForMaskedLM=FakeESMCForMaskedLM,
            torch=types.SimpleNamespace(float32="float32"),
            CUE_AVAILABLE=True,
            COMPILE=False,
        )
        old_cache = adapter_module._LOCAL_ESMC_CACHE
        adapter_module._LOCAL_ESMC_CACHE = None
        try:
            runtime = adapter_module._load_local_runtime_models(
                fake_module,
                inversion_model_name="inv-model",
                critic_name="critic-model",
            )
        finally:
            adapter_module._LOCAL_ESMC_CACHE = old_cache

        self.assertEqual(
            fold_from_pretrained_calls,
            [("biohub/inv-model", False), ("biohub/critic-model", False)],
        )
        self.assertEqual(
            esmc_from_pretrained_calls,
            [("biohub/ESMC-6B", "float32")],
        )
        self.assertEqual(list(runtime.inversion_models), ["inv-model"])
        self.assertEqual(list(runtime.critic_models), ["critic-model"])
        self.assertIsInstance(runtime.esmc_model, FakeESMCModel)
        self.assertEqual(
            [instance.loaded_esmc_ids for instance in fold_instances],
            [["biohub/inv-model:esmc"], []],
        )
        self.assertIs(fold_instances[1]._esmc, fold_instances[0]._esmc)
        self.assertEqual(fold_instances[0].dropout_calls, [(0.5, True)])
        self.assertEqual(fold_instances[1].dropout_calls, [(0.25, True)])
        self.assertEqual(
            [instance.kernel_backends for instance in fold_instances],
            [["cuequivariance"], ["cuequivariance"]],
        )
        self.assertEqual(
            [instance.to_devices for instance in fold_instances],
            [["cuda"], ["cuda"]],
        )
        self.assertTrue(all(instance.eval_called for instance in fold_instances))
        self.assertEqual(
            [instance.requires_grad_calls for instance in fold_instances],
            [[False], [False]],
        )
        self.assertTrue(runtime.esmc_model.cuda_called)
        self.assertTrue(runtime.esmc_model.eval_called)
        self.assertEqual(runtime.esmc_model.requires_grad_calls, [False])


    def test_local_design_runtime_cache_reuses_worker_models(self) -> None:
        loaded_runtime = object()
        loaded_models = RuntimeModels(
            inversion_models={"inv-model": object()},
            critic_models={"critic-model": object()},
            esmc_model=object(),
            helpers={},
        )
        spec = self._runtime_cache_test_spec()

        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_esm_folding_runtime",
            return_value=loaded_runtime,
        ) as load_runtime, patch(
            "esmfold2_pipeline.esm_adapter.binder_design._load_local_runtime_models",
            return_value=loaded_models,
        ) as load_models:
            first = adapter_module._get_or_load_local_design_runtime(spec)
            second = adapter_module._get_or_load_local_design_runtime(spec)

        self.assertIs(first, second)
        self.assertIs(first.binder_design, loaded_runtime)
        self.assertIs(first.runtime_models, loaded_models)
        load_runtime.assert_called_once_with("/tmp/esm")
        load_models.assert_called_once_with(
            loaded_runtime,
            inversion_model_name="inv-model",
            critic_name="critic-model",
        )

    def test_local_design_runtime_cache_can_be_disabled_by_env(self) -> None:
        loaded_models = RuntimeModels(
            inversion_models={"inv-model": object()},
            critic_models={"critic-model": object()},
            esmc_model=object(),
            helpers={},
        )
        spec = self._runtime_cache_test_spec()

        with patch.dict(
            "os.environ",
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "ESMFOLD2_PIPELINE_DISABLE_LOCAL_RUNTIME_CACHE": "1",
            },
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_esm_folding_runtime",
            side_effect=[object(), object()],
        ) as load_runtime, patch(
            "esmfold2_pipeline.esm_adapter.binder_design._load_local_runtime_models",
            return_value=loaded_models,
        ) as load_models:
            first = adapter_module._get_or_load_local_design_runtime(spec)
            second = adapter_module._get_or_load_local_design_runtime(spec)

        self.assertIsNot(first, second)
        self.assertEqual(load_runtime.call_count, 2)
        self.assertEqual(load_models.call_count, 2)
        self.assertEqual(adapter_module._LOCAL_DESIGN_RUNTIME_CACHE, {})

    def test_local_design_runtime_cache_separates_gpu_bindings(self) -> None:
        spec = self._runtime_cache_test_spec()
        loaded_models = RuntimeModels(
            inversion_models={"inv-model": object()},
            critic_models={"critic-model": object()},
            esmc_model=object(),
            helpers={},
        )

        with patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_esm_folding_runtime",
            side_effect=[object(), object()],
        ) as load_runtime, patch(
            "esmfold2_pipeline.esm_adapter.binder_design._load_local_runtime_models",
            return_value=loaded_models,
        ) as load_models:
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
                adapter_module._get_or_load_local_design_runtime(spec)
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}):
                adapter_module._get_or_load_local_design_runtime(
                    replace(spec, gpu_id="1")
                )

        self.assertEqual(load_runtime.call_count, 2)
        self.assertEqual(load_models.call_count, 2)

    def test_local_runtime_model_loader_dedupes_identical_model_specs(self) -> None:
        fold_from_pretrained_calls: list[tuple[str, bool]] = []
        fold_instances: list[object] = []

        class FakeFoldModel:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(esmc_id="esmc")
                self._esmc = None

            @classmethod
            def from_pretrained(cls, repo_id: str, *, load_esmc: bool):
                fold_from_pretrained_calls.append((repo_id, load_esmc))
                instance = cls()
                fold_instances.append(instance)
                return instance

            def load_esmc(self, esmc_id: str) -> None:
                self._esmc = object()

            def configure_lm_dropout(self, *args, **kwargs):
                return None

            def set_kernel_backend(self, backend):
                return None

            def to(self, *, device: str):
                return self

            def eval(self):
                return self

            def requires_grad_(self, requires_grad: bool):
                return self

        class FakeESMCModel:
            def cuda(self):
                return self

            def eval(self):
                return self

            def requires_grad_(self, requires_grad: bool):
                return self

        class FakeESMCForMaskedLM:
            @staticmethod
            def from_pretrained(repo_id: str, *, torch_dtype):
                return FakeESMCModel()

        fake_module = types.SimpleNamespace(
            ESMFold2ExperimentalModel=FakeFoldModel,
            ESMCForMaskedLM=FakeESMCForMaskedLM,
            torch=types.SimpleNamespace(float32="float32"),
            CUE_AVAILABLE=False,
            COMPILE=False,
        )
        old_esmc_cache = adapter_module._LOCAL_ESMC_CACHE
        adapter_module._LOCAL_ESMC_CACHE = None
        try:
            with patch.object(adapter_module, "_LOCAL_CRITIC_LM_DROPOUT", 0.5):
                runtime = adapter_module._load_local_runtime_models(
                    fake_module,
                    inversion_model_name="same-model",
                    critic_name="same-model",
                )
        finally:
            adapter_module._LOCAL_ESMC_CACHE = old_esmc_cache

        self.assertEqual(
            fold_from_pretrained_calls,
            [("biohub/same-model", False)],
        )
        self.assertIs(runtime.inversion_models["same-model"], fold_instances[0])
        self.assertIs(runtime.critic_models["same-model"], fold_instances[0])

    def test_local_runtime_model_loader_keeps_different_model_specs_separate(self) -> None:
        fold_from_pretrained_calls: list[tuple[str, bool]] = []

        class FakeFoldModel:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(esmc_id="esmc")
                self._esmc = None

            @classmethod
            def from_pretrained(cls, repo_id: str, *, load_esmc: bool):
                fold_from_pretrained_calls.append((repo_id, load_esmc))
                return cls()

            def load_esmc(self, esmc_id: str) -> None:
                self._esmc = object()

            def configure_lm_dropout(self, *args, **kwargs):
                return None

            def set_kernel_backend(self, backend):
                return None

            def to(self, *, device: str):
                return self

            def eval(self):
                return self

            def requires_grad_(self, requires_grad: bool):
                return self

        class FakeESMCModel:
            def cuda(self):
                return self

            def eval(self):
                return self

            def requires_grad_(self, requires_grad: bool):
                return self

        class FakeESMCForMaskedLM:
            @staticmethod
            def from_pretrained(repo_id: str, *, torch_dtype):
                return FakeESMCModel()

        fake_module = types.SimpleNamespace(
            ESMFold2ExperimentalModel=FakeFoldModel,
            ESMCForMaskedLM=FakeESMCForMaskedLM,
            torch=types.SimpleNamespace(float32="float32"),
            CUE_AVAILABLE=False,
            COMPILE=False,
        )
        old_esmc_cache = adapter_module._LOCAL_ESMC_CACHE
        adapter_module._LOCAL_ESMC_CACHE = None
        try:
            adapter_module._load_local_runtime_models(
                fake_module,
                inversion_model_name="same-model",
                critic_name="same-model",
            )
        finally:
            adapter_module._LOCAL_ESMC_CACHE = old_esmc_cache

        self.assertEqual(
            fold_from_pretrained_calls,
            [("biohub/same-model", False), ("biohub/same-model", False)],
        )

    def test_model_preflight_uses_local_runtime_by_default(self) -> None:
        fake_runtime = object()
        fake_models = types.SimpleNamespace(
            inversion_models={"inv-model": object()},
            critic_models={"critic-model": object()},
            esmc_model=object(),
        )
        with patch.dict("os.environ", {}, clear=True), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_esm_folding_runtime",
            return_value=fake_runtime,
        ) as load_runtime, patch(
            "esmfold2_pipeline.esm_adapter.binder_design._load_local_runtime_models",
            return_value=fake_models,
        ) as load_models, patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
            side_effect=AssertionError("tutorial module should not load"),
        ):
            result = preflight_models(
                esm_repo="/tmp/esm",
                gpu_id=None,
                inversion_model_name="inv-model",
                critic_name="critic-model",
            )

        load_runtime.assert_called_once_with("/tmp/esm")
        load_models.assert_called_once_with(
            fake_runtime,
            inversion_model_name="inv-model",
            critic_name="critic-model",
        )
        self.assertEqual(result.loaded_inversion_models, ["inv-model"])
        self.assertEqual(result.loaded_critic_models, ["critic-model"])
        self.assertTrue(result.esmc_loaded)

    def test_tutorial_model_preflight_disables_hf_xet_by_default(self) -> None:
        fake_module = types.SimpleNamespace(
            STEPS=150,
            LOG_INTERVAL=5,
            ESMFold2Design=FakeDesignApp,
        )

        with patch.dict(
            "os.environ",
            {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "tutorial"},
            clear=True,
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
            return_value=fake_module,
        ):
            preflight_models(
                esm_repo=None,
                gpu_id=None,
                inversion_model_name="ESMFold2-Experimental-Fast",
                critic_name=DEFAULT_CRITIC_NAME,
            )
            self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "1")

    def test_design_backend_rejects_unknown_value(self) -> None:
        with patch.dict("os.environ", {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "bogus"}):
            with self.assertRaisesRegex(ValueError, "ESMFOLD2_PIPELINE_DESIGN_BACKEND"):
                adapter_module._design_backend()

    def test_local_design_backend_reports_missing_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "local"},
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_esm_folding_runtime",
            side_effect=RuntimeError("missing ESM runtime"),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing ESM runtime"):
                run_binder_design_artifact(
                    campaign_dir=tmpdir,
                    candidate_id="cand_000000_0000",
                    shard_id="shard_000000",
                    seed=0,
                    esm_repo=None,
                    gpu_id=None,
                    steps=1,
                    target_name="ctla4",
                    binder_name="minibinder",
                    critic_name=DEFAULT_CRITIC_NAME,
                )

    def test_run_binder_design_artifact_defaults_to_local_backend(self) -> None:
        def fake_local_design(
            spec,
            *,
            prompt_plan,
            target_sequence_for_design,
            target_geometry_drift_indices,
        ) -> DesignRunResult:
            self.assertEqual(spec.candidate_id, "cand_000000_0000")
            self.assertEqual(prompt_plan.binder_sequence, "##")
            self.assertEqual(target_sequence_for_design, "TARGET")
            self.assertEqual(target_geometry_drift_indices, ())
            return DesignRunResult(
                best_sequences=["TARGET|LOCAL"],
                trajectory={},
                critic_results=[
                    {
                        "critic_name": DEFAULT_CRITIC_NAME,
                        "batch_idx": 0,
                        "designed_sequence": "TARGET|LOCAL",
                        "complex": FakeComplex(),
                        "final_loss": 2.5,
                        "iptm": 0.42,
                    }
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {},
            clear=True,
        ), patch.object(
            adapter_module,
            "_run_local_design",
            side_effect=fake_local_design,
        ) as run_local_design, patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
            side_effect=AssertionError("tutorial module should not load"),
        ):
            result = run_binder_design_artifact(
                campaign_dir=tmpdir,
                candidate_id="cand_000000_0000",
                shard_id="shard_000000",
                seed=0,
                esm_repo=None,
                gpu_id=None,
                steps=1,
                target_name="custom_sequence_target",
                target_sequence="TARGET",
                binder_name="minibinder",
                binder_length_range=(2, 2),
                critic_name=DEFAULT_CRITIC_NAME,
            )

        run_local_design.assert_called_once()
        self.assertEqual(result.designed_sequence, "LOCAL")
        self.assertEqual(result.critic_metrics["iptm"], 0.42)

    def test_local_design_backend_shapes_artifact_without_tutorial_loop(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_local_design(
            spec,
            *,
            prompt_plan,
            target_sequence_for_design,
            target_geometry_drift_indices,
        ) -> DesignRunResult:
            calls.append(
                {
                    "candidate_id": spec.candidate_id,
                    "prompt_binder_name": prompt_plan.binder_name,
                    "prompt_binder_sequence": prompt_plan.binder_sequence,
                    "target_sequence_for_design": target_sequence_for_design,
                    "target_geometry_drift_indices": target_geometry_drift_indices,
                }
            )
            return DesignRunResult(
                best_sequences=["TARGET|LOCAL"],
                trajectory={},
                critic_results=[
                    {
                        "critic_name": DEFAULT_CRITIC_NAME,
                        "batch_idx": 0,
                        "designed_sequence": "TARGET|LOCAL",
                        "complex": FakeComplex(),
                        "final_loss": 2.5,
                        "iptm": 0.42,
                        "distogram_iptm_proxy": 0.33,
                    }
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "local"},
        ), patch.object(
            adapter_module,
            "_run_local_design",
            side_effect=fake_local_design,
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
            side_effect=AssertionError("tutorial module should not load"),
        ):
            result = run_binder_design_artifact(
                campaign_dir=tmpdir,
                candidate_id="cand_000000_0000",
                shard_id="shard_000000",
                seed=0,
                esm_repo=None,
                gpu_id=None,
                steps=1,
                target_name="custom_sequence_target",
                target_sequence="TARGET",
                binder_name="minibinder",
                binder_length_range=(2, 2),
                critic_name=DEFAULT_CRITIC_NAME,
            )

            self.assertEqual(result.designed_sequence, "LOCAL")
            self.assertEqual(
                result.structure_path,
                "esmfold2/structures/cand_000000_0000.pdb",
            )
            self.assertIn(
                "FAKE GPU SMOKE COMPLEX",
                (Path(tmpdir) / result.structure_path).read_text(),
            )

        self.assertEqual(
            calls,
            [
                {
                    "candidate_id": "cand_000000_0000",
                    "prompt_binder_name": None,
                    "prompt_binder_sequence": "##",
                    "target_sequence_for_design": "TARGET",
                    "target_geometry_drift_indices": (),
                }
            ],
        )
        self.assertEqual(result.design_metrics["target_input_mode"], "sequence")
        self.assertEqual(result.design_metrics["target_sequence_length"], 6)
        self.assertEqual(result.design_metrics["final_loss"], 2.5)
        self.assertEqual(result.critic_metrics["iptm"], 0.42)
        self.assertEqual(result.critic_metrics["distogram_iptm_proxy"], 0.33)

    def test_local_design_backend_shapes_scfv_cdr_artifact_without_tutorial_loop(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_local_design(
            spec,
            *,
            prompt_plan,
            target_sequence_for_design,
            target_geometry_drift_indices,
        ) -> DesignRunResult:
            calls.append(
                {
                    "prompt_binder_sequence": prompt_plan.binder_sequence,
                    "prompt_cdr_indices": prompt_plan.cdr_indices,
                    "prompt_cdr_lengths": prompt_plan.cdr_lengths,
                    "prompt_cdr_report_names": prompt_plan.cdr_report_names,
                    "target_sequence_for_design": target_sequence_for_design,
                    "target_geometry_drift_indices": target_geometry_drift_indices,
                }
            )
            return DesignRunResult(
                best_sequences=["TARGET|EVAAQBSS"],
                trajectory={},
                critic_results=[
                    {
                        "critic_name": DEFAULT_CRITIC_NAME,
                        "batch_idx": 0,
                        "designed_sequence": "TARGET|EVAAQBSS",
                        "complex": FakeComplex(),
                        "final_loss": 2.5,
                        "iptm": 0.42,
                    }
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "local"},
        ), patch.object(
            adapter_module,
            "_run_local_design",
            side_effect=fake_local_design,
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
            side_effect=AssertionError("tutorial module should not load"),
        ):
            result = run_binder_design_artifact(
                campaign_dir=tmpdir,
                candidate_id="cand_000000_0000",
                shard_id="shard_000000",
                seed=0,
                esm_repo=None,
                gpu_id=None,
                steps=1,
                target_name="custom_sequence_target",
                target_sequence="TARGET",
                binder_name="lab_template",
                binder_scaffold="scfv",
                binder_framework_name="lab_template",
                binder_framework_source="template",
                binder_framework_template="EV{hcdr1}Q{hcdr2}SS",
                binder_framework_cdr_lengths={
                    "hcdr1": (2, 2),
                    "hcdr2": (1, 1),
                },
                critic_name=DEFAULT_CRITIC_NAME,
            )

        self.assertEqual(result.designed_sequence, "EVAAQBSS")
        self.assertTrue(result.design_metrics["is_antibody"])
        self.assertEqual(result.design_metrics["binder_type"], "scfv")
        self.assertEqual(result.design_metrics["cdr_indices"], [2, 3, 5])
        self.assertEqual(
            result.design_metrics["cdr_lengths"],
            {"hcdr1": 2, "hcdr2": 1},
        )
        self.assertEqual(
            result.design_metrics["cdr_sequences"],
            {"hcdr1": "AA", "hcdr2": "B"},
        )
        self.assertEqual(
            calls,
            [
                {
                    "prompt_binder_sequence": "EV##Q#SS",
                    "prompt_cdr_indices": (2, 3, 5),
                    "prompt_cdr_lengths": {"hcdr1": 2, "hcdr2": 1},
                    "prompt_cdr_report_names": ("hcdr1", "hcdr2"),
                    "target_sequence_for_design": "TARGET",
                    "target_geometry_drift_indices": (),
                }
            ],
        )

    def test_local_design_backend_shapes_custom_sequence_cdr_artifact_without_tutorial_loop(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_local_design(
            spec,
            *,
            prompt_plan,
            target_sequence_for_design,
            target_geometry_drift_indices,
        ) -> DesignRunResult:
            calls.append(
                {
                    "prompt_binder_sequence": prompt_plan.binder_sequence,
                    "prompt_cdr_indices": prompt_plan.cdr_indices,
                    "prompt_cdr_lengths": prompt_plan.cdr_lengths,
                    "prompt_cdr_report_names": prompt_plan.cdr_report_names,
                    "target_sequence_for_design": target_sequence_for_design,
                    "target_geometry_drift_indices": target_geometry_drift_indices,
                }
            )
            return DesignRunResult(
                best_sequences=["TARGET|EVAAQBSS"],
                trajectory={},
                critic_results=[
                    {
                        "critic_name": DEFAULT_CRITIC_NAME,
                        "batch_idx": 0,
                        "designed_sequence": "TARGET|EVAAQBSS",
                        "complex": FakeComplex(),
                        "final_loss": 2.5,
                        "iptm": 0.42,
                    }
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "local"},
        ), patch.object(
            adapter_module,
            "_run_local_design",
            side_effect=fake_local_design,
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
            side_effect=AssertionError("tutorial module should not load"),
        ):
            result = run_binder_design_artifact(
                campaign_dir=tmpdir,
                candidate_id="cand_000000_0000",
                shard_id="shard_000000",
                seed=0,
                esm_repo=None,
                gpu_id=None,
                steps=1,
                target_name="custom_sequence_target",
                target_sequence="TARGET",
                binder_name="lab_fixed",
                binder_scaffold="scfv",
                binder_framework_name="lab_fixed",
                binder_framework_source="sequence",
                binder_framework_sequence="EVAAQBSS",
                binder_framework_cdr_indices=(2, 3, 5),
                critic_name=DEFAULT_CRITIC_NAME,
            )

        self.assertEqual(result.designed_sequence, "EVAAQBSS")
        self.assertEqual(result.design_metrics["framework_source"], "sequence")
        self.assertEqual(result.design_metrics["cdr_indices"], [2, 3, 5])
        self.assertEqual(
            result.design_metrics["cdr_lengths"],
            {"hcdr1": 2, "hcdr2": 1},
        )
        self.assertEqual(
            result.design_metrics["cdr_sequences"],
            {"hcdr1": "AA", "hcdr2": "B"},
        )
        self.assertEqual(
            calls,
            [
                {
                    "prompt_binder_sequence": "EV##Q#SS",
                    "prompt_cdr_indices": (2, 3, 5),
                    "prompt_cdr_lengths": {"hcdr1": 2, "hcdr2": 1},
                    "prompt_cdr_report_names": ("hcdr1", "hcdr2"),
                    "target_sequence_for_design": "TARGET",
                    "target_geometry_drift_indices": (),
                }
            ],
        )

    def test_local_design_backend_shapes_vhh_cdr_artifact_without_tutorial_loop(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_local_design(
            spec,
            *,
            prompt_plan,
            target_sequence_for_design,
            target_geometry_drift_indices,
        ) -> DesignRunResult:
            calls.append(
                {
                    "prompt_binder_sequence": prompt_plan.binder_sequence,
                    "prompt_cdr_indices": prompt_plan.cdr_indices,
                    "prompt_cdr_lengths": prompt_plan.cdr_lengths,
                    "prompt_cdr_report_names": prompt_plan.cdr_report_names,
                    "target_sequence_for_design": target_sequence_for_design,
                    "target_geometry_drift_indices": target_geometry_drift_indices,
                }
            )
            return DesignRunResult(
                best_sequences=["TARGET|EVAAQBSSCCC"],
                trajectory={},
                critic_results=[
                    {
                        "critic_name": DEFAULT_CRITIC_NAME,
                        "batch_idx": 0,
                        "designed_sequence": "TARGET|EVAAQBSSCCC",
                        "complex": FakeComplex(),
                        "final_loss": 2.5,
                        "iptm": 0.42,
                    }
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "local"},
        ), patch.object(
            adapter_module,
            "_run_local_design",
            side_effect=fake_local_design,
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
            side_effect=AssertionError("tutorial module should not load"),
        ):
            result = run_binder_design_artifact(
                campaign_dir=tmpdir,
                candidate_id="cand_000000_0000",
                shard_id="shard_000000",
                seed=0,
                esm_repo=None,
                gpu_id=None,
                steps=1,
                target_name="custom_sequence_target",
                target_sequence="TARGET",
                binder_name="caplacizumab_framework_vhh",
                binder_scaffold="vhh",
                binder_framework_name="caplacizumab_framework_vhh",
                binder_framework_source="builtin",
                binder_framework_template="EV{cdr1}Q{cdr2}SS{cdr3}",
                binder_framework_cdr_lengths={
                    "cdr1": (2, 2),
                    "cdr2": (1, 1),
                    "cdr3": (3, 3),
                },
                critic_name=DEFAULT_CRITIC_NAME,
            )

        self.assertEqual(result.designed_sequence, "EVAAQBSSCCC")
        self.assertTrue(result.design_metrics["is_antibody"])
        self.assertEqual(result.design_metrics["binder_type"], "vhh")
        self.assertEqual(result.design_metrics["framework_source"], "builtin")
        self.assertEqual(result.design_metrics["cdr_indices"], [2, 3, 5, 8, 9, 10])
        self.assertEqual(
            result.design_metrics["cdr_lengths"],
            {"hcdr1": 2, "hcdr2": 1, "hcdr3": 3},
        )
        self.assertEqual(
            result.design_metrics["cdr_sequences"],
            {"hcdr1": "AA", "hcdr2": "B", "hcdr3": "CCC"},
        )
        self.assertEqual(
            calls,
            [
                {
                    "prompt_binder_sequence": "EV##Q#SS###",
                    "prompt_cdr_indices": (2, 3, 5, 8, 9, 10),
                    "prompt_cdr_lengths": {
                        "hcdr1": 2,
                        "hcdr2": 1,
                        "hcdr3": 3,
                    },
                    "prompt_cdr_report_names": ("hcdr1", "hcdr2", "hcdr3"),
                    "target_sequence_for_design": "TARGET",
                    "target_geometry_drift_indices": (),
                }
            ],
        )

    def test_local_design_backend_wires_runtime_into_local_gradient_loop(self) -> None:
        captured: dict[str, object] = {}

        class FakeLocalApp:
            inversion_models = {"inv": "inversion-model"}
            critic_models = {DEFAULT_CRITIC_NAME: "critic-model"}
            esmc_model = "esmc-model"

        complex_builder = object()
        fake_module = types.SimpleNamespace(
            TARGET_SEQUENCES={"ctla4": "TARGETSEQ"},
            fold_and_get_distogram=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("tutorial fold helper should not be used")
            ),
            build_complex=complex_builder,
            torch="torch-runtime",
            F="functional-runtime",
            optim="optim-runtime",
            seed_context="seed-context",
            ESMCTokenizer="tokenizer-runtime",
            get_mid_points=lambda: "bin-midpoints",
        )

        def fake_gradient_loop(**kwargs) -> DesignRunResult:
            captured.update(kwargs)
            return DesignRunResult(
                best_sequences=["TARGETSEQ|AA"],
                trajectory={},
                critic_results=[
                    {
                        "critic_name": DEFAULT_CRITIC_NAME,
                        "batch_idx": 0,
                        "designed_sequence": "TARGETSEQ|AA",
                        "complex": FakeComplex(),
                        "final_loss": 1.0,
                        "iptm": 0.5,
                    }
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "local"},
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_esm_folding_runtime",
            return_value=fake_module,
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
            side_effect=AssertionError("tutorial module should not load"),
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design._load_local_runtime_models",
            return_value=FakeLocalApp(),
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.run_gradient_design_loop",
            side_effect=fake_gradient_loop,
        ):
            result = run_binder_design_artifact(
                campaign_dir=tmpdir,
                candidate_id="cand_000000_0000",
                shard_id="shard_000000",
                seed=3,
                esm_repo=None,
                gpu_id=None,
                steps=2,
                target_name="ctla4",
                binder_name="minibinder",
                binder_length_range=(2, 2),
                critic_name=DEFAULT_CRITIC_NAME,
            )

        self.assertEqual(result.designed_sequence, "AA")
        self.assertEqual(captured["target_sequence"], "TARGETSEQ")
        self.assertEqual(captured["binder_sequence"], "##")
        self.assertEqual(captured["seed"], 3)
        self.assertEqual(captured["steps"], 2)
        self.assertEqual(captured["inversion_models"], {"inv": "inversion-model"})
        self.assertEqual(
            captured["critic_models"],
            {DEFAULT_CRITIC_NAME: "critic-model"},
        )
        with patch.object(
            adapter_module,
            "_fold_and_get_distogram_for_sequence_target",
            return_value={"seq_list": ["TARGETSEQ|AA"]},
        ) as sequence_fold:
            direct_fold = captured["fold_complex"](
                "model",
                "TARGETSEQ",
                "target-one-hot",
                "design",
                num_loops=1,
                num_sampling_steps=50,
                calculate_confidence=True,
                seed=4,
            )
        self.assertEqual(direct_fold, {"seq_list": ["TARGETSEQ|AA"]})
        sequence_fold.assert_called_once_with(
            fake_module,
            "model",
            "TARGETSEQ",
            "target-one-hot",
            "design",
            num_loops=1,
            num_sampling_steps=50,
            calculate_confidence=True,
            seed=4,
        )
        with patch.object(
            adapter_module.design_plm,
            "compute_esmc_pseudoperplexity_nll",
            return_value="plm-loss",
        ) as local_plm:
            plm_result = captured["compute_plm_loss"](
                esmc_model="esmc",
                binder_design="binder-design",
                score_mask="score-mask",
                batch_size=4,
                n_passes=4,
            )
        self.assertEqual(plm_result, "plm-loss")
        local_plm.assert_called_once_with(
            esmc_model="esmc",
            binder_design="binder-design",
            score_mask="score-mask",
            batch_size=4,
            n_passes=4,
            torch_module="torch-runtime",
            functional="functional-runtime",
            tokenizer_factory="tokenizer-runtime",
        )
        self.assertIs(captured["build_complex"], fake_module.build_complex)
        with patch.object(
            adapter_module.design_metrics,
            "compute_distogram_iptm_proxy",
            return_value={"distogram_iptm_proxy": 0.6},
        ) as local_proxy, patch.object(
            adapter_module.design_losses,
            "get_mid_points",
            return_value="bin-midpoints",
        ) as local_midpoints:
            proxy_result = captured["compute_distogram_iptm_proxy"](
                "distogram-logits",
                9,
                "AA",
                False,
            )
        self.assertEqual(proxy_result, {"distogram_iptm_proxy": 0.6})
        local_midpoints.assert_called_once_with("torch-runtime")
        local_proxy.assert_called_once_with(
            "distogram-logits",
            9,
            "AA",
            False,
            cdr_indices=None,
            bin_distance="bin-midpoints",
            torch_module="torch-runtime",
        )

    def test_local_design_backend_uses_structure_target_fold_callback(self) -> None:
        captured: dict[str, object] = {}

        class FakeLocalApp:
            inversion_models = {"inv": "inversion-model"}
            critic_models = {DEFAULT_CRITIC_NAME: "critic-model"}
            esmc_model = "esmc-model"

        fake_module = types.SimpleNamespace(
            TARGET_SEQUENCES={},
            fold_and_get_distogram=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("direct tutorial fold should not be used")
            ),
            build_complex=object(),
            compute_distogram_iptm_proxy=object(),
            torch=object(),
            F=object(),
            optim=object(),
            seed_context=object(),
            ESMCTokenizer=object(),
            get_mid_points=lambda: "bin-midpoints",
        )

        def fake_gradient_loop(**kwargs) -> DesignRunResult:
            captured.update(kwargs)
            return DesignRunResult(
                best_sequences=["GS|AA"],
                trajectory={},
                critic_results=[
                    {
                        "critic_name": DEFAULT_CRITIC_NAME,
                        "batch_idx": 0,
                        "designed_sequence": "GS|AA",
                        "complex": FakeComplex(),
                        "final_loss": 1.0,
                        "iptm": 0.5,
                    }
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A",),
                    conditioning_mode="distogram",
                )
            )

            with patch.dict(
                "os.environ",
                {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "local"},
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design.load_esm_folding_runtime",
                return_value=fake_module,
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design._load_local_runtime_models",
                return_value=FakeLocalApp(),
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design.run_gradient_design_loop",
                side_effect=fake_gradient_loop,
            ):
                result = run_binder_design_artifact(
                    campaign_dir=root,
                    candidate_id="cand_000000_0000",
                    shard_id="shard_000000",
                    seed=3,
                    esm_repo=None,
                    gpu_id=None,
                    steps=2,
                    target_name=None,
                    binder_name="minibinder",
                    binder_length_range=(2, 2),
                    critic_name=DEFAULT_CRITIC_NAME,
                    structure_target=structure_target,
                    conditioning_mode="distogram",
                )

            with patch(
                "esmfold2_pipeline.esm_adapter.binder_design."
                "_fold_and_get_distogram_for_structure_target",
                return_value={"seq_list": ["GS|AA"]},
            ) as fold_helper:
                structure_fold = captured["fold_complex"](
                    "model",
                    "GS",
                    "target-one-hot",
                    "design",
                    num_loops=1,
                    num_sampling_steps=50,
                    calculate_confidence=True,
                    seed=4,
                )

        self.assertEqual(result.designed_sequence, "AA")
        self.assertEqual(captured["target_sequence"], "GS")
        self.assertEqual(captured["binder_sequence"], "##")
        self.assertEqual(structure_fold, {"seq_list": ["GS|AA"]})
        fold_helper.assert_called_once()
        self.assertIs(fold_helper.call_args.args[0], fake_module)
        self.assertEqual(fold_helper.call_args.args[1], "model")
        self.assertEqual(fold_helper.call_args.args[2], "target-one-hot")
        self.assertEqual(fold_helper.call_args.args[3], "design")
        self.assertIs(fold_helper.call_args.kwargs["structure_target"], structure_target)
        self.assertTrue(fold_helper.call_args.kwargs["condition_distograms"])
        self.assertFalse(fold_helper.call_args.kwargs["condition_assembly"])
        self.assertIsNone(fold_helper.call_args.kwargs["conditioning_chain_pairs"])
        self.assertEqual(fold_helper.call_args.kwargs["num_loops"], 1)
        self.assertEqual(fold_helper.call_args.kwargs["num_sampling_steps"], 50)
        self.assertTrue(fold_helper.call_args.kwargs["calculate_confidence"])
        self.assertEqual(fold_helper.call_args.kwargs["seed"], 4)

    def test_local_fold_callback_preserves_multichain_conditioning_options(self) -> None:
        fake_module = types.SimpleNamespace()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    conditioning_mode="distogram",
                    conditioning_assembly=True,
                    conditioning_chain_pairs=(("A", "C"),),
                )
            )
            fold_complex = adapter_module._local_fold_complex_callback(
                fake_module,
                structure_target=structure_target,
                condition_distograms=True,
                condition_assembly=True,
                conditioning_chain_pairs=(("A", "C"),),
            )

            with patch.object(
                adapter_module,
                "_fold_and_get_distogram_for_structure_target",
                return_value={"seq_list": ["GS|GT|AA"]},
            ) as fold_helper:
                result = fold_complex(
                    "model",
                    "GSGT",
                    "target-one-hot",
                    "design",
                    num_loops=1,
                    num_sampling_steps=50,
                    calculate_confidence=True,
                    seed=4,
                )

        self.assertEqual(result, {"seq_list": ["GS|GT|AA"]})
        fold_helper.assert_called_once()
        self.assertIs(fold_helper.call_args.args[0], fake_module)
        self.assertEqual(fold_helper.call_args.args[1], "model")
        self.assertEqual(fold_helper.call_args.args[2], "target-one-hot")
        self.assertEqual(fold_helper.call_args.args[3], "design")
        self.assertIs(fold_helper.call_args.kwargs["structure_target"], structure_target)
        self.assertTrue(fold_helper.call_args.kwargs["condition_distograms"])
        self.assertTrue(fold_helper.call_args.kwargs["condition_assembly"])
        self.assertEqual(
            fold_helper.call_args.kwargs["conditioning_chain_pairs"],
            (("A", "C"),),
        )
        self.assertEqual(fold_helper.call_args.kwargs["num_loops"], 1)
        self.assertEqual(fold_helper.call_args.kwargs["num_sampling_steps"], 50)
        self.assertTrue(fold_helper.call_args.kwargs["calculate_confidence"])
        self.assertEqual(fold_helper.call_args.kwargs["seed"], 4)

    def test_local_design_backend_shapes_multichain_structure_metadata_without_tutorial_loop(self) -> None:
        calls: list[dict[str, object]] = []
        written_pdb = ""

        class FakePredictedMultichainComplex:
            def to_pdb_string(self) -> str:
                return "".join(
                    [
                        _pdb_atom_line(1, "N", "GLY", "X", 1, "", 0.0, 0.0, 0.0),
                        _pdb_atom_line(2, "N", "SER", "X", 2, "", 1.0, 0.0, 0.0),
                        _pdb_atom_line(3, "N", "GLY", "Y", 1, "", 0.0, 8.0, 0.0),
                        _pdb_atom_line(4, "N", "THR", "Y", 2, "", 1.0, 8.0, 0.0),
                        _pdb_atom_line(5, "N", "ALA", "Z", 1, "", 0.0, 16.0, 0.0),
                        _pdb_atom_line(6, "N", "ALA", "Z", 2, "", 1.0, 16.0, 0.0),
                    ]
                )

        def fake_local_design(
            spec,
            *,
            prompt_plan,
            target_sequence_for_design,
            target_geometry_drift_indices,
        ) -> DesignRunResult:
            calls.append(
                {
                    "prompt_binder_sequence": prompt_plan.binder_sequence,
                    "target_sequence_for_design": target_sequence_for_design,
                    "conditioning_mode": spec.conditioning_mode,
                    "conditioning_assembly": spec.conditioning_assembly,
                    "conditioning_chain_pairs": spec.conditioning_chain_pairs,
                    "target_geometry_drift_indices": target_geometry_drift_indices,
                }
            )
            return DesignRunResult(
                best_sequences=["GS|GT|AA"],
                trajectory={},
                critic_results=[
                    {
                        "critic_name": DEFAULT_CRITIC_NAME,
                        "batch_idx": 0,
                        "designed_sequence": "GS|GT|AA",
                        "complex": FakePredictedMultichainComplex(),
                        "final_loss": 1.0,
                        "iptm": 0.5,
                    }
                ],
                last_confidence_fold={
                    "seq_list": ["GS|GT|AA"],
                    "plddt": np.array([[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]),
                },
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    hotspots={"A": ("2",), "C": ("1",)},
                    conditioning_mode="distogram",
                    conditioning_assembly=True,
                    conditioning_chain_pairs=(("A", "C"),),
                )
            )

            with patch.dict(
                "os.environ",
                {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "local"},
            ), patch.object(
                adapter_module,
                "_run_local_design",
                side_effect=fake_local_design,
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
                side_effect=AssertionError("tutorial module should not load"),
            ), patch.object(
                adapter_module,
                "_binder_target_iptm_metrics_from_capture",
                return_value={},
            ), patch.object(
                adapter_module,
                "_target_geometry_metrics_from_capture",
                return_value={},
            ), patch.object(
                adapter_module,
                "_hotspot_metrics_from_capture",
                return_value={},
            ):
                result = run_binder_design_artifact(
                    campaign_dir=root,
                    candidate_id="cand_000000_0000",
                    shard_id="shard_000000",
                    seed=3,
                    esm_repo=None,
                    gpu_id=None,
                    steps=2,
                    target_name=None,
                    binder_name="minibinder",
                    binder_length_range=(2, 2),
                    critic_name=DEFAULT_CRITIC_NAME,
                    structure_target=structure_target,
                    conditioning_mode="distogram",
                    conditioning_assembly=True,
                    conditioning_chain_pairs=(("A", "C"),),
                )
                written_pdb = (root / result.structure_path).read_text()

        self.assertEqual(result.designed_sequence, "AA")
        self.assertEqual(
            calls,
            [
                {
                    "prompt_binder_sequence": "##",
                    "target_sequence_for_design": "GSGT",
                    "conditioning_mode": "distogram",
                    "conditioning_assembly": True,
                    "conditioning_chain_pairs": (("A", "C"),),
                    "target_geometry_drift_indices": (),
                }
            ],
        )
        self.assertEqual(result.design_metrics["target_input_mode"], "structure")
        self.assertEqual(result.design_metrics["target_sequence_length"], 4)
        self.assertEqual(result.design_metrics["designed_target_sequence"], "GS|GT")
        self.assertEqual(result.design_metrics["binder_chain_id"], "B")
        self.assertEqual(result.design_metrics["target_chain_ids"], ["A", "C"])
        self.assertEqual(
            result.design_metrics["target_chain_spans"],
            [
                {"chain_id": "A", "start": 0, "end": 2, "length": 2},
                {"chain_id": "C", "start": 2, "end": 4, "length": 2},
            ],
        )
        self.assertEqual(result.design_metrics["target_hotspots"], {"A": [1], "C": [0]})
        self.assertEqual(result.design_metrics["target_hotspot_indices"], [1, 2])
        self.assertTrue(result.design_metrics["target_conditioning_assembly"])
        self.assertEqual(
            result.design_metrics["target_conditioning_chain_pairs"],
            [["A", "C"]],
        )
        atom_lines = [
            line for line in written_pdb.splitlines() if line.startswith("ATOM")
        ]
        self.assertEqual([line[21] for line in atom_lines], ["A", "A", "C", "C", "B", "B"])
        self.assertEqual(
            [float(line[60:66]) for line in atom_lines],
            [80.0, 70.0, 60.0, 50.0, 40.0, 30.0],
        )
        self.assertEqual(result.critic_metrics["plddt_target"], 65.0)
        self.assertEqual(result.critic_metrics["plddt_binder"], 35.0)

    def test_local_structure_loss_callback_prepares_design_loss_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A",),
                    hotspots={"A": ("1",)},
                    conditioning_mode="none",
                )
            )

        fake_binder_design = types.SimpleNamespace(
            torch="torch-runtime",
            get_mid_points=lambda: "bin-midpoints",
        )

        with patch.object(
            adapter_module.design_losses,
            "compute_design_structure_losses",
            return_value={"total_loss": 3.0},
        ) as design_losses, patch.object(
            adapter_module.design_losses,
            "get_mid_points",
            return_value="bin-midpoints",
        ) as local_midpoints:
            callback = adapter_module._local_structure_loss_callback(
                fake_binder_design,
                structure_target=structure_target,
                target_geometry_drift=TargetGeometryDriftConfig(
                    enabled=True,
                    weight=0.5,
                    tolerance_angstrom=1.0,
                    stiffness_angstrom=2.0,
                    regions=None,
                ),
                target_geometry_drift_indices=(0, 1),
                hotspot_contact_weight=0.25,
                hotspot_distogram_contact_cutoff_angstrom=20.0,
                hotspot_num_contacts=2,
                hotspot_contact_probability_target=0.6,
                hotspot_loss_mode="probability_hinge",
                binder_contact_indices=(1,),
            )
            result = callback("distogram-logits", binder_length=5)

        self.assertEqual(result, {"total_loss": 3.0})
        local_midpoints.assert_called_once_with("torch-runtime")
        design_losses.assert_called_once()
        self.assertEqual(
            design_losses.call_args.args,
            ("distogram-logits", 5),
        )
        self.assertEqual(
            design_losses.call_args.kwargs["torch_module"],
            "torch-runtime",
        )
        self.assertEqual(
            design_losses.call_args.kwargs["bin_distance"],
            "bin-midpoints",
        )
        self.assertEqual(
            design_losses.call_args.kwargs["target_geometry_weight"],
            0.5,
        )
        self.assertEqual(
            design_losses.call_args.kwargs["target_geometry_tolerance_angstrom"],
            1.0,
        )
        self.assertEqual(
            design_losses.call_args.kwargs["target_geometry_stiffness_angstrom"],
            2.0,
        )
        self.assertEqual(
            design_losses.call_args.kwargs["target_geometry_reference_distances"].shape,
            (2, 2),
        )
        self.assertEqual(
            int(design_losses.call_args.kwargs["target_geometry_pair_mask"].sum()),
            1,
        )
        self.assertEqual(
            design_losses.call_args.kwargs["hotspot_indices"],
            (0,),
        )
        self.assertEqual(
            design_losses.call_args.kwargs["hotspot_contact_weight"],
            0.25,
        )
        self.assertEqual(
            design_losses.call_args.kwargs["hotspot_contact_cutoff_angstrom"],
            20.0,
        )
        self.assertEqual(
            design_losses.call_args.kwargs["hotspot_num_contacts"],
            2,
        )
        self.assertEqual(
            design_losses.call_args.kwargs["hotspot_contact_probability_target"],
            0.6,
        )
        self.assertEqual(
            design_losses.call_args.kwargs["hotspot_loss_mode"],
            "probability_hinge",
        )
        self.assertEqual(
            design_losses.call_args.kwargs["binder_contact_indices"],
            (1,),
        )

    def test_hotspot_scores_use_local_midpoints_without_tutorial_helper(self) -> None:
        fake_runtime = types.SimpleNamespace(torch="torch-runtime")

        with patch.object(
            adapter_module.design_losses,
            "get_mid_points",
            return_value="local-midpoints",
        ) as local_midpoints, patch.object(
            adapter_module.design_losses,
            "hotspot_contact_probability_scores",
            return_value="scores",
        ) as hotspot_scores:
            result = adapter_module._hotspot_contact_probability_scores(
                fake_runtime,
                "distogram-logits",
                5,
                hotspot_indices=(0,),
                contact_cutoff_angstrom=20.0,
                hotspot_num_contacts=2,
                binder_contact_indices=(1,),
            )

        self.assertEqual(result, "scores")
        local_midpoints.assert_called_once_with("torch-runtime")
        hotspot_scores.assert_called_once_with(
            "torch-runtime",
            "distogram-logits",
            5,
            hotspot_indices=(0,),
            contact_cutoff_angstrom=20.0,
            hotspot_num_contacts=2,
            binder_contact_indices=(1,),
            bin_distances="local-midpoints",
        )

    def test_local_design_backend_allows_structure_hotspot_and_drift_loss_callback(self) -> None:
        captured: dict[str, object] = {}

        class FakeLocalApp:
            inversion_models = {"inv": "inversion-model"}
            critic_models = {DEFAULT_CRITIC_NAME: "critic-model"}
            esmc_model = "esmc-model"

        fake_module = types.SimpleNamespace(
            TARGET_SEQUENCES={},
            fold_and_get_distogram=lambda *args, **kwargs: {"seq_list": ["GS|AA"]},
            build_complex=object(),
            compute_distogram_iptm_proxy=object(),
            torch=object(),
            F=object(),
            optim=object(),
            seed_context=object(),
            ESMCTokenizer=object(),
            get_mid_points=lambda: "bin-midpoints",
        )

        def fake_gradient_loop(**kwargs) -> DesignRunResult:
            captured.update(kwargs)
            return DesignRunResult(
                best_sequences=["GS|AA"],
                trajectory={},
                critic_results=[
                    {
                        "critic_name": DEFAULT_CRITIC_NAME,
                        "batch_idx": 0,
                        "designed_sequence": "GS|AA",
                        "complex": FakeComplex(),
                        "final_loss": 1.0,
                        "iptm": 0.5,
                    }
                ],
                last_design_fold={
                    "seq_list": ["GS|AA"],
                    "distogram_logits": "distogram-logits",
                },
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A",),
                    hotspots={"A": ("1",)},
                    conditioning_mode="none",
                )
            )

            with patch.dict(
                "os.environ",
                {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "local"},
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design.load_esm_folding_runtime",
                return_value=fake_module,
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design._load_local_runtime_models",
                return_value=FakeLocalApp(),
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design.run_gradient_design_loop",
                side_effect=fake_gradient_loop,
            ), patch.object(
                adapter_module,
                "_hotspot_design_contact_probability_metrics",
                return_value={"hotspot_design_contact_probability_mean": 0.7},
            ) as hotspot_design_metrics:
                result = run_binder_design_artifact(
                    campaign_dir=root,
                    candidate_id="cand_000000_0000",
                    shard_id="shard_000000",
                    seed=3,
                    esm_repo=None,
                    gpu_id=None,
                    steps=2,
                    target_name=None,
                    binder_name="minibinder",
                    binder_length_range=(2, 2),
                    critic_name=DEFAULT_CRITIC_NAME,
                    structure_target=structure_target,
                    conditioning_mode="none",
                    target_geometry_drift=TargetGeometryDriftConfig(
                        enabled=True,
                        weight=0.5,
                    ),
                )

        self.assertEqual(result.designed_sequence, "AA")
        self.assertEqual(captured["target_sequence"], "GS")
        self.assertEqual(captured["binder_sequence"], "##")
        self.assertTrue(callable(captured["compute_structure_losses"]))
        self.assertEqual(
            result.design_metrics["hotspot_design_contact_probability_mean"],
            0.7,
        )
        hotspot_design_metrics.assert_called_once()
        self.assertIs(hotspot_design_metrics.call_args.args[0], fake_module)
        self.assertEqual(
            hotspot_design_metrics.call_args.args[1],
            {"seq_list": ["GS|AA"], "distogram_logits": "distogram-logits"},
        )
        self.assertEqual(
            hotspot_design_metrics.call_args.kwargs["hotspot_indices"],
            (0,),
        )

    def test_adapter_uses_structure_target_sequence_and_conditioning_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A",),
                    conditioning_mode="distogram",
                )
            )
            fake_module = types.SimpleNamespace(
                STEPS=150,
                LOG_INTERVAL=5,
                ESMFold2Design=FakeDesignApp,
                fold_and_get_distogram=lambda *args, **kwargs: None,
                compute_structure_losses=lambda *args, **kwargs: {},
            )

            with patch.dict(
                "os.environ",
                {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "tutorial"},
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
                return_value=fake_module,
            ):
                result = run_binder_design_artifact(
                    campaign_dir=root,
                    candidate_id="cand_000000_0000",
                    shard_id="shard_000000",
                    seed=0,
                    esm_repo=None,
                    gpu_id=None,
                    steps=1,
                    target_name=None,
                    binder_name="minibinder",
                    critic_name=DEFAULT_CRITIC_NAME,
                    structure_target=structure_target,
                    conditioning_mode="distogram",
                )

            self.assertEqual(FakeDesignApp.last_design_kwargs["target_name"], None)
            self.assertEqual(FakeDesignApp.last_design_kwargs["target_sequence"], "GS")
            self.assertIsNone(FakeDesignApp.last_design_kwargs["is_antibody"])
            self.assertEqual(result.designed_sequence, "BINDER")
            self.assertIsNone(result.sequence_path)
            self.assertEqual(result.design_metrics["target_chain_id"], "A")
            self.assertEqual(result.design_metrics["binder_chain_id"], "B")
            self.assertEqual(result.design_metrics["target_length"], 2)
            self.assertEqual(result.design_metrics["target_conditioning_mode"], "distogram")
            self.assertEqual(
                result.design_metrics["inversion_model_name"],
                DEFAULT_INVERSION_MODEL_NAME,
            )
            self.assertEqual(result.design_metrics["critic_name"], DEFAULT_CRITIC_NAME)
            self.assertEqual(
                result.design_metrics["hotspot_distogram_contact_cutoff_angstrom"],
                20.0,
            )
            self.assertEqual(
                result.design_metrics["hotspot_critic_contact_cutoff_angstrom"],
                5.0,
            )
            self.assertEqual(result.design_metrics["hotspot_contact_cutoff_angstrom"], 5.0)

    def test_adapter_uses_direct_target_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_module = types.SimpleNamespace(
                STEPS=150,
                LOG_INTERVAL=5,
                ESMFold2Design=FakeDesignApp,
                fold_and_get_distogram=lambda *args, **kwargs: None,
                compute_structure_losses=lambda *args, **kwargs: {},
            )

            with patch.dict(
                "os.environ",
                {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "tutorial"},
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
                return_value=fake_module,
            ):
                result = run_binder_design_artifact(
                    campaign_dir=root,
                    candidate_id="cand_000000_0000",
                    shard_id="shard_000000",
                    seed=0,
                    esm_repo=None,
                    gpu_id=None,
                    steps=1,
                    target_name="custom_sequence_target",
                    target_sequence="ACDEFGHIK",
                    binder_name="minibinder",
                    critic_name=DEFAULT_CRITIC_NAME,
                )

            self.assertEqual(FakeDesignApp.last_design_kwargs["target_name"], None)
            self.assertEqual(
                FakeDesignApp.last_design_kwargs["target_sequence"],
                "ACDEFGHIK",
            )
            self.assertEqual(result.design_metrics["target_name"], "custom_sequence_target")
            self.assertEqual(result.design_metrics["target_input_mode"], "sequence")
            self.assertEqual(result.design_metrics["target_sequence_length"], 9)
            self.assertEqual(result.designed_sequence, "BINDER")

    def test_adapter_uses_multichain_structure_target_sequence_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    hotspots={"A": ("2",), "C": ("1",)},
                    conditioning_mode="distogram",
                )
            )
            fake_module = types.SimpleNamespace(
                STEPS=150,
                LOG_INTERVAL=5,
                ESMFold2Design=FakeDesignApp,
                fold_and_get_distogram=lambda *args, **kwargs: None,
                compute_structure_losses=lambda *args, **kwargs: {},
            )

            with patch.dict(
                "os.environ",
                {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "tutorial"},
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
                return_value=fake_module,
            ):
                result = run_binder_design_artifact(
                    campaign_dir=root,
                    candidate_id="cand_000000_0000",
                    shard_id="shard_000000",
                    seed=0,
                    esm_repo=None,
                    gpu_id=None,
                    steps=1,
                    target_name=None,
                    binder_name="minibinder",
                    critic_name=DEFAULT_CRITIC_NAME,
                    structure_target=structure_target,
                    conditioning_mode="distogram",
                )

        self.assertEqual(FakeDesignApp.last_design_kwargs["target_name"], None)
        self.assertEqual(FakeDesignApp.last_design_kwargs["target_sequence"], "GSGT")
        self.assertEqual(result.design_metrics["binder_chain_id"], "B")
        self.assertEqual(result.design_metrics["target_chain_ids"], ["A", "C"])
        self.assertEqual(result.design_metrics["target_length"], 4)
        self.assertEqual(
            result.design_metrics["target_chain_spans"],
            [
                {"chain_id": "A", "start": 0, "end": 2, "length": 2},
                {"chain_id": "C", "start": 2, "end": 4, "length": 2},
            ],
        )
        self.assertEqual(result.design_metrics["target_hotspots"], {"A": [1], "C": [0]})
        self.assertEqual(result.design_metrics["target_hotspot_indices"], [1, 2])
        self.assertEqual(
            result.design_metrics["target_hotspot_global_indices"],
            [1, 2],
        )

    def test_complex_sequence_splits_binder_from_multichain_target_suffix(self) -> None:
        target_sequence, binder_sequence = adapter_module._split_complex_sequence(
            "AA|CC|BINDER"
        )

        self.assertEqual(target_sequence, "AA|CC")
        self.assertEqual(binder_sequence, "BINDER")

    def test_antibody_cdr_indices_patch_uses_template_positions_and_restores(self) -> None:
        def original_cdr_indices(_sequence: str) -> list[int]:
            return [-1]

        fake_module = types.SimpleNamespace(_cdr_indices=original_cdr_indices)

        with adapter_module._patched_antibody_cdr_indices(
            fake_module,
            cdr_indices=(1, 2),
            binder_length=4,
        ):
            self.assertEqual(fake_module._cdr_indices("ABCD"), [1, 2])
            self.assertEqual(fake_module._cdr_indices("ABCDE"), [-1])

        self.assertIs(fake_module._cdr_indices, original_cdr_indices)

    def test_scfv_artifact_uses_template_cdr_positions_for_final_scoring(self) -> None:
        module_holder: dict[str, object] = {}

        class FakeScfvDesignApp(FakeDesignApp):
            scoring_cdr_indices = None

            def design(self, **kwargs):
                assert self.loaded
                FakeDesignApp.last_design_kwargs = kwargs
                fake_module = module_holder["module"]
                scoring_cdr_indices = fake_module._cdr_indices("EVAAQB")
                FakeScfvDesignApp.scoring_cdr_indices = scoring_cdr_indices
                return (
                    ["TARGET|EVAAQB"],
                    {},
                    [
                        {
                            "critic_name": DEFAULT_CRITIC_NAME,
                            "batch_idx": 0,
                            "designed_sequence": "TARGET|EVAAQB",
                            "complex": FakeComplex(),
                            "final_loss": 1.23,
                            "iptm": 0.91,
                            "distogram_iptm_proxy": 0.82,
                            "cdr_distogram_iptm_proxy": 0.77,
                            "scoring_cdr_indices": scoring_cdr_indices,
                        }
                    ],
                )

        def original_cdr_indices(_sequence: str) -> list[int]:
            raise AssertionError("tutorial _cdr_indices should not be called")

        fake_module = types.SimpleNamespace(
            STEPS=150,
            LOG_INTERVAL=5,
            ESMFold2Design=FakeScfvDesignApp,
            fold_and_get_distogram=lambda *args, **kwargs: None,
            compute_structure_losses=lambda *args, **kwargs: {},
            _cdr_indices=original_cdr_indices,
        )
        module_holder["module"] = fake_module

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "tutorial"},
        ), patch(
            "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
            return_value=fake_module,
        ):
            root = Path(tmpdir)
            result = run_binder_design_artifact(
                campaign_dir=root,
                candidate_id="cand_000000_0000",
                shard_id="shard_000000",
                seed=0,
                esm_repo=None,
                gpu_id=None,
                steps=1,
                target_name="custom_sequence_target",
                target_sequence="TARGET",
                binder_name="test_scfv",
                binder_scaffold="scfv",
                binder_framework_name="test_scfv",
                binder_framework_source="template",
                binder_framework_template="EV{hcdr1}Q{hcdr2}",
                binder_framework_cdr_lengths={
                    "hcdr1": (2, 2),
                    "hcdr2": (1, 1),
                    "hcdr3": (1, 1),
                    "lcdr1": (1, 1),
                    "lcdr2": (1, 1),
                    "lcdr3": (1, 1),
                },
                binder_framework_cdr_indices=None,
                critic_name=DEFAULT_CRITIC_NAME,
        )

        self.assertEqual(result.designed_sequence, "EVAAQB")
        self.assertEqual(FakeScfvDesignApp.scoring_cdr_indices, [2, 3, 5])
        self.assertEqual(result.design_metrics["cdr_indices"], [2, 3, 5])
        self.assertEqual(result.design_metrics["cdr_lengths"], {"hcdr1": 2, "hcdr2": 1})
        self.assertEqual(result.design_metrics["cdr_sequences"], {"hcdr1": "AA", "hcdr2": "B"})
        self.assertIs(fake_module._cdr_indices, original_cdr_indices)

    def test_multichain_structure_artifact_preserves_target_chain_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    conditioning_mode="none",
                )
            )

        pdb_text = "".join(
            [
                _pdb_atom_line(1, "CA", "GLY", "A", 1, "", 0.0, 0.0, 0.0),
                _pdb_atom_line(2, "CA", "GLY", "B", 1, "", 1.0, 0.0, 0.0),
                _pdb_atom_line(3, "CA", "GLY", "C", 1, "", 2.0, 0.0, 0.0),
            ]
        )

        rewritten = adapter_module._rewrite_pdb_chain_ids_for_structure_target(
            pdb_text,
            structure_target,
        )
        chain_ids = [
            line[21]
            for line in rewritten.splitlines()
            if line.startswith("ATOM")
        ]

        self.assertEqual(chain_ids, ["A", "C", "B"])

    def test_binder_chain_id_uses_first_free_single_character_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_chain_test_pdb(target_path, ("A", "B", "C"))
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "B", "C"),
                    conditioning_mode="none",
                )
            )
            fake_module = types.SimpleNamespace(
                STEPS=150,
                LOG_INTERVAL=5,
                ESMFold2Design=FakeDesignApp,
                fold_and_get_distogram=lambda *args, **kwargs: None,
                compute_structure_losses=lambda *args, **kwargs: {},
            )

            with patch.dict(
                "os.environ",
                {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "tutorial"},
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
                return_value=fake_module,
            ):
                result = run_binder_design_artifact(
                    campaign_dir=root,
                    candidate_id="cand_000000_0000",
                    shard_id="shard_000000",
                    seed=0,
                    esm_repo=None,
                    gpu_id=None,
                    steps=1,
                    target_name=None,
                    binder_name="minibinder",
                    critic_name=DEFAULT_CRITIC_NAME,
                    structure_target=structure_target,
                    conditioning_mode="none",
                )

        self.assertEqual(result.design_metrics["target_chain_ids"], ["A", "B", "C"])
        self.assertEqual(result.design_metrics["binder_chain_id"], "D")

    def test_binder_chain_id_can_be_a_when_target_uses_other_chain_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_path = root / "target.pdb"
            _write_chain_test_pdb(target_path, ("X", "Y"))
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("X", "Y"),
                    conditioning_mode="none",
                )
            )
            fake_module = types.SimpleNamespace(
                STEPS=150,
                LOG_INTERVAL=5,
                ESMFold2Design=FakeDesignApp,
                fold_and_get_distogram=lambda *args, **kwargs: None,
                compute_structure_losses=lambda *args, **kwargs: {},
            )

            with patch.dict(
                "os.environ",
                {"ESMFOLD2_PIPELINE_DESIGN_BACKEND": "tutorial"},
            ), patch(
                "esmfold2_pipeline.esm_adapter.binder_design.load_binder_design_module",
                return_value=fake_module,
            ):
                result = run_binder_design_artifact(
                    campaign_dir=root,
                    candidate_id="cand_000000_0000",
                    shard_id="shard_000000",
                    seed=0,
                    esm_repo=None,
                    gpu_id=None,
                    steps=1,
                    target_name=None,
                    binder_name="minibinder",
                    critic_name=DEFAULT_CRITIC_NAME,
                    structure_target=structure_target,
                    conditioning_mode="none",
                )

        self.assertEqual(result.design_metrics["target_chain_ids"], ["X", "Y"])
        self.assertEqual(result.design_metrics["binder_chain_id"], "A")

        pdb_text = "".join(
            [
                _pdb_atom_line(1, "CA", "GLY", "A", 1, "", 0.0, 0.0, 0.0),
                _pdb_atom_line(2, "CA", "GLY", "B", 1, "", 1.0, 0.0, 0.0),
                _pdb_atom_line(3, "CA", "GLY", "C", 1, "", 2.0, 0.0, 0.0),
            ]
        )
        rewritten = adapter_module._rewrite_pdb_chain_ids_for_structure_target(
            pdb_text,
            structure_target,
        )
        chain_ids = [
            line[21]
            for line in rewritten.splitlines()
            if line.startswith("ATOM")
        ]
        self.assertEqual(chain_ids, ["X", "Y", "A"])

    def test_multichain_geometry_metrics_include_chain_and_pair_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    conditioning_mode="none",
                )
            )

        fold_result = {"inputs": {}, "output": {}}
        base_metrics = types.SimpleNamespace(
            target_distance_rmse=2.5,
            target_aligned_rmsd=1.2,
            target_residue_count=4,
        )
        diagnostics = {
            "target_chain_geometry": {
                "A": {"distance_rmse": 0.1, "aligned_rmsd": 0.2, "residue_count": 2},
                "C": {"distance_rmse": 0.3, "aligned_rmsd": 0.4, "residue_count": 2},
            },
            "target_assembly_geometry": {
                "A__C": {
                    "pair_distance_rmse": 1.5,
                    "contact_recovery_8A": 0.5,
                    "contact_recovery_12A": 1.0,
                    "residue_pair_count": 4,
                }
            },
        }

        with patch.object(
            adapter_module,
            "compute_fold_target_geometry_metrics",
            return_value=base_metrics,
        ), patch.object(
            adapter_module,
            "compute_fold_target_geometry_diagnostics",
            return_value=diagnostics,
        ):
            metrics = adapter_module._target_geometry_metrics_from_capture(
                structure_target,
                fold_result,
            )

        self.assertEqual(metrics["target_distance_rmse"], 2.5)
        self.assertEqual(metrics["target_chain_geometry"]["A"]["residue_count"], 2)
        self.assertEqual(
            metrics["target_assembly_geometry"]["A__C"]["pair_distance_rmse"],
            1.5,
        )

    def test_binder_target_iptm_excludes_target_target_chain_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    conditioning_mode="none",
                )
            )

        metrics = adapter_module._binder_target_iptm_metrics_from_capture(
            structure_target,
            {
                "pair_chains_iptm": np.array(
                    [
                        [
                            [1.00, 0.99, 0.10],
                            [0.99, 1.00, 0.80],
                            [0.30, 0.60, 1.00],
                        ]
                    ],
                    dtype=np.float32,
                ),
            },
            complex_iptm=0.95,
        )

        self.assertEqual(metrics["iptm_scope"], "binder_target")
        self.assertEqual(metrics["complex_iptm"], 0.95)
        self.assertAlmostEqual(metrics["binder_target_iptm_by_chain"]["A"], 0.2)
        self.assertAlmostEqual(metrics["binder_target_iptm_by_chain"]["C"], 0.7)
        self.assertAlmostEqual(metrics["binder_target_iptm"], 0.45)
        self.assertAlmostEqual(metrics["iptm"], 0.45)

    def test_binder_target_iptm_falls_back_to_complex_scope_when_pair_scores_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    conditioning_mode="none",
                )
            )

        metrics = adapter_module._binder_target_iptm_metrics_from_capture(
            structure_target,
            {},
            complex_iptm=0.95,
        )

        self.assertEqual(metrics, {"complex_iptm": 0.95, "iptm_scope": "complex"})

    def test_multichain_hotspot_metrics_include_per_chain_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    hotspots={"A": ("2",), "C": ("1",)},
                    conditioning_mode="none",
                )
            )

        calls = []

        def fake_contact_metrics(
            _inputs,
            _output,
            *,
            target_sequence: str,
            binder_sequence: str,
            hotspot_indices: tuple[int, ...],
            contact_cutoff_angstrom: float,
        ):
            calls.append(
                (
                    target_sequence,
                    binder_sequence,
                    hotspot_indices,
                    contact_cutoff_angstrom,
                )
            )
            distance = 3.0 + len(calls)
            return types.SimpleNamespace(
                hotspot_contact_cutoff_angstrom=contact_cutoff_angstrom,
                hotspot_heavy_atom_contact_fraction=1.0,
                hotspot_min_heavy_atom_distance_mean=distance,
                hotspot_min_heavy_atom_distance_min=distance,
                hotspot_representative_contact_fraction=1.0,
                hotspot_min_representative_distance_mean=distance + 1.0,
                hotspot_min_representative_distance_min=distance + 1.0,
                hotspot_count=len(hotspot_indices),
            )

        fold_result = {"inputs": {}, "output": {}, "seq_list": ["GS|GT|BINDER"]}
        with patch.object(
            adapter_module,
            "compute_fold_hotspot_contact_metrics",
            side_effect=fake_contact_metrics,
        ):
            metrics = adapter_module._hotspot_metrics_from_capture(
                structure_target,
                fold_result,
                contact_cutoff_angstrom=5.0,
            )

        self.assertEqual([call[2] for call in calls], [(1, 2), (1,), (2,)])
        self.assertEqual(metrics["hotspot_count"], 2)
        self.assertEqual(metrics["hotspot_by_chain"]["A"]["hotspot_indices"], [1])
        self.assertEqual(
            metrics["hotspot_by_chain"]["C"]["hotspot_global_indices"],
            [2],
        )

    def test_assembly_distogram_conditioning_sets_target_pair_blocks_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    conditioning_mode="distogram",
                    conditioning_assembly=True,
                )
            )

        features = {
            "asym_id": np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int64),
            "disto_cond": np.zeros((7, 7), dtype=np.int64),
            "disto_cond_mask": np.zeros((7, 7), dtype=bool),
        }
        features["disto_cond_mask"][0:2, 0:2] = True
        features["disto_cond_mask"][2:4, 2:4] = True
        pairs = adapter_module._resolve_assembly_chain_pairs(structure_target, None)

        adapter_module._apply_assembly_distogram_conditioning(
            features,
            structure_target=structure_target,
            assembly_pairs=pairs,
        )
        adapter_module._inspect_structure_target_distogram_tensors(
            features,
            structure_target=structure_target,
            binder_length=3,
            expect_conditioned=True,
            assembly_pairs=pairs,
        )

        self.assertTrue(features["disto_cond_mask"][0:2, 2:4].all())
        self.assertTrue(features["disto_cond_mask"][2:4, 0:2].all())
        self.assertFalse(features["disto_cond_mask"][0:4, 4:7].any())
        self.assertFalse(features["disto_cond_mask"][4:7, :].any())
        self.assertEqual(features["disto_cond"][0:2, 2:4].shape, (2, 2))

    def test_template_distogram_conditioning_matrix_includes_assembly_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    conditioning_mode="distogram",
                    conditioning_assembly=True,
                )
            )

        pairs = adapter_module._resolve_assembly_chain_pairs(structure_target, None)
        distances, mask = adapter_module._target_distogram_conditioning_matrix(
            structure_target,
            binder_length=3,
            assembly_pairs=pairs,
        )

        self.assertEqual(distances.shape, (7, 7))
        self.assertEqual(mask.shape, (7, 7))
        self.assertTrue(mask[0:2, 0:2].all())
        self.assertTrue(mask[2:4, 2:4].all())
        self.assertTrue(mask[0:2, 2:4].all())
        self.assertTrue(mask[2:4, 0:2].all())
        self.assertFalse(mask[0:4, 4:7].any())
        self.assertFalse(mask[4:7, :].any())
        self.assertAlmostEqual(float(distances[0, 2]), 8.0)
        self.assertAlmostEqual(float(distances[0, 1]), float(np.sqrt(10.0)))

    def test_template_distogram_patch_adds_pair_bias_and_restores(self) -> None:
        class FakeTensor:
            def __init__(self, array):
                self.array = np.asarray(array, dtype=np.float32)
                self.shape = self.array.shape
                self.device = "cpu"
                self.dtype = self.array.dtype

            def to(self, **_kwargs):
                return self

            def __add__(self, other):
                return FakeTensor(self.array + other.array)

        def forward(pair, **_kwargs):
            return pair

        folding_trunk = types.SimpleNamespace(forward=forward)
        model = types.SimpleNamespace(folding_trunk=folding_trunk)
        original_forward = folding_trunk.forward
        pair = FakeTensor(np.zeros((1, 2, 2, 1), dtype=np.float32))
        template_pair_bias = FakeTensor(np.ones((1, 2, 2, 1), dtype=np.float32))

        with adapter_module._patched_model_folding_trunk_with_distogram_template(
            model,
            template_pair_bias,
        ):
            output = model.folding_trunk.forward(pair)

        self.assertTrue(np.allclose(output.array, 1.0))
        self.assertIs(folding_trunk.forward, original_forward)

    def test_assembly_distogram_conditioning_rejects_bad_pair_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    conditioning_mode="distogram",
                    conditioning_assembly=True,
                )
            )

        features = {
            "asym_id": np.array([0, 0, 1, 1], dtype=np.int64),
            "disto_cond": np.zeros((4, 4), dtype=np.int64),
            "disto_cond_mask": np.zeros((4, 4), dtype=bool),
        }
        pairs = adapter_module._resolve_assembly_chain_pairs(structure_target, None)
        with patch.object(
            adapter_module,
            "_compute_pair_distogram_array",
            return_value=np.zeros((1, 2), dtype=np.float32),
        ):
            with self.assertRaisesRegex(ValueError, "assembly distogram shape"):
                adapter_module._apply_assembly_distogram_conditioning(
                    features,
                    structure_target=structure_target,
                    assembly_pairs=pairs,
                )

    def test_distogram_conditioning_applies_to_design_and_critic_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    conditioning_mode="distogram",
                    conditioning_assembly=True,
                    conditioning_chain_pairs=(("A", "C"),),
                )
            )

        original_fold = object()
        fake_module = types.SimpleNamespace(fold_and_get_distogram=original_fold)
        capture = adapter_module._FoldCapture()
        calls = []

        def fake_structure_fold(*args, **kwargs):
            calls.append(kwargs)
            return {"call_index": len(calls)}

        with patch.object(
            adapter_module,
            "_fold_and_get_distogram_for_structure_target",
            side_effect=fake_structure_fold,
        ):
            with adapter_module._patched_fold_with_distogram_conditioning(
                fake_module,
                structure_target=structure_target,
                enabled=True,
                condition_assembly=True,
                conditioning_chain_pairs=(("A", "C"),),
                capture=capture,
            ):
                design_fold = fake_module.fold_and_get_distogram(
                    "model",
                    "GSGT",
                    "target_one_hot",
                    "design",
                    num_loops=1,
                    num_sampling_steps=7,
                    calculate_confidence=False,
                    seed=11,
                )
                confidence_fold = fake_module.fold_and_get_distogram(
                    "model",
                    "GSGT",
                    "target_one_hot",
                    "design",
                    num_loops=3,
                    num_sampling_steps=200,
                    calculate_confidence=True,
                    seed=12,
                )

        self.assertIs(fake_module.fold_and_get_distogram, original_fold)
        self.assertEqual(design_fold, {"call_index": 1})
        self.assertEqual(confidence_fold, {"call_index": 2})
        self.assertEqual(capture.last_design_fold, {"call_index": 1})
        self.assertEqual(capture.last_confidence_fold, {"call_index": 2})
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIs(call["structure_target"], structure_target)
            self.assertTrue(call["condition_distograms"])
            self.assertTrue(call["condition_assembly"])
            self.assertEqual(call["conditioning_chain_pairs"], (("A", "C"),))
        self.assertEqual(calls[0]["num_loops"], 1)
        self.assertFalse(calls[0]["calculate_confidence"])
        self.assertEqual(calls[1]["num_loops"], 3)
        self.assertTrue(calls[1]["calculate_confidence"])

    def test_adapter_patches_miniprotein_length_range_and_restores(self) -> None:
        factory = types.SimpleNamespace(length_ranges={"seq": (60, 200)})
        fake_module = types.SimpleNamespace(
            BINDER_PROMPT_FACTORIES={"minibinder": factory},
        )

        with adapter_module._patched_binder_length_range(
            fake_module,
            binder_name="minibinder",
            length_range=(45, 90),
        ):
            self.assertEqual(factory.length_ranges["seq"], (45, 90))

        self.assertEqual(factory.length_ranges["seq"], (60, 200))

    def test_adapter_patches_frozen_miniprotein_factory_and_restores(self) -> None:
        @dataclass(frozen=True)
        class FrozenFactory:
            length_ranges: dict[str, tuple[int, int]]

        factory = FrozenFactory(length_ranges={"seq": (60, 200)})
        fake_module = types.SimpleNamespace(
            BINDER_PROMPT_FACTORIES={"minibinder": factory},
        )

        with adapter_module._patched_binder_length_range(
            fake_module,
            binder_name="minibinder",
            length_range=(45, 90),
        ):
            patched = fake_module.BINDER_PROMPT_FACTORIES["minibinder"]
            self.assertIsNot(patched, factory)
            self.assertEqual(patched.length_ranges["seq"], (45, 90))

        self.assertIs(fake_module.BINDER_PROMPT_FACTORIES["minibinder"], factory)

    def test_hotspot_loss_patch_adds_separate_weighted_loss_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A",),
                    hotspots={"A": ("2",)},
                    conditioning_mode="distogram",
                )
            )

            def base_structure_losses(_distogram_logits, _binder_length: int) -> dict:
                return {"inter_contact_loss": 3.0, "total_loss": 10.0}

            fake_module = types.SimpleNamespace(
                compute_structure_losses=base_structure_losses,
            )

            with patch.object(
                adapter_module,
                "_compute_hotspot_contact_loss",
                return_value=2.0,
            ) as hotspot_loss:
                with adapter_module._patched_structure_losses_for_hotspots(
                    fake_module,
                    structure_target=structure_target,
                    hotspot_contact_weight=0.25,
                    hotspot_distogram_contact_cutoff_angstrom=20.0,
                    hotspot_num_contacts=2,
                    hotspot_contact_probability_target=0.6,
                    hotspot_loss_mode="probability_hinge",
                ):
                    losses = fake_module.compute_structure_losses("logits", 5)

            self.assertEqual(losses["inter_contact_loss"], 3.0)
            self.assertEqual(losses["hotspot_contact_loss"], 2.0)
            self.assertEqual(losses["total_loss"], 10.5)
            hotspot_loss.assert_called_once_with(
                fake_module,
                "logits",
                5,
                hotspot_indices=(1,),
                contact_cutoff_angstrom=20.0,
                hotspot_num_contacts=2,
                contact_probability_target=0.6,
                hotspot_loss_mode="probability_hinge",
                binder_contact_indices=None,
            )
            self.assertIs(fake_module.compute_structure_losses, base_structure_losses)

    def test_target_geometry_drift_patch_adds_hinge_loss_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(path=target_path, chains=("A",))
            )

            def base_structure_losses(_distogram_logits, _binder_length: int) -> dict:
                return {"inter_contact_loss": 3.0, "total_loss": 10.0}

            fake_module = types.SimpleNamespace(
                compute_structure_losses=base_structure_losses,
            )

            with patch.object(
                adapter_module,
                "_compute_target_geometry_drift_loss",
                return_value=(2.0, 0.4),
            ) as drift_loss:
                with adapter_module._patched_structure_losses_for_target_geometry_drift(
                    fake_module,
                    structure_target=structure_target,
                    drift_config=TargetGeometryDriftConfig(
                        enabled=True,
                        weight=0.25,
                        tolerance_angstrom=1.5,
                        stiffness_angstrom=0.1,
                    ),
                    selected_indices=(0, 1),
                ):
                    losses = fake_module.compute_structure_losses("logits", 5)

            self.assertEqual(losses["inter_contact_loss"], 3.0)
            self.assertEqual(losses["target_geometry_drift_loss"], 2.0)
            self.assertEqual(losses["target_geometry_drift_rmse"], 0.4)
            self.assertEqual(losses["total_loss"], 10.5)
            _, args, kwargs = drift_loss.mock_calls[0]
            self.assertEqual(args[:3], (fake_module, "logits", 5))
            self.assertEqual(kwargs["tolerance_angstrom"], 1.5)
            self.assertEqual(kwargs["stiffness_angstrom"], 0.1)
            self.assertEqual(kwargs["reference_distances"].shape, (2, 2))
            self.assertEqual(kwargs["pair_mask"].sum(), 1)
            self.assertIs(fake_module.compute_structure_losses, base_structure_losses)

    def test_target_geometry_drift_hinge_uses_linear_margin_penalty(self) -> None:
        fake_torch = types.SimpleNamespace(
            relu=lambda value: np.maximum(value, np.float32(0.0)),
        )

        loss = adapter_module._compute_target_geometry_drift_hinge_loss(
            fake_torch,
            np.array([0.9], dtype=np.float32),
            tolerance_angstrom=0.5,
            stiffness_angstrom=0.1,
        )
        below_tolerance_loss = adapter_module._compute_target_geometry_drift_hinge_loss(
            fake_torch,
            np.array([0.25], dtype=np.float32),
            tolerance_angstrom=0.5,
            stiffness_angstrom=0.1,
        )

        self.assertAlmostEqual(float(loss[0]), 4.0, places=6)
        self.assertAlmostEqual(float(below_tolerance_loss[0]), 0.0, places=6)

    def test_target_geometry_drift_patch_disabled_is_noop(self) -> None:
        def base_structure_losses(_distogram_logits, _binder_length: int) -> dict:
            return {"total_loss": 10.0}

        fake_module = types.SimpleNamespace(
            compute_structure_losses=base_structure_losses,
        )

        with adapter_module._patched_structure_losses_for_target_geometry_drift(
            fake_module,
            structure_target=None,
            drift_config=TargetGeometryDriftConfig(enabled=False),
            selected_indices=(),
        ):
            self.assertIs(fake_module.compute_structure_losses, base_structure_losses)

        self.assertIs(fake_module.compute_structure_losses, base_structure_losses)

    def test_hotspot_loss_patch_uses_multichain_global_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_multichain_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A", "C"),
                    hotspots={"A": ("2",), "C": ("1",)},
                    conditioning_mode="distogram",
                )
            )

            def base_structure_losses(_distogram_logits, _binder_length: int) -> dict:
                return {"inter_contact_loss": 3.0, "total_loss": 10.0}

            fake_module = types.SimpleNamespace(
                compute_structure_losses=base_structure_losses,
            )

            with patch.object(
                adapter_module,
                "_compute_hotspot_contact_loss",
                return_value=2.0,
            ) as hotspot_loss:
                with adapter_module._patched_structure_losses_for_hotspots(
                    fake_module,
                    structure_target=structure_target,
                    hotspot_contact_weight=0.25,
                    hotspot_distogram_contact_cutoff_angstrom=20.0,
                    hotspot_num_contacts=2,
                    hotspot_contact_probability_target=0.6,
                    hotspot_loss_mode="probability_hinge",
                ):
                    losses = fake_module.compute_structure_losses("logits", 5)

        self.assertEqual(losses["total_loss"], 10.5)
        hotspot_loss.assert_called_once_with(
            fake_module,
            "logits",
            5,
            hotspot_indices=(1, 2),
            contact_cutoff_angstrom=20.0,
            hotspot_num_contacts=2,
            contact_probability_target=0.6,
            hotspot_loss_mode="probability_hinge",
            binder_contact_indices=None,
        )

    def test_entropy_hotspot_loss_adds_to_existing_structure_loss_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A",),
                    hotspots={"A": ("2",)},
                    conditioning_mode="distogram",
                )
            )

            def base_structure_losses(_distogram_logits, _binder_length: int) -> dict:
                return {"inter_contact_loss": 3.0, "total_loss": 10.0}

            fake_module = types.SimpleNamespace(
                compute_structure_losses=base_structure_losses,
            )

            with patch.object(
                adapter_module,
                "_compute_hotspot_entropy_contact_loss",
                return_value=2.0,
            ) as hotspot_loss:
                with adapter_module._patched_structure_losses_for_hotspots(
                    fake_module,
                    structure_target=structure_target,
                    hotspot_contact_weight=1.0,
                    hotspot_distogram_contact_cutoff_angstrom=20.0,
                    hotspot_num_contacts=2,
                    hotspot_contact_probability_target=0.6,
                    hotspot_loss_mode="entropy_hotspot",
                ):
                    losses = fake_module.compute_structure_losses("logits", 5)

            self.assertEqual(losses["inter_contact_loss"], 3.0)
            self.assertEqual(losses["hotspot_contact_loss"], 2.0)
            self.assertEqual(losses["total_loss"], 12.0)
            hotspot_loss.assert_called_once_with(
                fake_module,
                "logits",
                5,
                hotspot_indices=(1,),
                contact_cutoff_angstrom=20.0,
                hotspot_num_contacts=2,
                binder_contact_indices=None,
            )
            self.assertIs(fake_module.compute_structure_losses, base_structure_losses)

    def test_scfv_prompt_plan_samples_builtin_framework_and_records_cdrs(self) -> None:
        class FakePromptFactory:
            def sample(self, *, seed: int) -> str:
                self.seed = seed
                return "EVQ##SSGG##"

        fake_module = types.SimpleNamespace(
            BINDER_PROMPT_FACTORIES={
                "trastuzumab_framework_vhvl": FakePromptFactory(),
            }
        )

        plan = adapter_module._prepare_binder_prompt_plan(
            fake_module,
            binder_name="trastuzumab_framework_vhvl",
            binder_scaffold="scfv",
            binder_framework_name="trastuzumab_framework_vhvl",
            binder_framework_source="builtin",
            binder_framework_template=None,
            binder_framework_cdr_lengths=None,
            binder_framework_sequence=None,
            binder_framework_cdr_indices=None,
            seed=17,
            is_antibody=None,
        )

        self.assertIsNone(plan.binder_name)
        self.assertEqual(plan.binder_sequence, "EVQ##SSGG##")
        self.assertTrue(plan.is_antibody)
        self.assertEqual(plan.cdr_indices, (3, 4, 9, 10))
        self.assertEqual(plan.cdr_lengths, {"hcdr1": 2, "hcdr2": 2})
        self.assertEqual(
            adapter_module._mutable_run_sequences("EVQABSSGGCD", plan.cdr_indices),
            {"hcdr1": "AB", "hcdr2": "CD"},
        )

    def test_scfv_builtin_prompt_plan_uses_local_template_when_present(self) -> None:
        cdr_lengths = {
            "hcdr1": (2, 2),
            "hcdr2": (1, 1),
            "hcdr3": (3, 3),
            "lcdr1": (2, 2),
            "lcdr2": (1, 1),
            "lcdr3": (2, 2),
        }

        plan = adapter_module._prepare_binder_prompt_plan(
            types.SimpleNamespace(),
            binder_name="trastuzumab_framework_vhvl",
            binder_scaffold="scfv",
            binder_framework_name="trastuzumab_framework_vhvl",
            binder_framework_source="builtin",
            binder_framework_template=(
                "EV{hcdr1}Q{hcdr2}SS{hcdr3}GG{lcdr1}S{lcdr2}T{lcdr3}"
            ),
            binder_framework_cdr_lengths=cdr_lengths,
            binder_framework_sequence=None,
            binder_framework_cdr_indices=None,
            seed=0,
            is_antibody=None,
        )

        self.assertIsNone(plan.binder_name)
        self.assertEqual(plan.binder_sequence, "EV##Q#SS###GG##S#T##")
        self.assertTrue(plan.is_antibody)
        self.assertEqual(
            plan.cdr_lengths,
            {
                "hcdr1": 2,
                "hcdr2": 1,
                "hcdr3": 3,
                "lcdr1": 2,
                "lcdr2": 1,
                "lcdr3": 2,
            },
        )

    def test_vhh_prompt_plan_samples_builtin_framework_and_reports_heavy_cdrs(self) -> None:
        cdr_lengths = {
            "cdr1": (2, 2),
            "cdr2": (1, 1),
            "cdr3": (3, 3),
        }

        plan = adapter_module._prepare_binder_prompt_plan(
            types.SimpleNamespace(),
            binder_name="caplacizumab_framework_vhh",
            binder_scaffold="vhh",
            binder_framework_name="caplacizumab_framework_vhh",
            binder_framework_source="builtin",
            binder_framework_template="EV{cdr1}Q{cdr2}SS{cdr3}",
            binder_framework_cdr_lengths=cdr_lengths,
            binder_framework_sequence=None,
            binder_framework_cdr_indices=None,
            seed=0,
            is_antibody=None,
        )

        self.assertIsNone(plan.binder_name)
        self.assertEqual(plan.binder_sequence, "EV##Q#SS###")
        self.assertTrue(plan.is_antibody)
        self.assertEqual(plan.cdr_indices, (2, 3, 5, 8, 9, 10))
        self.assertEqual(
            plan.cdr_lengths,
            {
                "hcdr1": 2,
                "hcdr2": 1,
                "hcdr3": 3,
            },
        )
        self.assertEqual(
            adapter_module._mutable_run_sequences(
                "EVAAQBSSCCC",
                plan.cdr_indices,
                cdr_names=plan.cdr_report_names,
            ),
            {"hcdr1": "AA", "hcdr2": "B", "hcdr3": "CCC"},
        )

    def test_scfv_template_prompt_plan_samples_cdr_lengths_deterministically(self) -> None:
        cdr_lengths = {
            "hcdr1": (2, 2),
            "hcdr2": (1, 1),
            "hcdr3": (3, 3),
            "lcdr1": (2, 2),
            "lcdr2": (1, 1),
            "lcdr3": (2, 2),
        }

        plan = adapter_module._prepare_binder_prompt_plan(
            types.SimpleNamespace(),
            binder_name="custom_framework",
            binder_scaffold="scfv",
            binder_framework_name="custom_framework",
            binder_framework_source="template",
            binder_framework_template=(
                "EV{hcdr1}Q{hcdr2}SS{hcdr3}GG{lcdr1}S{lcdr2}T{lcdr3}"
            ),
            binder_framework_cdr_lengths=cdr_lengths,
            binder_framework_sequence=None,
            binder_framework_cdr_indices=None,
            seed=0,
            is_antibody=None,
        )

        self.assertIsNone(plan.binder_name)
        self.assertEqual(plan.binder_sequence, "EV##Q#SS###GG##S#T##")
        self.assertTrue(plan.is_antibody)
        self.assertEqual(plan.cdr_lengths["hcdr3"], 3)

    def test_scfv_sequence_prompt_uses_explicit_cdr_indices(self) -> None:
        prompt = adapter_module._scfv_cdr_prompt_from_indices(
            "EVAAQB",
            (2, 3, 5),
        )

        self.assertEqual(prompt, "EV##Q#")

    def test_hotspot_loss_patch_passes_cdr_contact_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "target.pdb"
            _write_test_pdb(target_path)
            structure_target = parse_structure_target(
                StructureTargetConfig(
                    path=target_path,
                    chains=("A",),
                    hotspots={"A": ("2",)},
                    conditioning_mode="distogram",
                )
            )

            def base_structure_losses(_distogram_logits, _binder_length: int) -> dict:
                return {"total_loss": 10.0}

            fake_module = types.SimpleNamespace(
                compute_structure_losses=base_structure_losses,
            )

            with patch.object(
                adapter_module,
                "_compute_hotspot_contact_loss",
                return_value=2.0,
            ) as hotspot_loss:
                with adapter_module._patched_structure_losses_for_hotspots(
                    fake_module,
                    structure_target=structure_target,
                    hotspot_contact_weight=1.0,
                    hotspot_distogram_contact_cutoff_angstrom=20.0,
                    hotspot_num_contacts=1,
                    hotspot_contact_probability_target=0.6,
                    hotspot_loss_mode="probability_hinge",
                    binder_contact_indices=(3, 4),
                ):
                    losses = fake_module.compute_structure_losses("logits", 8)

        self.assertEqual(losses["total_loss"], 12.0)
        hotspot_loss.assert_called_once_with(
            fake_module,
            "logits",
            8,
            hotspot_indices=(1,),
            contact_cutoff_angstrom=20.0,
            hotspot_num_contacts=1,
            contact_probability_target=0.6,
            hotspot_loss_mode="probability_hinge",
            binder_contact_indices=(3, 4),
        )

    def test_plddt_metrics_split_complex_target_and_binder(self) -> None:
        fold_result = {
            "seq_list": ["AA|BBB|CCCC"],
            "plddt": np.array([[0.80, 0.90, 0.70, 0.60, 0.55, 0.50, 0.45]]),
        }

        metrics = adapter_module._plddt_metrics_from_capture(fold_result)

        self.assertAlmostEqual(metrics["plddt_complex"], 64.28571428571429)
        self.assertAlmostEqual(metrics["plddt_target"], 71.0)
        self.assertAlmostEqual(metrics["plddt_binder"], 47.5)
        self.assertEqual(metrics["plddt"], metrics["plddt_complex"])

    def test_confidence_scalar_metrics_capture_ptm(self) -> None:
        metrics = adapter_module._confidence_scalar_metrics_from_capture(
            {"output": {"ptm": np.array(0.73)}}
        )

        self.assertEqual(metrics["ptm"], 0.73)

    def test_rewrite_pdb_b_factors_uses_residue_plddt(self) -> None:
        pdb_text = "".join(
            [
                _pdb_atom_line(1, "N", "GLY", "A", 1, "", 0.0, 0.0, 0.0),
                _pdb_atom_line(2, "CA", "GLY", "A", 1, "", 1.0, 0.0, 0.0),
                _pdb_atom_line(3, "N", "SER", "B", 1, "", 0.0, 4.0, 0.0),
            ]
        )

        rewritten = adapter_module._rewrite_pdb_b_factors(
            pdb_text,
            np.array([91.25, 42.5]),
        )

        atom_lines = [line for line in rewritten.splitlines() if line.startswith("ATOM")]
        self.assertEqual(float(atom_lines[0][60:66]), 91.25)
        self.assertEqual(float(atom_lines[1][60:66]), 91.25)
        self.assertEqual(float(atom_lines[2][60:66]), 42.5)


if __name__ == "__main__":
    unittest.main()


def _write_test_pdb(path: Path) -> None:
    lines = [
        _pdb_atom_line(1, "N", "GLY", "A", 1, "", 0.0, 0.0, 0.0),
        _pdb_atom_line(2, "CA", "GLY", "A", 1, "", 1.0, 0.0, 0.0),
        _pdb_atom_line(3, "C", "GLY", "A", 1, "", 2.0, 0.0, 0.0),
        _pdb_atom_line(4, "O", "GLY", "A", 1, "", 2.5, 0.5, 0.0),
        _pdb_atom_line(5, "N", "SER", "A", 2, "", 3.0, 0.0, 0.0),
        _pdb_atom_line(6, "CA", "SER", "A", 2, "", 4.0, 0.0, 0.0),
        _pdb_atom_line(7, "CB", "SER", "A", 2, "", 4.0, 1.0, 0.0),
        _pdb_atom_line(8, "C", "SER", "A", 2, "", 5.0, 0.0, 0.0),
        _pdb_atom_line(9, "O", "SER", "A", 2, "", 5.5, 0.5, 0.0),
    ]
    path.write_text("".join(lines))


def _write_multichain_test_pdb(path: Path) -> None:
    lines = [
        _pdb_atom_line(1, "N", "GLY", "A", 1, "", 0.0, 0.0, 0.0),
        _pdb_atom_line(2, "CA", "GLY", "A", 1, "", 1.0, 0.0, 0.0),
        _pdb_atom_line(3, "C", "GLY", "A", 1, "", 2.0, 0.0, 0.0),
        _pdb_atom_line(4, "O", "GLY", "A", 1, "", 2.5, 0.5, 0.0),
        _pdb_atom_line(5, "N", "SER", "A", 2, "", 3.0, 0.0, 0.0),
        _pdb_atom_line(6, "CA", "SER", "A", 2, "", 4.0, 0.0, 0.0),
        _pdb_atom_line(7, "CB", "SER", "A", 2, "", 4.0, 1.0, 0.0),
        _pdb_atom_line(8, "C", "SER", "A", 2, "", 5.0, 0.0, 0.0),
        _pdb_atom_line(9, "O", "SER", "A", 2, "", 5.5, 0.5, 0.0),
        _pdb_atom_line(10, "N", "GLY", "C", 1, "", 0.0, 8.0, 0.0),
        _pdb_atom_line(11, "CA", "GLY", "C", 1, "", 1.0, 8.0, 0.0),
        _pdb_atom_line(12, "C", "GLY", "C", 1, "", 2.0, 8.0, 0.0),
        _pdb_atom_line(13, "O", "GLY", "C", 1, "", 2.5, 8.5, 0.0),
        _pdb_atom_line(14, "N", "THR", "C", 2, "", 3.0, 8.0, 0.0),
        _pdb_atom_line(15, "CA", "THR", "C", 2, "", 4.0, 8.0, 0.0),
        _pdb_atom_line(16, "CB", "THR", "C", 2, "", 4.0, 9.0, 0.0),
        _pdb_atom_line(17, "C", "THR", "C", 2, "", 5.0, 8.0, 0.0),
        _pdb_atom_line(18, "O", "THR", "C", 2, "", 5.5, 8.5, 0.0),
    ]
    path.write_text("".join(lines))


def _write_chain_test_pdb(path: Path, chain_ids: tuple[str, ...]) -> None:
    lines: list[str] = []
    serial = 1
    for chain_index, chain_id in enumerate(chain_ids):
        y = float(chain_index * 8)
        lines.extend(
            [
                _pdb_atom_line(serial, "N", "GLY", chain_id, 1, "", 0.0, y, 0.0),
                _pdb_atom_line(serial + 1, "CA", "GLY", chain_id, 1, "", 1.0, y, 0.0),
                _pdb_atom_line(serial + 2, "C", "GLY", chain_id, 1, "", 2.0, y, 0.0),
                _pdb_atom_line(
                    serial + 3,
                    "O",
                    "GLY",
                    chain_id,
                    1,
                    "",
                    2.5,
                    y + 0.5,
                    0.0,
                ),
            ]
        )
        serial += 4
    path.write_text("".join(lines))


def _pdb_atom_line(
    serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_id: int,
    ins_code: str,
    x: float,
    y: float,
    z: float,
) -> str:
    element = atom_name.strip()[0]
    return (
        f"ATOM  {serial:5d} {atom_name:^4} {res_name:>3} {chain_id}"
        f"{res_id:4d}{ins_code:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00 20.00          {element:>2s}\n"
    )
