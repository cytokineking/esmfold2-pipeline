from __future__ import annotations

from contextlib import nullcontext
import types
import unittest

import numpy as np

from esmfold2_pipeline.design.loop import (
    AA_DIMS,
    CYS_IDX,
    LEARNING_RATE,
    StepResult,
    build_gradient_mask,
    build_initial_soft_sequence_logits,
    calculate_confidence_for_temperature,
    design_fold_sampling_steps,
    normalized_gradient_tensor,
    plm_gradient_weight,
    run_design_loop,
    run_gradient_design_loop,
    sequence_to_one_hot_indices,
    select_inversion_model,
    temperature_for_step,
)


class _FakeTensor(np.ndarray):
    __array_priority__ = 1000

    def __array_finalize__(self, _obj):
        pass

    def requires_grad_(self, _requires_grad: bool = True):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def unsqueeze(self, axis: int):
        return _fake_tensor(np.expand_dims(self, axis=axis))

    def float(self):
        return _fake_tensor(self.astype(np.float32))

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, *, dim=None):
        if dim is not None:
            axis = dim
        return _fake_tensor(
            np.asarray(self).sum(
                axis=axis,
                dtype=dtype,
                out=out,
                keepdims=keepdims,
            )
        )

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, dim=None):
        if dim is not None:
            axis = dim
        return _fake_tensor(
            np.asarray(self).mean(
                axis=axis,
                dtype=dtype,
                out=out,
                keepdims=keepdims,
            )
        )


def _fake_tensor(values):
    return np.asarray(values, dtype=np.float32).view(_FakeTensor)


class _FakeTorch:
    square = staticmethod(np.square)
    sqrt = staticmethod(np.sqrt)
    linalg = np.linalg
    autograd = types.SimpleNamespace(
        grad=lambda _loss, logits: [_fake_tensor(np.ones_like(logits))]
    )

    @staticmethod
    def zeros(shape):
        return _fake_tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def ones(shape):
        return _fake_tensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def randn(*shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return _fake_tensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def tensor(values):
        return _fake_tensor(values)

    @staticmethod
    def device(_name):
        return nullcontext()


class _FakeFunctional:
    @staticmethod
    def one_hot(indices, *, num_classes: int):
        values = np.asarray(indices, dtype=int)
        result = np.zeros(values.shape + (num_classes,), dtype=np.float32)
        for index in np.ndindex(values.shape):
            result[index + (values[index],)] = 1.0
        return _fake_tensor(result)

    @staticmethod
    def softmax(values, *, dim: int):
        shifted = values - np.max(values, axis=dim, keepdims=True)
        exp = np.exp(shifted)
        return _fake_tensor(exp / exp.sum(axis=dim, keepdims=True))


class _FakeSGD:
    instances: list["_FakeSGD"] = []

    def __init__(self, _params, *, lr: float):
        self.param_groups = [{"lr": lr}]
        self.step_lrs: list[float] = []
        _FakeSGD.instances.append(self)

    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        self.step_lrs.append(float(self.param_groups[0]["lr"]))


class _FakeOptim:
    SGD = _FakeSGD


class DesignLoopTest(unittest.TestCase):
    def test_initial_logits_match_tutorial_fixed_mutable_and_cys_policy(self) -> None:
        logits = build_initial_soft_sequence_logits(
            "A#C",
            batch_size=1,
            torch_module=_FakeTorch,
        )

        self.assertEqual(logits.shape, (1, 3, AA_DIMS))
        self.assertEqual(float(logits[0, 0, 0]), 10.0)
        self.assertEqual(float(logits[0, 1, CYS_IDX]), -1e6)
        self.assertEqual(float(logits[0, 2, CYS_IDX]), 10.0)

    def test_gradient_mask_matches_tutorial_mutable_non_cys_policy(self) -> None:
        mask = build_gradient_mask("A#C", batch_size=1, torch_module=_FakeTorch)

        self.assertTrue(np.all(mask[0, 0, :] == 0.0))
        self.assertTrue(np.all(mask[0, 2, :] == 0.0))
        self.assertEqual(float(mask[0, 1, CYS_IDX]), 0.0)
        self.assertEqual(float(mask[0, 1, 0]), 1.0)

    def test_sequence_to_one_hot_indices_use_esmfold2_token_order(self) -> None:
        self.assertEqual(sequence_to_one_hot_indices("ACV"), [2, 6, 21])

    def test_loop_policy_helpers_match_tutorial_constants(self) -> None:
        self.assertEqual(plm_gradient_weight(False), 0.15)
        self.assertEqual(plm_gradient_weight(None), 0.15)
        self.assertEqual(plm_gradient_weight(True), 0.05)
        self.assertEqual(design_fold_sampling_steps(calculate_confidence=False), 1)
        self.assertEqual(design_fold_sampling_steps(calculate_confidence=True), 50)
        self.assertIs(
            select_inversion_model({"a": "A", "b": "B"}, seed=1, step=0),
            "A",
        )

    def test_normalized_gradient_tensor_matches_tutorial_formula(self) -> None:
        grad = _fake_tensor([[[1.0, 1.0], [3.0, 4.0]]])
        mask = _fake_tensor([[[1.0, 1.0], [0.0, 0.0]]])

        result = normalized_gradient_tensor(
            grad,
            mask,
            torch_module=_FakeTorch,
        )

        expected = np.array([[[1 / np.sqrt(2), 1 / np.sqrt(2)], [0.0, 0.0]]])
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_gradient_design_loop_owns_tutorial_step_and_critic_settings(self) -> None:
        _FakeSGD.instances.clear()
        fold_calls: list[dict[str, object]] = []
        plm_calls: list[dict[str, object]] = []
        seed_context_calls: list[int] = []

        class FakeComplex:
            pass

        def seed_context(seed: int):
            seed_context_calls.append(seed)
            return nullcontext()

        def fold_complex(
            model,
            target_sequence,
            _target_one_hot,
            _design,
            *,
            num_loops,
            num_sampling_steps,
            calculate_confidence,
            seed,
        ):
            fold_calls.append(
                {
                    "model": model,
                    "target_sequence": target_sequence,
                    "num_loops": num_loops,
                    "num_sampling_steps": num_sampling_steps,
                    "calculate_confidence": calculate_confidence,
                    "seed": seed,
                }
            )
            if num_loops == 1:
                design_step = sum(
                    1 for call in fold_calls if call["num_loops"] == 1
                )
                binder_seq = "AA" if design_step == 1 else "VV"
                return {
                    "seq_list": [f"{target_sequence}|{binder_seq}"],
                    "distogram_logits": _fake_tensor([0.0]),
                    "iptm": [0.7] if calculate_confidence else None,
                    "inputs": {},
                    "output": {},
                }
            return {
                "seq_list": [f"{target_sequence}|VV"],
                "distogram_logits": _fake_tensor([0.0]),
                "iptm": _fake_tensor(0.9),
                "inputs": {"fold": "inputs"},
                "output": {"fold": "output"},
            }

        def compute_structure_losses(_distogram_logits, binder_length):
            self.assertEqual(binder_length, 2)
            return {"total_loss": _fake_tensor([1.0])}

        def compute_plm_loss(**kwargs):
            plm_calls.append(kwargs)
            self.assertEqual(kwargs["batch_size"], 4)
            self.assertEqual(kwargs["n_passes"], 4)
            return _fake_tensor([0.2])

        result = run_gradient_design_loop(
            target_sequence="TAR",
            binder_sequence="##",
            is_antibody=False,
            seed=7,
            steps=2,
            batch_size=1,
            inversion_models={"first": "inv0", "second": "inv1"},
            critic_models={"critic": "critic_model"},
            esmc_model="esmc",
            fold_complex=fold_complex,
            compute_structure_losses=compute_structure_losses,
            compute_plm_loss=compute_plm_loss,
            build_complex=lambda inputs, output: FakeComplex(),
            compute_distogram_iptm_proxy=lambda *_args: {
                "distogram_iptm_proxy": 0.5
            },
            torch_module=_FakeTorch,
            functional=_FakeFunctional,
            optim_module=_FakeOptim,
            seed_context=seed_context,
            device="fake",
        )

        self.assertEqual(result.best_sequences, ["TAR|VV"])
        self.assertIsInstance(result.critic_results[0]["complex"], FakeComplex)
        self.assertAlmostEqual(result.critic_results[0]["final_loss"], 1.2)
        self.assertAlmostEqual(result.critic_results[0]["iptm"], 0.9)
        self.assertEqual(result.critic_results[0]["distogram_iptm_proxy"], 0.5)
        self.assertEqual([call["num_loops"] for call in fold_calls], [1, 1, 3])
        self.assertEqual([call["num_sampling_steps"] for call in fold_calls], [1, 50, 200])
        self.assertEqual(
            [call["calculate_confidence"] for call in fold_calls],
            [False, True, True],
        )
        self.assertEqual([call["seed"] for call in fold_calls], [7, 8, 7])
        self.assertEqual(len(plm_calls), 2)
        self.assertEqual(seed_context_calls, [7, 7, 8])
        expected_lrs = [
            LEARNING_RATE * temperature_for_step(0, 2),
            LEARNING_RATE * temperature_for_step(1, 2),
        ]
        np.testing.assert_allclose(_FakeSGD.instances[0].step_lrs, expected_lrs)

    def test_temperature_schedule_matches_tutorial_cosine_schedule(self) -> None:
        self.assertAlmostEqual(temperature_for_step(0, 4), 0.855017856687341)
        self.assertAlmostEqual(temperature_for_step(3, 4), 0.01)
        self.assertFalse(calculate_confidence_for_temperature(0.05))
        self.assertTrue(calculate_confidence_for_temperature(0.049999))

    def test_run_design_loop_selects_best_sequence_by_iptm(self) -> None:
        calls: list[tuple[int, float, bool]] = []

        def run_step(step: int, temperature: float, calculate_confidence: bool):
            calls.append((step, temperature, calculate_confidence))
            if step == 0:
                return StepResult(
                    sequences=["TARGET|AAA"],
                    iptm=None,
                    losses={"total_loss": [10.0]},
                )
            if step == 1:
                return StepResult(
                    sequences=["TARGET|BBB"],
                    iptm=[0.4],
                    losses={"total_loss": [8.0]},
                )
            return StepResult(
                sequences=["TARGET|CCC"],
                iptm=[0.3],
                losses={"total_loss": [7.0]},
            )

        scored: list[tuple[int, str]] = []

        def score_sequence(batch_index, best_sequence, _trajectory):
            scored.append((batch_index, best_sequence))
            return [
                {
                    "batch_idx": batch_index,
                    "designed_sequence": best_sequence,
                    "critic_name": "critic",
                }
            ]

        result = run_design_loop(
            steps=3,
            batch_size=1,
            run_step=run_step,
            score_sequence=score_sequence,
        )

        self.assertEqual(result.best_sequences, ["TARGET|BBB"])
        self.assertEqual(scored, [(0, "TARGET|BBB")])
        self.assertEqual([call[0] for call in calls], [0, 1, 2])
        self.assertEqual(result.critic_results[0]["designed_sequence"], "TARGET|BBB")

    def test_run_design_loop_fallback_result_preserves_final_loss(self) -> None:
        def run_step(step: int, _temperature: float, _calculate_confidence: bool):
            return StepResult(
                sequences=[f"TARGET|B{step}"],
                iptm=[float(step)],
                losses={"total_loss": [10.0 - step]},
            )

        result = run_design_loop(
            steps=2,
            batch_size=1,
            run_step=run_step,
            score_sequence=lambda _batch_index, _best_sequence, _trajectory: [],
        )

        self.assertEqual(result.best_sequences, ["TARGET|B1"])
        self.assertEqual(result.critic_results[0]["designed_sequence"], "TARGET|B1")
        self.assertEqual(result.critic_results[0]["final_loss"], 9.0)


if __name__ == "__main__":
    unittest.main()
