from __future__ import annotations

import types
import unittest

import numpy as np

from esmfold2_pipeline.design import plm


class _FakeTensor(np.ndarray):
    __array_priority__ = 1000

    def __array_finalize__(self, _obj):
        pass

    def to(self, *args, **kwargs):
        if args and args[0] == np.float32:
            return _fake_tensor(self.astype(np.float32))
        dtype = kwargs.get("dtype")
        if dtype is not None:
            return _fake_tensor(self.astype(dtype))
        return self

    def detach(self):
        return self

    def argmax(self, *, dim: int):
        return _fake_tensor(np.argmax(self, axis=dim))

    def size(self, dim: int | None = None):
        return self.shape if dim is None else self.shape[dim]


def _fake_tensor(values):
    return np.asarray(values).view(_FakeTensor)


class _FakeTorch:
    @staticmethod
    def zeros(*shape, **_kwargs):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return _fake_tensor(np.zeros(shape, dtype=np.float32))


class _FakeFunctional:
    @staticmethod
    def one_hot(indices, *, num_classes: int):
        values = np.asarray(indices, dtype=int)
        result = np.zeros(values.shape + (num_classes,), dtype=np.float32)
        for index in np.ndindex(values.shape):
            result[index + (values[index],)] = 1.0
        return _fake_tensor(result)


class _FakeTokenizer:
    cls_token_id = 0
    eos_token_id = 1
    vocab = {
        "<cls>": 0,
        "<eos>": 1,
        "<pad>": 2,
        "<mask>": 3,
        "A": 4,
        "R": 5,
        "N": 6,
        "D": 7,
        "C": 8,
        "Q": 9,
        "E": 10,
        "G": 11,
        "H": 12,
        "I": 13,
        "L": 14,
        "K": 15,
        "M": 16,
        "F": 17,
        "P": 18,
        "S": 19,
        "T": 20,
        "W": 21,
        "Y": 22,
        "V": 23,
    }


class DesignPlmTest(unittest.TestCase):
    def test_folding_trunk_to_lm_vocab_matrix_matches_token_order(self) -> None:
        matrix = plm.folding_trunk_to_lm_aa_vocab_matrix(
            device="cpu",
            torch_module=_FakeTorch,
            tokenizer_factory=_FakeTokenizer,
        )

        self.assertEqual(matrix.shape, (20, 20))
        self.assertEqual(float(matrix[0, 0]), 1.0)  # ALA -> A
        self.assertEqual(float(matrix[4, 4]), 1.0)  # CYS -> C
        self.assertEqual(float(matrix[19, 19]), 1.0)  # VAL -> V
        self.assertEqual(float(matrix.sum()), 20.0)

    def test_one_hot_and_straight_through_helpers(self) -> None:
        probs = _fake_tensor([[[0.1, 0.9], [0.8, 0.2]]])

        one_hot = plm.one_hot_from_probs(probs, functional=_FakeFunctional)
        straight = plm.straight_through(one_hot, probs)

        np.testing.assert_array_equal(
            one_hot,
            np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32),
        )
        np.testing.assert_array_equal(straight, one_hot)


if __name__ == "__main__":
    unittest.main()
