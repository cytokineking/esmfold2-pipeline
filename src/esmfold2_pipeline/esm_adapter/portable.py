"""Source-only ESMFold2 design optimizations for the pinned experimental model.

The design loss uses the distogram. Early, seeded folds do not use confidence
metrics or sampled coordinates; final confidence folds and critics remain eager.

Lazy sampler adaptation inspired by Anthropic's ef2_lazy_structure.py and
skip_unused_confidence, Copyright 2026 Anthropic, PBC, Apache-2.0:
https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/ef2inv
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable


_LOGGER = logging.getLogger(__name__)
_SETTING = "ESMFOLD2_PIPELINE_OPTIMIZATIONS"
_COORDS = "sample_atom_coords"


def optimization_mode() -> str:
    mode = os.environ.get(_SETTING, "off").strip().lower()
    if mode not in {"off", "portable"}:
        raise ValueError(f"{_SETTING} must be 'off' or 'portable', got {mode!r}")
    return mode


class _DeferredSample:
    def __init__(self, sample: Callable[..., Any], args: tuple, kwargs: dict,
                 seed: int, autocast_enabled: bool, autocast_dtype: Any):
        self.sample = sample
        self.args = args
        self.kwargs = kwargs
        self.seed = seed
        self.autocast_enabled = autocast_enabled
        self.autocast_dtype = autocast_dtype
        self.result = None

    def materialize(self):
        if self.result is None:
            import torch
            from transformers.models.esmfold2.modeling_esmfold2_common import _seed_context

            with torch.no_grad(), _seed_context(self.seed), torch.autocast(
                "cuda", enabled=self.autocast_enabled, dtype=self.autocast_dtype
            ):
                self.result = self.sample(*self.args, **self.kwargs)
            self.sample = None
            self.args = ()
            self.kwargs = {}
        return self.result


class _Placeholder:
    pass


class LazySampleOutput(dict):
    """A dict that materializes the genuine sampled coordinates on access."""

    def __init__(self, output: dict, deferred: _DeferredSample):
        super().__init__(output)
        self._deferred = deferred

    def _materialize(self) -> None:
        if self._deferred is not None:
            result = self._deferred.materialize()
            if not isinstance(result, dict) or _COORDS not in result:
                raise RuntimeError("ESMFold2 sampler did not return coordinates")
            dict.__setitem__(self, _COORDS, result[_COORDS])
            self._deferred = None

    def __getitem__(self, key):
        if key == _COORDS:
            self._materialize()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key == _COORDS:
            self._materialize()
        return dict.get(self, key, default)

    def pop(self, key, *default):
        if key == _COORDS:
            self._materialize()
        return dict.pop(self, key, *default)

    def setdefault(self, key, default=None):
        if key == _COORDS:
            self._materialize()
        return dict.setdefault(self, key, default)

    def __iter__(self):
        return dict.__iter__(self)

    def values(self):
        self._materialize()
        return dict.values(self)

    def items(self):
        self._materialize()
        return dict.items(self)

    def copy(self):
        self._materialize()
        return dict.copy(self)

    def __repr__(self):
        self._materialize()
        return dict.__repr__(self)

    def __eq__(self, other):
        self._materialize()
        return dict.__eq__(self, other)

    def __reduce__(self):
        self._materialize()
        return (dict, (dict(self),))


def call_model(model, run: Callable[[], dict], *, calculate_confidence: bool,
               seed: int | None, num_sampling_steps: int) -> dict:
    """Call the stock model, omitting only results unused by a design step."""
    if optimization_mode() == "off" or calculate_confidence:
        return run()

    confidence_head = getattr(model, "confidence_head", None)
    if confidence_head is None:
        return run()

    structure_head = getattr(model, "structure_head", None)
    sample = getattr(structure_head, "sample", None)
    lazy = seed is not None and num_sampling_steps == 1 and sample is not None
    deferred = None
    had_sample_override = lazy and "sample" in vars(structure_head)
    if lazy:
        def defer_sample(*args, **kwargs):
            nonlocal deferred
            if deferred is not None:
                raise RuntimeError("ESMFold2 sampler called more than once in a design fold")
            import torch
            deferred = _DeferredSample(
                sample, args, kwargs, seed,
                torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"),
            )
            return {_COORDS: _Placeholder()}

        structure_head.sample = defer_sample

    model.confidence_head = None
    try:
        output = run()
    finally:
        model.confidence_head = confidence_head
        if lazy:
            if had_sample_override:
                structure_head.sample = sample
            else:
                del structure_head.sample

    stats = getattr(model, "_esmfold2_portable_stats", None)
    if stats is None:
        stats = {"confidence_skipped": 0, "sampling_deferred": 0}
        model._esmfold2_portable_stats = stats
        _LOGGER.info("ESMFold2 portable model path enabled: %s", type(model).__name__)
    stats["confidence_skipped"] += 1
    if deferred is not None:
        if not isinstance(output, dict) or not isinstance(dict.get(output, _COORDS), _Placeholder):
            raise RuntimeError("ESMFold2 forward changed its sampled-coordinate contract")
        stats["sampling_deferred"] += 1
        return LazySampleOutput(output, deferred)
    return output
