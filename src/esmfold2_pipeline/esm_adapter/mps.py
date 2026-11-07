"""PyTorch-only compatibility helpers for the local Apple MPS worker."""
from __future__ import annotations

from contextlib import contextmanager
import os


# Model snapshots tested for this backend. Leave other aliases at their
# configured upstream defaults until they are qualified.
MPS_MODEL_REVISIONS = {
    "biohub/ESMFold2-Experimental-Fast-Cutoff2025": "13f892d3bc9de27aa475e8a12ec76dba4054f4a2",
}

_INT64_SCATTER_FALLBACKS = 0


def install_mps_compatibility(torch) -> None:
    """Route unsupported integer bookkeeping through CPU, leaving autograd on MPS."""
    current = torch.Tensor.scatter_reduce_
    if getattr(current, "_esmfold2_mps_compat", False):
        return

    def scatter_reduce(tensor, dim, index, src, reduce, *, include_self=True):
        global _INT64_SCATTER_FALLBACKS
        if tensor.device.type == "mps" and tensor.dtype == torch.int64:
            _INT64_SCATTER_FALLBACKS += 1
            result = current(
                tensor.cpu(), dim, index.cpu(), src.cpu(), reduce,
                include_self=include_self,
            )
            return tensor.copy_(result)
        return current(tensor, dim, index, src, reduce, include_self=include_self)

    scatter_reduce._esmfold2_mps_compat = True
    torch.Tensor.scatter_reduce_ = scatter_reduce


def int64_scatter_fallback_count() -> int:
    return _INT64_SCATTER_FALLBACKS


@contextmanager
def mps_seed_context(torch, upstream_context, seed):
    # The upstream context restores CPU and CUDA RNGs but not MPS's RNG.
    state = torch.mps.get_rng_state() if seed is not None else None
    try:
        with upstream_context(seed):
            yield
    finally:
        if state is not None:
            torch.mps.set_rng_state(state)


def optimize_mps_trunk(model) -> None:
    """Split frozen projection weights instead of slow strided activations.

    The input and all intermediate activations remain PyTorch tensors, so the
    full gradient with respect to soft residue types remains available.  The
    projection and transition code is adapted from abhi0raj/esmfold2-mlx
    (MIT; see esmfold2_pipeline/notices).
    """
    import torch
    import torch.nn.functional as F
    from transformers.models.esmfold2.modeling_esmfold2_common import (
        Transition,
        TriangleMultiplicativeBlock,
    )

    def triangle_forward(module, pair_grid, visibility=None):
        if visibility is None:
            visibility = pair_grid.new_ones(pair_grid.shape[:-1])
        normalized = module.norm_start(pair_grid)
        signal = F.linear(normalized, module._mps_signal_weight)
        gate = F.linear(normalized, module._mps_gate_weight)
        routed = signal * torch.sigmoid(gate) * visibility.unsqueeze(-1)
        left, right = routed.float().chunk(2, dim=-1)
        if module._chunk_size is None:
            contracted = module._triangular_contract(left, right)
        else:
            contracted = module._triangular_contract_chunked(left, right, module._chunk_size)
        mixed = module.proj_emit(module.norm_mix(contracted))
        return mixed * torch.sigmoid(module.proj_gate(normalized))

    def transition_forward(module, x):
        def apply(part):
            normalized = module.norm(part)
            hidden = F.silu(F.linear(normalized, module._mps_w1)) * F.linear(
                normalized, module._mps_w2
            )
            return part + F.linear(hidden, module.ffn.w3.weight)

        if module._chunk_size is None or x.shape[1] <= module._chunk_size:
            return apply(x)
        return torch.cat(
            [apply(x[:, start : start + module._chunk_size])
             for start in range(0, x.shape[1], module._chunk_size)],
            dim=1,
        )

    for module in model.modules():
        if isinstance(module, TriangleMultiplicativeBlock):
            if module.proj_bundle.bias is not None:
                raise RuntimeError("MPS triangle optimization requires bias-free weights")
            weight = module.proj_bundle.weight.detach()
            module._mps_signal_weight = weight[: 2 * module.latent_channels].contiguous()
            module._mps_gate_weight = weight[2 * module.latent_channels :].contiguous()
            module.forward = triangle_forward.__get__(module)
        elif isinstance(module, Transition):
            if module.ffn.w12.bias is not None or module.ffn.w3.bias is not None:
                raise RuntimeError("MPS transition optimization requires bias-free weights")
            weight = module.ffn.w12.weight.detach()
            module._mps_w1 = weight[: module.ffn.hidden_features].contiguous()
            module._mps_w2 = weight[module.ffn.hidden_features :].contiguous()
            module.forward = transition_forward.__get__(module)
    # Zero selects the unchunked path measured on the 240-residue benchmark.
    # Larger targets may need smaller working tensors on the same device.
    value = os.environ.get("ESMFOLD2_MPS_CHUNK_SIZE", "0")
    try:
        chunk_size = int(value)
        if chunk_size < 0:
            raise ValueError
    except ValueError:
        raise ValueError("ESMFOLD2_MPS_CHUNK_SIZE must be zero or a positive integer") from None
    model.set_chunk_size(chunk_size or None)
