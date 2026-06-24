from __future__ import annotations

import math
from typing import Any, Callable

from esmfold2_pipeline.design.loop import ESMC_MASK_FRACTION, PROTEIN_1TO3, TOKENS


def folding_trunk_to_lm_aa_vocab_matrix(
    *,
    device,
    torch_module: Any | None = None,
    tokenizer_factory: Callable[[], Any] | None = None,
):
    torch = _torch(torch_module)
    tokenizer_factory = tokenizer_factory or _esm_tokenizer_factory()
    three_to_one_map = {value: key for key, value in PROTEIN_1TO3.items()}
    ft_aas = [three_to_one_map[token_3letter] for token_3letter in TOKENS[2:22]]

    lm_vocab = sorted(tokenizer_factory().vocab.items(), key=lambda item: item[1])
    lm_aas = [lm_vocab[index][0] for index in range(4, 24)]

    matrix = torch.zeros(20, 20)
    for ft_index, ft_aa in enumerate(ft_aas):
        lm_index = lm_aas.index(ft_aa)
        matrix[ft_index, lm_index] = 1

    return matrix.to(device=device)


def one_hot_from_probs(probs, *, functional: Any | None = None):
    functional = functional or _torch_functional()
    return functional.one_hot(
        probs.argmax(dim=-1),
        num_classes=probs.size(-1),
    ).to(probs.dtype)


def straight_through(discrete, continuous):
    return continuous + (discrete - continuous).detach()


def compute_esmc_pseudoperplexity_nll(
    *,
    esmc_model,
    binder_design,
    score_mask,
    batch_size: int = 4,
    n_passes: int = 4,
    torch_module: Any | None = None,
    functional: Any | None = None,
    tokenizer_factory: Callable[[], Any] | None = None,
):
    """Algorithm 14 ESMC pseudo-perplexity sequence regularization."""

    torch = _torch(torch_module)
    functional = functional or _torch_functional()
    tokenizer_factory = tokenizer_factory or _esm_tokenizer_factory()
    device = binder_design.device
    lm_vocab_size = esmc_model.config.vocab_size
    model_dtype = esmc_model.esmc.embed.weight.dtype

    target_esm = binder_design @ folding_trunk_to_lm_aa_vocab_matrix(
        device=device,
        torch_module=torch,
        tokenizer_factory=tokenizer_factory,
    )
    input_esm = straight_through(
        one_hot_from_probs(target_esm, functional=functional),
        target_esm,
    )
    input_ids = torch.zeros(
        (binder_design.size(0), binder_design.size(1) + 2, lm_vocab_size),
        dtype=model_dtype,
        device=device,
    )
    tokenizer = tokenizer_factory()
    input_ids[:, 0, tokenizer.cls_token_id] = 1
    input_ids[:, -1, tokenizer.eos_token_id] = 1
    input_ids[:, 1:-1, 4:24] = input_esm.to(model_dtype)

    if score_mask.ndim == 1:
        score_mask = score_mask.unsqueeze(0).expand(binder_design.size(0), -1)
    elif score_mask.shape != binder_design.shape[:2]:
        raise ValueError(
            f"Expected score_mask with shape "
            f"{(binder_design.size(0), binder_design.size(1))}, "
            f"got {tuple(score_mask.shape)}"
        )
    score_mask = score_mask.to(device=device, dtype=torch.bool)

    mask_token = torch.zeros(lm_vocab_size, dtype=model_dtype, device=device)
    mask_token[esmc_model.config.mask_token_id] = 1
    esmc = esmc_model.esmc

    losses = []
    for batch_index in range(binder_design.size(0)):
        position_indices = score_mask[batch_index].nonzero(as_tuple=False).flatten()
        num_positions = int(position_indices.numel())
        if num_positions == 0:
            raise ValueError(
                "ESMC pseudoperplexity score mask selected zero positions."
            )

        num_masked = max(1, math.ceil(ESMC_MASK_FRACTION * num_positions))
        random_scores = torch.rand((n_passes, num_positions), device=device)
        masked_offsets = random_scores.topk(
            num_masked,
            dim=-1,
            largest=False,
        ).indices
        pass_masks = torch.zeros(
            (n_passes, binder_design.size(1)),
            dtype=torch.bool,
            device=device,
        )
        pass_masks[
            torch.arange(n_passes, device=device)[:, None],
            position_indices[masked_offsets],
        ] = True

        masked_sequences = input_ids[batch_index : batch_index + 1].repeat(
            n_passes,
            1,
            1,
        )
        mask_rows, mask_cols = pass_masks.nonzero(as_tuple=True)
        masked_sequences[mask_rows, mask_cols + 1] = mask_token

        target_weights = target_esm[batch_index]
        masked_nlls = []
        for start in range(0, n_passes, batch_size):
            stop = min(start + batch_size, n_passes)
            chunk = masked_sequences[start:stop]
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                hidden, *_unused = esmc.transformer(
                    chunk @ esmc.embed.weight.to(chunk.dtype),
                    sequence_id=None,
                    layers_to_collect=[],
                    output_attentions=False,
                )
                logits = esmc_model.lm_head(hidden)
            log_probs = logits.log_softmax(dim=-1)[:, 1:-1, 4:24]
            nlls = -(
                log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)
            ).sum(dim=-1)
            masked_nlls.append(nlls[pass_masks[start:stop]])

        losses.append(torch.cat(masked_nlls, dim=0).mean())

    return torch.stack(losses, dim=0)


def _torch(torch_module: Any | None):
    if torch_module is not None:
        return torch_module
    import torch  # type: ignore

    return torch


def _torch_functional():
    import torch.nn.functional as functional

    return functional


def _esm_tokenizer_factory():
    from transformers.models.esmc.tokenization_esmc import ESMCTokenizer

    return ESMCTokenizer
