from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import random
from typing import Any, Callable

from esmfold2_pipeline.design.spec import DesignRunResult


MUTABLE_TOKEN = "#"
PROTEIN_1TO3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}
TOKENS = (
    "<pad>",
    "-",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "A",
    "G",
    "C",
    "U",
    "N",
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",
)
TOKEN_IDS = {token: index for index, token in enumerate(TOKENS)}
AA_DIMS = 20
CYS_IDX = TOKEN_IDS[PROTEIN_1TO3["C"]] - 2
LEARNING_RATE = 0.1
TEMPERATURE_MIN = 1e-2
ESMC_MASK_FRACTION = 0.15


@dataclass(frozen=True)
class StepResult:
    sequences: list[str]
    iptm: list[float | None] | None
    losses: dict[str, Any]


RunStep = Callable[[int, float, bool], StepResult]
ScoreSequence = Callable[[int, str, dict[int, dict[str, Any]]], list[dict[str, Any]]]


def plm_gradient_weight(is_antibody: bool | None) -> float:
    return 0.05 if is_antibody else 0.15


def design_fold_sampling_steps(*, calculate_confidence: bool) -> int:
    return 50 if calculate_confidence else 1


def select_inversion_model(
    inversion_models: dict[str, Any],
    *,
    seed: int,
    step: int,
) -> Any:
    if not inversion_models:
        raise ValueError("inversion_models cannot be empty")
    random.seed(seed + step)
    replicate_choice = random.randint(0, len(inversion_models) - 1)
    return list(inversion_models.values())[replicate_choice]


def build_initial_soft_sequence_logits(
    sequence: str,
    batch_size: int,
    *,
    torch_module: Any | None = None,
):
    torch = torch_module or _torch()
    if all(aa == MUTABLE_TOKEN for aa in sequence):
        logits = 0.01 * torch.randn([batch_size, len(sequence), AA_DIMS])
        logits[:, :, CYS_IDX] = -1e6
    else:
        logits = torch.zeros([batch_size, len(sequence), AA_DIMS])
        for index, aa in enumerate(sequence):
            if aa == MUTABLE_TOKEN:
                logits[:, index, :] = 0.01 * torch.randn(batch_size, AA_DIMS)
                logits[:, index, CYS_IDX] = -1e6
            else:
                if aa not in PROTEIN_1TO3:
                    raise AssertionError(aa)
                token_id = TOKEN_IDS[PROTEIN_1TO3[aa]]
                logits[:, index, token_id - 2] = 10.0

    if hasattr(logits, "requires_grad_"):
        return logits.requires_grad_(True)
    return logits


def build_gradient_mask(
    sequence: str,
    batch_size: int,
    *,
    torch_module: Any | None = None,
):
    torch = torch_module or _torch()
    mask = torch.ones([batch_size, len(sequence), AA_DIMS])
    fixed_positions = [
        index for index, aa in enumerate(sequence) if aa != MUTABLE_TOKEN
    ]
    mask[:, fixed_positions, :] = 0.0
    mask[:, :, CYS_IDX] = 0.0
    return mask


def sequence_to_one_hot_indices(sequence: str) -> list[int]:
    return [TOKEN_IDS[PROTEIN_1TO3[letter]] for letter in sequence]


def sequence_to_one_hot(
    sequence: str,
    *,
    device: str = "cuda",
    torch_module: Any | None = None,
    functional: Any | None = None,
):
    torch = torch_module or _torch()
    functional = functional or _torch_functional()
    one_hot = functional.one_hot(
        torch.tensor(sequence_to_one_hot_indices(sequence)),
        num_classes=len(TOKENS),
    )
    return one_hot.to(device).unsqueeze(0).float()


def normalized_gradient_tensor(
    grad,
    gradient_mask,
    *,
    torch_module: Any | None = None,
):
    torch = torch_module or _torch()
    masked_grad = grad * gradient_mask
    index_has_nonzero_grad = torch.square(masked_grad).sum(-1) > 0
    eff_l = index_has_nonzero_grad.sum(-1)
    grad_norm = torch.linalg.norm(masked_grad, axis=(-1, -2))
    normalized_grad = (masked_grad / (grad_norm[:, None, None] + 1e-7)) * torch.sqrt(
        eff_l[:, None, None]
    )
    return normalized_grad * gradient_mask


def run_gradient_design_loop(
    *,
    target_sequence: str,
    binder_sequence: str,
    is_antibody: bool | None,
    seed: int,
    steps: int,
    batch_size: int,
    inversion_models: dict[str, Any],
    critic_models: dict[str, Any],
    esmc_model: Any,
    fold_complex: Callable[..., dict[str, Any]],
    compute_structure_losses: Callable[[Any, int], dict[str, Any]],
    compute_plm_loss: Callable[..., Any],
    build_complex: Callable[[Any, Any], Any],
    compute_distogram_iptm_proxy: Callable[[Any, int, str, bool | None], dict[str, Any]],
    torch_module: Any | None = None,
    functional: Any | None = None,
    optim_module: Any | None = None,
    seed_context: Callable[[int], Any] | None = None,
    device: str = "cuda",
) -> DesignRunResult:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if "|" in binder_sequence:
        raise AssertionError("binder_sequence must be a single chain")

    torch = torch_module or _torch()
    functional = functional or _torch_functional()
    optim = optim_module or _torch_optim()
    seed_context = seed_context or (lambda _seed: nullcontext())
    target_one_hot = sequence_to_one_hot(
        target_sequence,
        device=device,
        torch_module=torch,
        functional=functional,
    )
    binder_length = len(binder_sequence)

    with seed_context(seed), torch.device(device):
        logits = build_initial_soft_sequence_logits(
            binder_sequence,
            batch_size=batch_size,
            torch_module=torch,
        )
        gradient_mask = build_gradient_mask(
            binder_sequence,
            batch_size=batch_size,
            torch_module=torch,
        )

    optimizer = optim.SGD([logits], lr=LEARNING_RATE)
    last_design_fold: dict[str, Any] | None = None
    last_confidence_fold: dict[str, Any] | None = None

    def run_step(
        step: int,
        temperature: float,
        calculate_confidence: bool,
    ) -> StepResult:
        nonlocal logits, last_design_fold, last_confidence_fold
        optimizer.zero_grad()

        inversion_model = select_inversion_model(
            inversion_models,
            seed=seed,
            step=step,
        )
        design = functional.softmax(logits / temperature, dim=-1)
        fold_result = fold_complex(
            inversion_model,
            target_sequence,
            target_one_hot,
            design,
            num_loops=1,
            num_sampling_steps=design_fold_sampling_steps(
                calculate_confidence=calculate_confidence,
            ),
            calculate_confidence=calculate_confidence,
            seed=seed + step,
        )
        last_design_fold = fold_result
        sequences = list(fold_result["seq_list"])
        if calculate_confidence:
            last_confidence_fold = fold_result

        losses = compute_structure_losses(
            fold_result["distogram_logits"],
            binder_length,
        )
        structure_loss = losses["total_loss"]
        structure_grad = torch.autograd.grad(structure_loss.mean(), logits)[0]

        design = functional.softmax(logits / temperature, dim=-1)
        score_mask = gradient_mask.sum(dim=-1) > 0
        with seed_context(seed + step):
            plm_loss = compute_plm_loss(
                esmc_model=esmc_model,
                binder_design=design,
                score_mask=score_mask,
                batch_size=4,
                n_passes=4,
            )
        plm_grad = torch.autograd.grad(plm_loss.mean(), logits)[0]

        logits.grad = normalized_gradient_tensor(
            structure_grad,
            gradient_mask,
            torch_module=torch,
        ) + plm_gradient_weight(is_antibody) * normalized_gradient_tensor(
            plm_grad,
            gradient_mask,
            torch_module=torch,
        )

        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * temperature
        optimizer.step()

        step_losses = {key: value.detach().cpu() for key, value in losses.items()}
        step_losses["plm_loss"] = plm_loss.detach().cpu()
        step_losses["total_loss"] = (structure_loss + plm_loss).detach().cpu()
        return StepResult(
            sequences=sequences,
            iptm=fold_result.get("iptm", None),
            losses=step_losses,
        )

    def score_sequence(
        batch_index: int,
        best_sequence: str,
        trajectory: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        nonlocal last_confidence_fold
        target_length = len(target_sequence.replace("|", ""))
        binder_seq = best_sequence.split("|")[-1]
        binder_design = sequence_to_one_hot(
            binder_seq,
            device=device,
            torch_module=torch,
            functional=functional,
        )[..., 2:22]
        results: list[dict[str, Any]] = []
        for critic_name, critic_model in critic_models.items():
            is_scaling_critic = "ESMFold2-Experimental-Fast-base" in critic_name
            if is_scaling_critic:
                critic_model.cuda()
            final_fold = fold_complex(
                critic_model,
                target_sequence,
                target_one_hot,
                binder_design,
                num_loops=3,
                num_sampling_steps=200,
                calculate_confidence=True,
                seed=seed,
            )
            last_confidence_fold = final_fold
            if is_scaling_critic:
                critic_model.cpu()
            complex_obj = build_complex(final_fold["inputs"], final_fold["output"])
            iptm_proxy_scores = compute_distogram_iptm_proxy(
                final_fold["distogram_logits"],
                target_length,
                binder_seq,
                is_antibody,
            )
            iptm_value = final_fold.get("iptm")
            iptm = iptm_value.item() if iptm_value is not None else None
            results.append(
                {
                    "is_antibody": is_antibody,
                    "critic_name": critic_name,
                    "batch_idx": batch_index,
                    "designed_sequence": best_sequence,
                    "complex": complex_obj,
                    "final_loss": _loss_item(
                        trajectory[steps - 1]["total_loss"],
                        batch_index,
                    ),
                    "iptm": iptm,
                    "logits": logits[batch_index].detach().cpu(),
                    **iptm_proxy_scores,
                }
            )
        return results

    result = run_design_loop(
        steps=steps,
        batch_size=batch_size,
        run_step=run_step,
        score_sequence=score_sequence,
    )
    return DesignRunResult(
        best_sequences=result.best_sequences,
        trajectory=result.trajectory,
        critic_results=result.critic_results,
        last_design_fold=last_design_fold,
        last_confidence_fold=last_confidence_fold,
    )


def temperature_for_step(step: int, steps: int) -> float:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if step < 0 or step >= steps:
        raise ValueError(f"step {step} is outside [0, {steps})")
    t = (step + 1) / steps
    remaining = 0.5 * (1 + math.cos(math.pi * t))
    return TEMPERATURE_MIN + (1 - TEMPERATURE_MIN) * remaining


def calculate_confidence_for_temperature(temperature: float) -> bool:
    return temperature < 0.05


def run_design_loop(
    *,
    steps: int,
    batch_size: int,
    run_step: RunStep,
    score_sequence: ScoreSequence,
) -> DesignRunResult:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    trajectory: dict[int, dict[str, Any]] = {}
    best_iptm: list[float] = [-1.0] * batch_size
    best_sequences: list[str] = [""] * batch_size

    for step in range(steps):
        temperature = temperature_for_step(step, steps)
        step_result = run_step(
            step,
            temperature,
            calculate_confidence_for_temperature(temperature),
        )
        if len(step_result.sequences) != batch_size:
            raise ValueError(
                f"step {step} returned {len(step_result.sequences)} sequence(s), "
                f"expected batch_size={batch_size}"
            )
        trajectory[step] = step_result.losses
        if step_result.iptm is None:
            continue
        if len(step_result.iptm) != batch_size:
            raise ValueError(
                f"step {step} returned {len(step_result.iptm)} iPTM value(s), "
                f"expected batch_size={batch_size}"
            )
        for batch_index, value in enumerate(step_result.iptm):
            if value is not None and value > best_iptm[batch_index]:
                best_iptm[batch_index] = float(value)
                best_sequences[batch_index] = step_result.sequences[batch_index]

    if not all(sequence != "" for sequence in best_sequences):
        raise AssertionError("design loop did not select a best sequence for every batch")

    critic_results: list[dict[str, Any]] = []
    for batch_index, best_sequence in enumerate(best_sequences):
        critic_results.extend(score_sequence(batch_index, best_sequence, trajectory))

    if not critic_results:
        final_losses = trajectory[steps - 1].get("total_loss")
        for batch_index, best_sequence in enumerate(best_sequences):
            critic_results.append(
                {
                    "is_antibody": None,
                    "batch_idx": batch_index,
                    "designed_sequence": best_sequence,
                    "final_loss": _loss_item(final_losses, batch_index),
                }
            )

    return DesignRunResult(
        best_sequences=best_sequences,
        trajectory=trajectory,
        critic_results=critic_results,
    )


def _loss_item(value: Any, batch_index: int) -> Any:
    if value is None:
        return None
    try:
        value = value[batch_index]
    except (IndexError, KeyError, TypeError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _torch():
    import torch

    return torch


def _torch_functional():
    import torch.nn.functional as functional

    return functional


def _torch_optim():
    import torch.optim as optim

    return optim
