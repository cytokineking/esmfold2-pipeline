from __future__ import annotations

from typing import Any

import numpy as np

from esmfold2_pipeline.config import HOTSPOT_LOSS_MODES


LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2}


def get_mid_points(torch_module: Any | None = None):
    torch = _torch(torch_module)
    boundaries = torch.linspace(2, 52.0, 127)
    lower = torch.tensor([1.0])
    upper = torch.tensor([52.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    return (exp_boundaries[:-1] + exp_boundaries[1:]) / 2


def binned_entropy(
    dgram,
    bin_distance,
    cutoff: float,
    *,
    torch_module: Any | None = None,
):
    torch = _torch(torch_module)
    bin_distance = bin_distance.to(device=dgram.device, dtype=dgram.dtype)
    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def masked_min_k(
    x,
    mask,
    k: int,
    *,
    torch_module: Any | None = None,
):
    torch = _torch(torch_module)
    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def masked_average(
    x,
    mask,
    *,
    torch_module: Any | None = None,
):
    torch = _torch(torch_module)
    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


def compute_contact_loss(
    distogram_logits,
    bin_distance,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask,
    binder_mask,
    *,
    torch_module: Any | None = None,
):
    torch = _torch(torch_module)
    con_loss = binned_entropy(
        distogram_logits,
        bin_distance,
        cutoff,
        torch_module=torch,
    )
    position = torch.arange(
        distogram_logits.shape[1],
        device=distogram_logits.device,
    )
    p_dist = position[:, None] - position[None, :]
    if min_sep > 0:
        separation_mask = (torch.abs(p_dist) >= min_sep).to(distogram_logits.device)
        binder_mask = torch.logical_and(separation_mask, binder_mask)
    per_residue = masked_min_k(
        con_loss,
        mask=binder_mask,
        k=num_contacts,
        torch_module=torch,
    ).to(distogram_logits.device)
    return masked_average(
        per_residue,
        mask=chain_mask,
        torch_module=torch,
    ).to(distogram_logits.device)


def compute_intra_contact_loss(
    distogram_logits,
    binder_length: int,
    bin_distance,
    *,
    torch_module: Any | None = None,
):
    torch = _torch(torch_module)
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=2,
        min_sep=9,
        cutoff=14.0,
        chain_mask=is_binder,
        binder_mask=is_binder,
        torch_module=torch,
    )


def compute_inter_contact_loss(
    distogram_logits,
    binder_length: int,
    bin_distance,
    *,
    torch_module: Any | None = None,
):
    torch = _torch(torch_module)
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=1,
        min_sep=0,
        cutoff=22.0,
        chain_mask=1 - is_binder,
        binder_mask=is_binder,
        torch_module=torch,
    )


def compute_globularity_loss(
    distogram_logits,
    binder_length: int,
    bin_distance,
    *,
    torch_module: Any | None = None,
):
    torch = _torch(torch_module)
    binder_disto = distogram_logits[:, -binder_length:, -binder_length:, :]
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    bin_distance = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(bin_distance), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n**0.365)
    return torch.nn.functional.elu(rg_term - rg_th)


def compute_structure_losses(
    distogram_logits,
    binder_length: int,
    *,
    bin_distance=None,
    torch_module: Any | None = None,
    include_inter_contact: bool = True,
) -> dict[str, Any]:
    torch = _torch(torch_module)
    if bin_distance is None:
        bin_distance = get_mid_points(torch)
    bin_distance = bin_distance.to(device=distogram_logits.device, dtype=distogram_logits.dtype)
    losses: dict[str, Any] = {}
    losses["intra_contact_loss"] = compute_intra_contact_loss(
        distogram_logits,
        binder_length,
        bin_distance,
        torch_module=torch,
    )
    if include_inter_contact:
        losses["inter_contact_loss"] = compute_inter_contact_loss(
            distogram_logits,
            binder_length,
            bin_distance,
            torch_module=torch,
        )
    losses["glob_loss"] = compute_globularity_loss(
        distogram_logits,
        binder_length,
        bin_distance,
        torch_module=torch,
    )
    batch_size = distogram_logits.size(0)
    total = torch.tensor(
        [0.0] * batch_size,
        device=distogram_logits.device,
        requires_grad=True,
    )
    total = total + LOSS_WEIGHTS["intra_contact"] * losses["intra_contact_loss"]
    if include_inter_contact:
        total = total + LOSS_WEIGHTS["inter_contact"] * losses["inter_contact_loss"]
    total = total + LOSS_WEIGHTS["glob"] * losses["glob_loss"]
    losses["total_loss"] = total
    return losses


def compute_design_structure_losses(
    distogram_logits,
    binder_length: int,
    *,
    torch_module: Any | None = None,
    bin_distance=None,
    target_geometry_reference_distances: np.ndarray | None = None,
    target_geometry_pair_mask: np.ndarray | None = None,
    target_geometry_weight: float = 0.0,
    target_geometry_tolerance_angstrom: float | None = None,
    target_geometry_stiffness_angstrom: float | None = None,
    hotspot_indices: tuple[int, ...] = (),
    hotspot_contact_weight: float = 0.0,
    hotspot_contact_cutoff_angstrom: float | None = None,
    hotspot_num_contacts: int = 1,
    hotspot_contact_probability_target: float = 0.6,
    hotspot_loss_mode: str | None = None,
    binder_contact_indices: tuple[int, ...] | None = None,
    include_inter_contact: bool = True,
) -> dict[str, Any]:
    losses = compute_structure_losses(
        distogram_logits,
        binder_length,
        torch_module=torch_module,
        bin_distance=bin_distance,
        include_inter_contact=include_inter_contact,
    )
    if target_geometry_weight > 0:
        if target_geometry_reference_distances is None:
            raise ValueError("target geometry drift requires reference distances")
        if target_geometry_pair_mask is None:
            raise ValueError("target geometry drift requires a pair mask")
        if target_geometry_tolerance_angstrom is None:
            raise ValueError("target geometry drift requires a tolerance")
        if target_geometry_stiffness_angstrom is None:
            raise ValueError("target geometry drift requires a stiffness")
        drift_loss, drift_rmse = compute_target_geometry_drift_loss(
            torch_module,
            distogram_logits,
            binder_length,
            reference_distances=target_geometry_reference_distances,
            pair_mask=target_geometry_pair_mask,
            tolerance_angstrom=target_geometry_tolerance_angstrom,
            stiffness_angstrom=target_geometry_stiffness_angstrom,
            bin_distances=bin_distance,
        )
        losses["target_geometry_drift_loss"] = drift_loss
        losses["target_geometry_drift_rmse"] = drift_rmse
        losses["total_loss"] = losses["total_loss"] + target_geometry_weight * drift_loss

    if hotspot_indices and hotspot_contact_weight > 0:
        if hotspot_contact_cutoff_angstrom is None:
            raise ValueError("hotspot contact loss requires a contact cutoff")
        if hotspot_loss_mode is None:
            raise ValueError("hotspot contact loss requires a loss mode")
        hotspot_loss = compute_hotspot_contact_loss(
            torch_module,
            distogram_logits,
            binder_length,
            hotspot_indices=hotspot_indices,
            contact_cutoff_angstrom=hotspot_contact_cutoff_angstrom,
            hotspot_num_contacts=hotspot_num_contacts,
            contact_probability_target=hotspot_contact_probability_target,
            hotspot_loss_mode=hotspot_loss_mode,
            binder_contact_indices=binder_contact_indices,
            bin_distances=bin_distance,
        )
        losses["hotspot_contact_loss"] = hotspot_loss
        losses["total_loss"] = losses["total_loss"] + hotspot_contact_weight * hotspot_loss

    return losses


def compute_target_geometry_drift_loss(
    torch_module,
    distogram_logits,
    binder_length: int,
    *,
    reference_distances: np.ndarray,
    pair_mask: np.ndarray,
    tolerance_angstrom: float,
    stiffness_angstrom: float,
    bin_distances=None,
):
    torch = _torch(torch_module)
    target_length = int(distogram_logits.shape[1]) - binder_length
    if target_length <= 0:
        raise ValueError("target length inferred from distogram logits is not positive")
    if reference_distances.shape != (target_length, target_length):
        raise ValueError(
            "target geometry reference distances do not match folded target length: "
            f"{reference_distances.shape} != {(target_length, target_length)}"
        )
    if pair_mask.shape != (target_length, target_length):
        raise ValueError(
            "target geometry pair mask does not match folded target length: "
            f"{pair_mask.shape} != {(target_length, target_length)}"
        )

    if bin_distances is None:
        bin_distances = get_mid_points(torch)
    bin_distances = bin_distances.to(
        device=distogram_logits.device,
        dtype=distogram_logits.dtype,
    )
    if int(bin_distances.shape[0]) != int(distogram_logits.shape[-1]):
        raise ValueError(
            "ESMFold2 distogram bin count does not match get_mid_points(): "
            f"{distogram_logits.shape[-1]} != {bin_distances.shape[0]}"
        )
    target_logits = distogram_logits[:, :target_length, :target_length, :]
    predicted_distances = torch.softmax(target_logits, dim=-1).matmul(bin_distances)
    reference_tensor = torch.as_tensor(
        reference_distances,
        dtype=predicted_distances.dtype,
        device=predicted_distances.device,
    )
    reference_tensor = torch.minimum(reference_tensor, bin_distances[-1])
    mask_tensor = torch.as_tensor(
        pair_mask,
        dtype=torch.bool,
        device=predicted_distances.device,
    )
    if not bool(mask_tensor.any().item()):
        raise ValueError("target geometry drift loss selected no valid residue pairs")

    delta = predicted_distances - reference_tensor
    selected_delta = delta[:, mask_tensor]
    drift_rmse = torch.sqrt(selected_delta.pow(2).mean(dim=-1) + 1e-8)
    drift_loss = compute_target_geometry_drift_hinge_loss(
        torch,
        drift_rmse,
        tolerance_angstrom=tolerance_angstrom,
        stiffness_angstrom=stiffness_angstrom,
    )
    return drift_loss, drift_rmse


def compute_target_geometry_drift_hinge_loss(
    torch_module,
    drift_rmse,
    *,
    tolerance_angstrom: float,
    stiffness_angstrom: float,
):
    torch = _torch(torch_module)
    if tolerance_angstrom <= 0:
        raise ValueError("target geometry drift tolerance must be positive")
    if stiffness_angstrom <= 0:
        raise ValueError("target geometry drift stiffness must be positive")
    return torch.relu((drift_rmse - tolerance_angstrom) / stiffness_angstrom)


def compute_hotspot_contact_loss(
    torch_module,
    distogram_logits,
    binder_length: int,
    *,
    hotspot_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    hotspot_num_contacts: int,
    contact_probability_target: float,
    hotspot_loss_mode: str,
    binder_contact_indices: tuple[int, ...] | None = None,
    bin_distances=None,
):
    if hotspot_loss_mode == "probability_hinge":
        return compute_hotspot_contact_probability_hinge_loss(
            torch_module,
            distogram_logits,
            binder_length,
            hotspot_indices=hotspot_indices,
            contact_cutoff_angstrom=contact_cutoff_angstrom,
            hotspot_num_contacts=hotspot_num_contacts,
            contact_probability_target=contact_probability_target,
            binder_contact_indices=binder_contact_indices,
            bin_distances=bin_distances,
        )
    if hotspot_loss_mode == "entropy_hotspot":
        return compute_hotspot_entropy_contact_loss(
            torch_module,
            distogram_logits,
            binder_length,
            hotspot_indices=hotspot_indices,
            contact_cutoff_angstrom=contact_cutoff_angstrom,
            hotspot_num_contacts=hotspot_num_contacts,
            binder_contact_indices=binder_contact_indices,
            bin_distances=bin_distances,
        )

    choices = ", ".join(sorted(HOTSPOT_LOSS_MODES))
    raise ValueError(f"hotspot_loss_mode must be one of: {choices}")


def compute_hotspot_entropy_contact_loss(
    torch_module,
    distogram_logits,
    binder_length: int,
    *,
    hotspot_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    hotspot_num_contacts: int,
    binder_contact_indices: tuple[int, ...] | None = None,
    bin_distances=None,
):
    torch = _torch(torch_module)
    target_length = _validate_hotspot_indices(
        distogram_logits,
        binder_length,
        hotspot_indices=hotspot_indices,
    )
    full_len = distogram_logits.shape[1]
    binder_mask = binder_contact_mask(
        torch,
        full_len=full_len,
        target_length=target_length,
        binder_length=binder_length,
        binder_contact_indices=binder_contact_indices,
        device=distogram_logits.device,
    )
    hotspot_mask = torch.zeros(full_len, device=distogram_logits.device)
    hotspot_mask[list(hotspot_indices)] = 1.0
    if bin_distances is None:
        bin_distances = get_mid_points(torch)
    return compute_contact_loss(
        distogram_logits,
        bin_distances.to(distogram_logits.device),
        num_contacts=hotspot_num_contacts,
        min_sep=0,
        cutoff=contact_cutoff_angstrom,
        chain_mask=hotspot_mask,
        binder_mask=binder_mask,
        torch_module=torch,
    )


def compute_hotspot_contact_probability_hinge_loss(
    torch_module,
    distogram_logits,
    binder_length: int,
    *,
    hotspot_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    hotspot_num_contacts: int,
    contact_probability_target: float,
    binder_contact_indices: tuple[int, ...] | None = None,
    bin_distances=None,
):
    torch = _torch(torch_module)
    per_hotspot_score = hotspot_contact_probability_scores(
        torch,
        distogram_logits,
        binder_length,
        hotspot_indices=hotspot_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        hotspot_num_contacts=hotspot_num_contacts,
        binder_contact_indices=binder_contact_indices,
        bin_distances=bin_distances,
    )
    deficit = torch.relu(contact_probability_target - per_hotspot_score)
    return (deficit / contact_probability_target).pow(2).mean(dim=-1)


def hotspot_contact_probability_scores(
    torch_module,
    distogram_logits,
    binder_length: int,
    *,
    hotspot_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    hotspot_num_contacts: int,
    binder_contact_indices: tuple[int, ...] | None = None,
    bin_distances=None,
):
    torch = _torch(torch_module)
    if hotspot_num_contacts <= 0:
        raise ValueError("hotspot_num_contacts must be positive")
    target_length = _validate_hotspot_indices(
        distogram_logits,
        binder_length,
        hotspot_indices=hotspot_indices,
    )
    if bin_distances is None:
        bin_distances = get_mid_points(torch)
    bin_distances = bin_distances.to(distogram_logits.device)
    contact_bins = bin_distances <= contact_cutoff_angstrom
    if not bool(contact_bins.any().item()):
        raise ValueError(
            "contact_cutoff_angstrom does not include any ESMFold2 distogram bins"
        )

    probs = torch.softmax(distogram_logits, dim=-1)
    contact_prob = probs[..., contact_bins].sum(dim=-1)
    binder_start = target_length
    if binder_contact_indices is None:
        binder_columns = torch.arange(
            binder_start,
            binder_start + binder_length,
            device=distogram_logits.device,
        )
    else:
        validate_binder_contact_indices(binder_contact_indices, binder_length)
        binder_columns = torch.tensor(
            [binder_start + index for index in binder_contact_indices],
            device=distogram_logits.device,
            dtype=torch.long,
        )
    hotspot_to_binder = contact_prob[:, list(hotspot_indices), :].index_select(
        dim=-1,
        index=binder_columns,
    )
    k = min(hotspot_num_contacts, int(binder_columns.numel()))
    return torch.topk(hotspot_to_binder, k=k, dim=-1).values.mean(dim=-1)


def compute_mosaic_cdr_contact_loss(
    torch_module,
    distogram_logits,
    binder_length: int,
    *,
    cdr_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    num_target_contacts: int,
    hotspot_indices: tuple[int, ...] = (),
    bin_distances=None,
):
    torch = _torch(torch_module)
    if contact_cutoff_angstrom <= 0:
        raise ValueError("contact_cutoff_angstrom must be positive")
    if num_target_contacts <= 0:
        raise ValueError("num_target_contacts must be positive")
    validate_binder_contact_indices(cdr_indices, binder_length)
    target_length = _validate_hotspot_indices(
        distogram_logits,
        binder_length,
        hotspot_indices=hotspot_indices,
    )
    full_len = int(distogram_logits.shape[1])
    cdr_rows = _binder_absolute_indices(
        torch,
        binder_length=binder_length,
        target_length=target_length,
        binder_indices=cdr_indices,
        device=distogram_logits.device,
    )
    target_columns = _target_contact_columns(
        torch,
        target_length=target_length,
        hotspot_indices=hotspot_indices,
        device=distogram_logits.device,
    )
    cdr_mask = torch.zeros(full_len, device=distogram_logits.device)
    cdr_mask[cdr_rows] = 1.0
    target_mask = torch.zeros(full_len, device=distogram_logits.device)
    target_mask[target_columns] = 1.0
    if bin_distances is None:
        bin_distances = get_mid_points(torch)
    bin_distances = bin_distances.to(
        device=distogram_logits.device,
        dtype=distogram_logits.dtype,
    )
    return compute_contact_loss(
        distogram_logits,
        bin_distances,
        num_contacts=num_target_contacts,
        min_sep=0,
        cutoff=contact_cutoff_angstrom,
        chain_mask=cdr_mask,
        binder_mask=target_mask,
        torch_module=torch,
    )


def mosaic_cdr_contact_probability_scores(
    torch_module,
    distogram_logits,
    binder_length: int,
    *,
    cdr_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    num_target_contacts: int,
    hotspot_indices: tuple[int, ...] = (),
    bin_distances=None,
):
    torch = _torch(torch_module)
    if num_target_contacts <= 0:
        raise ValueError("num_target_contacts must be positive")
    validate_binder_contact_indices(cdr_indices, binder_length)
    target_length = _validate_hotspot_indices(
        distogram_logits,
        binder_length,
        hotspot_indices=hotspot_indices,
    )
    contact_prob = _contact_probability_matrix(
        torch,
        distogram_logits,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        bin_distances=bin_distances,
    )
    binder_rows = _binder_absolute_indices(
        torch,
        binder_length=binder_length,
        target_length=target_length,
        binder_indices=cdr_indices,
        device=distogram_logits.device,
    )
    target_columns = _target_contact_columns(
        torch,
        target_length=target_length,
        hotspot_indices=hotspot_indices,
        device=distogram_logits.device,
    )
    cdr_to_target = contact_prob[:, :, :].index_select(
        dim=1,
        index=binder_rows,
    ).index_select(
        dim=-1,
        index=target_columns,
    )
    k = min(num_target_contacts, int(target_columns.numel()))
    return torch.topk(cdr_to_target, k=k, dim=-1).values.mean(dim=-1)


def compute_framework_contact_penalty_loss(
    torch_module,
    distogram_logits,
    binder_length: int,
    *,
    cdr_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    num_target_contacts: int,
    contact_probability_threshold: float,
    hotspot_indices: tuple[int, ...] = (),
    bin_distances=None,
):
    torch = _torch(torch_module)
    if not 0 < contact_probability_threshold <= 1:
        raise ValueError(
            "contact_probability_threshold must be greater than 0 and at most 1"
        )
    scores = framework_contact_probability_scores(
        torch,
        distogram_logits,
        binder_length,
        cdr_indices=cdr_indices,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        num_target_contacts=num_target_contacts,
        hotspot_indices=hotspot_indices,
        bin_distances=bin_distances,
    )
    if int(scores.shape[-1]) == 0:
        return distogram_logits.new_zeros((distogram_logits.shape[0],))
    excess = torch.relu(scores - contact_probability_threshold)
    return excess.pow(2).mean(dim=-1)


def framework_contact_probability_scores(
    torch_module,
    distogram_logits,
    binder_length: int,
    *,
    cdr_indices: tuple[int, ...],
    contact_cutoff_angstrom: float,
    num_target_contacts: int,
    hotspot_indices: tuple[int, ...] = (),
    bin_distances=None,
):
    torch = _torch(torch_module)
    if num_target_contacts <= 0:
        raise ValueError("num_target_contacts must be positive")
    validate_binder_contact_indices(cdr_indices, binder_length)
    cdr_index_set = set(cdr_indices)
    framework_indices = tuple(
        index for index in range(binder_length) if index not in cdr_index_set
    )
    target_length = _validate_hotspot_indices(
        distogram_logits,
        binder_length,
        hotspot_indices=hotspot_indices,
    )
    if not framework_indices:
        return distogram_logits.new_zeros((distogram_logits.shape[0], 0))
    contact_prob = _contact_probability_matrix(
        torch,
        distogram_logits,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        bin_distances=bin_distances,
    )
    binder_rows = _binder_absolute_indices(
        torch,
        binder_length=binder_length,
        target_length=target_length,
        binder_indices=framework_indices,
        device=distogram_logits.device,
    )
    target_columns = _target_contact_columns(
        torch,
        target_length=target_length,
        hotspot_indices=hotspot_indices,
        device=distogram_logits.device,
    )
    framework_to_target = contact_prob[:, :, :].index_select(
        dim=1,
        index=binder_rows,
    ).index_select(
        dim=-1,
        index=target_columns,
    )
    k = min(num_target_contacts, int(target_columns.numel()))
    return torch.topk(framework_to_target, k=k, dim=-1).values.mean(dim=-1)


def binder_contact_mask(
    torch_module,
    *,
    full_len: int,
    target_length: int,
    binder_length: int,
    binder_contact_indices: tuple[int, ...] | None,
    device,
):
    torch = _torch(torch_module)
    mask = torch.zeros(full_len, device=device)
    if binder_contact_indices is None:
        mask[-binder_length:] = 1.0
        return mask
    validate_binder_contact_indices(binder_contact_indices, binder_length)
    for index in binder_contact_indices:
        mask[target_length + index] = 1.0
    return mask


def _contact_probability_matrix(
    torch_module,
    distogram_logits,
    *,
    contact_cutoff_angstrom: float,
    bin_distances=None,
):
    torch = _torch(torch_module)
    if contact_cutoff_angstrom <= 0:
        raise ValueError("contact_cutoff_angstrom must be positive")
    if bin_distances is None:
        bin_distances = get_mid_points(torch)
    bin_distances = bin_distances.to(
        device=distogram_logits.device,
        dtype=distogram_logits.dtype,
    )
    contact_bins = bin_distances < contact_cutoff_angstrom
    if not bool(contact_bins.any().item()):
        raise ValueError(
            "contact_cutoff_angstrom does not include any ESMFold2 distogram bins"
        )
    probs = torch.softmax(distogram_logits, dim=-1)
    return probs[..., contact_bins].sum(dim=-1)


def _binder_absolute_indices(
    torch_module,
    *,
    binder_length: int,
    target_length: int,
    binder_indices: tuple[int, ...],
    device,
):
    torch = _torch(torch_module)
    validate_binder_contact_indices(binder_indices, binder_length)
    return torch.tensor(
        [target_length + index for index in binder_indices],
        device=device,
        dtype=torch.long,
    )


def _target_contact_columns(
    torch_module,
    *,
    target_length: int,
    hotspot_indices: tuple[int, ...],
    device,
):
    torch = _torch(torch_module)
    if hotspot_indices:
        return torch.tensor(
            list(hotspot_indices),
            device=device,
            dtype=torch.long,
        )
    return torch.arange(target_length, device=device, dtype=torch.long)


def validate_binder_contact_indices(
    binder_contact_indices: tuple[int, ...],
    binder_length: int,
) -> None:
    if not binder_contact_indices:
        raise ValueError("binder_contact_indices cannot be empty")
    for index in binder_contact_indices:
        if index < 0 or index >= binder_length:
            raise ValueError(
                f"binder contact index {index} is outside binder length {binder_length}"
            )


def _validate_hotspot_indices(
    distogram_logits,
    binder_length: int,
    *,
    hotspot_indices: tuple[int, ...],
) -> int:
    target_length = int(distogram_logits.shape[1]) - binder_length
    if target_length <= 0:
        raise ValueError("target length inferred from distogram logits is not positive")
    for index in hotspot_indices:
        if index < 0 or index >= target_length:
            raise ValueError(
                f"hotspot index {index} is outside target length {target_length}"
            )
    return target_length


def _torch(torch_module: Any | None):
    if torch_module is not None:
        return torch_module
    import torch  # type: ignore

    return torch
