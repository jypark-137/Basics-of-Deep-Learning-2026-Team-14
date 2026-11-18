"""Mask checkpoint helpers and the original NuWa normalization utilities."""


import math
from pathlib import Path
import torch
from ..paths import DEFAULT_OUTPUT_ROOT
from ..utils.io import atomic_torch_save, file_sha256, frozen_state_sha256
from ..species import normalize_species

@torch.no_grad()
def export_masks(model):
    from .get_anchor_model import TopKBinarizer, get_skp_parameters
    masks, betas = get_skp_parameters(model)
    scores = [parameter.detach().cpu().clone() for parameter in masks]
    controls = [parameter.detach().cpu().clone() for parameter in betas]
    # Binarize on the model's device, preserving its actual sorting of ties.
    binary = [TopKBinarizer.apply(mask, beta).detach().cpu().clone()
              for mask, beta in zip(masks, betas)]
    return {
        "mlp_masks": scores,
        "mlp_betas": controls,
        "binary_mlp_masks": binary,
        "kept_neurons": [int(mask.sum().item()) for mask in binary],
        "hidden_sizes": [int(mask.numel()) for mask in binary],
    }


@torch.no_grad()
def load_skp_masks(model, payload):
    """Load scores/betas into an already masked model; useful for later steps.

    Anchor export should use the saved binary_mlp_masks as authoritative for neuron
    indices: sorting tied scores can differ across devices.
    """
    from .get_anchor_model import get_skp_parameters
    masks, betas = get_skp_parameters(model)
    saved_masks, saved_betas = payload["mlp_masks"], payload["mlp_betas"]
    if len(masks) != len(saved_masks) or len(betas) != len(saved_betas):
        raise ValueError("Saved masks have a different transformer depth")
    for target, source in zip(masks + betas, list(saved_masks) + list(saved_betas)):
        if target.shape != source.shape or not torch.isfinite(source).all().item():
            raise ValueError("Saved mask/beta has an invalid shape or value")
    for target, source in zip(masks + betas, list(saved_masks) + list(saved_betas)):
        target.copy_(source.to(device=target.device, dtype=target.dtype))


def remaining_neurons(mask_info):
    return sum(mask_info["kept_neurons"]), sum(mask_info["hidden_sizes"])


def save_masks(model, path, metadata, step, val_ap, baseline_ap):
    mask_info = export_masks(model)
    payload = dict(metadata)
    payload.update(mask_info)
    payload.update({"format_version": 1, "stage": "NuWa-SKP",
                    "step": step, "val_ap": val_ap, "baseline_val_ap": baseline_ap})
    atomic_torch_save(payload, path)
    return mask_info


def better_acc(acc_1,acc_2):
    if isinstance(acc_1,dict):
        acc1 = 0.0
        for key in acc_1.keys():
            acc1 += acc_1[key]
        acc2 = 0.0
        for key in acc_2.keys():
            acc2 += acc_2[key]
        return acc1>=acc2
    else:
        return acc_1>=acc_2


def l2_norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def min_max_norm(x):
    min = x.min()
    max = x.max()
    return (x-min)/(max-min)


def z_score_norm(x):
    mean = x.mean()
    std = x.std(unbiased=False)
    return (x-mean)/std


def softmax_norm(x):
    T = x.abs().max().clamp(min=1e-6)
    return torch.softmax(x/T,dim=-1)


def normalization(x,norm_type="l2"):
    if norm_type == "l2":
        return l2_norm(x)
    elif norm_type == "min_max":
        return min_max_norm(x)
    elif norm_type == "z_score":
        return z_score_norm(x)
    elif norm_type == "softmax":
        return softmax_norm(x)
    else:
        raise ValueError(f"Unknown norm type {norm_type}")


def pruning2energy(pruning_rate):
    a = pruning_rate
    energy_rate = -0.41*math.pow(a,3)+0.14*math.pow(a,2)-0.03*a+1.0
    return energy_rate


def save_mask(args, model_mask, pruning_rate=None):
    """Save the same mask-only format used by anchor learning."""
    from .get_anchor_model import _options
    args = _options(args)
    directory = Path(args.out_dir or DEFAULT_OUTPUT_ROOT / "masks")
    directory.mkdir(parents=True, exist_ok=True)
    original_keys = [key for key in model_mask.state_dict() if not key.endswith((".mask", ".beta"))]
    metadata = {"species": normalize_species(args.species), "split": args.split,
                "base_checkpoint_sha256": file_sha256(args.checkpoint),
                "original_state_sha256": frozen_state_sha256(model_mask, original_keys)}
    if pruning_rate is not None:
        metadata["rate"] = float(pruning_rate)
    path = directory / "last_masks.pth"
    save_masks(model_mask, path, metadata, getattr(args, "global_step", 0),
               getattr(args, "val_ap", None), getattr(args, "baseline_val_ap", None))
    args.masks = str(path)
    return path


def load_mask(args):
    from .get_anchor_model import _options, _saved_masks_path
    args = _options(args)
    path = _saved_masks_path(args)
    if path is None and args.out_dir:
        candidate = Path(args.out_dir) / "last_masks.pth"
        path = candidate if candidate.is_file() else None
    if path is None:
        raise FileNotFoundError("Select an existing mask checkpoint with --masks")
    return torch.load(str(path), map_location="cpu")
