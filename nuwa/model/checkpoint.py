"""Portable baseline, SKP-anchor and OFP architectures and checkpoint loading."""


from contextlib import contextmanager
from numbers import Integral
import os
from pathlib import Path
import tempfile
import torch
from ..paths import resolve_input_file
from device_utils import resolve_device
from .vit import (
    AttentionPruned, _linear_like, _sizes, attention_specs, forward_heatmaps, hidden_sizes, mlp_layers,
)
from ..utils.io import file_sha256

BASE_STAGE = "ViTPose-base"
ANCHOR_STAGE = "NuWa-SKP-anchor"
OFP_STAGE = "NuWa-OFP"

def saved_binary_masks(payload, model):
    """Check the complete mask plan before changing any model layer."""
    if payload.get("format_version") != 1 or payload.get("stage") != "NuWa-SKP":
        raise ValueError("Expected a SKP NuWa-SKP mask checkpoint (format 1)")
    widths = hidden_sizes(model)
    saved_widths = _sizes(payload.get("hidden_sizes"), "hidden_sizes", len(widths))
    if saved_widths != widths:
        raise ValueError("Saved mask widths do not match this base model")
    counts = _sizes(payload.get("kept_neurons"), "kept_neurons", len(widths))
    masks = payload.get("binary_mlp_masks")
    if not isinstance(masks, (list, tuple)) or len(masks) != len(widths):
        raise ValueError("Checkpoint needs one saved binary_mlp_masks tensor per block")
    checked = []
    for index, (mask, width, count) in enumerate(zip(masks, widths, counts)):
        if not torch.is_tensor(mask) or mask.ndim != 1 or mask.numel() != width:
            raise ValueError("Invalid binary mask shape in block {}".format(index))
        if not torch.isfinite(mask).all().item() or not ((mask == 0) | (mask == 1)).all().item():
            raise ValueError("Mask in block {} must contain only zeros and ones".format(index))
        if int(mask.sum().item()) != count:
            raise ValueError("Saved retained-neuron count differs in block {}".format(index))
        checked.append(mask.detach().cpu().clone())
    return checked


@torch.no_grad()
def prune_skp_mlp(model, payload):
    """Replace dense fc1/fc2 with smaller Linear layers, in place."""
    masks = saved_binary_masks(payload, model)
    plans = []
    for mlp, mask in zip(mlp_layers(model), masks):
        indices = torch.nonzero(mask, as_tuple=False).flatten().to(mlp.fc1.weight.device)
        count = indices.numel()
        if count == mlp.fc1.out_features:
            plans.append((mlp, mlp.fc1, mlp.fc2))
            continue
        fc1 = _linear_like(mlp.fc1, mlp.fc1.in_features, count)
        fc2 = _linear_like(mlp.fc2, count, mlp.fc2.out_features)
        fc1.weight.copy_(mlp.fc1.weight.index_select(0, indices))
        fc2.weight.copy_(mlp.fc2.weight.index_select(1, indices))
        if fc1.bias is not None:
            fc1.bias.copy_(mlp.fc1.bias.index_select(0, indices))
        if fc2.bias is not None:
            fc2.bias.copy_(mlp.fc2.bias)
        plans.append((mlp, fc1, fc2))
    for mlp, fc1, fc2 in plans:
        mlp.fc1, mlp.fc2 = fc1, fc2
    return model


@contextmanager
def apply_saved_masks(model, payload):
    """Temporarily reproduce saved gates on the full model, without an STE.

    This uses the saved selection directly rather than sorting scores again
    on a different device. Hooks are removed even when inference raises.
    """
    masks = saved_binary_masks(payload, model)
    handles = []
    try:
        for mlp, mask in zip(mlp_layers(model), masks):
            def gate(module, inputs, binary=mask):
                value = inputs[0]
                return (value * binary.to(device=value.device, dtype=value.dtype),) + inputs[1:]
            handles.append(mlp.fc2.register_forward_pre_hook(gate))
        yield model
    finally:
        for handle in handles:
            handle.remove()


def anchor_architecture(payload, model):
    if payload.get("format_version") != 1 or payload.get("stage") != ANCHOR_STAGE:
        raise ValueError("Expected an anchor.pth exported by anchor export")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("Anchor checkpoint is missing its architecture")
    original = hidden_sizes(model)
    expected = _sizes(architecture.get("original_hidden_sizes"),
                      "original_hidden_sizes", len(original))
    target = _sizes(architecture.get("hidden_sizes"), "hidden_sizes", len(original))
    if expected != original or any(new > old for new, old in zip(target, original)):
        raise ValueError("Anchor architecture does not match the supplied ViTPose config")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Anchor checkpoint is missing its model weights")
    return target


def restore_anchor_state(model, payload):
    """Resize a freshly constructed model, then strictly load all anchor weights."""
    target = anchor_architecture(payload, model)
    for mlp, count in zip(mlp_layers(model), target):
        if mlp.fc1.out_features != count:
            mlp.fc1 = _linear_like(mlp.fc1, mlp.fc1.in_features, count)
            mlp.fc2 = _linear_like(mlp.fc2, count, mlp.fc2.out_features)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model


def load_anchor_model(cfg, checkpoint_path, device="auto"):
    """Load without the original base checkpoint or the SKP mask file.

    A normal unmodified ViTPose loader cannot infer variable MLP widths from
    a state_dict. Use this loader or engine/eval.py for anchor.pth.
    """
    from mmpose.models import build_posenet

    payload = torch.load(str(resolve_input_file(checkpoint_path, "checkpoint")), map_location="cpu")
    model = restore_anchor_state(build_posenet(cfg.model), payload)
    return model.to(resolve_device(device)).eval(), payload


def restore_ofp_state(model, payload):
    if payload.get("stage") != OFP_STAGE:
        raise ValueError("Expected a NuWa-OFP checkpoint")
    anchor_view = dict(payload, stage=ANCHOR_STAGE)
    widths = anchor_architecture(anchor_view, model)
    specs = payload["architecture"].get("attention")
    if not isinstance(specs, list) or len(specs) != len(widths):
        raise ValueError("OFP checkpoint needs one attention specification per block")
    for block, mlp, width, spec in zip(model.backbone.blocks, mlp_layers(model), widths, specs):
        old = block.attn
        keys = {"num_heads", "qk_dim", "vo_dim", "qkv_bias", "proj_bias", "attn_drop", "proj_drop"}
        if not isinstance(spec, dict) or set(spec) not in (keys, keys | {"kind"}):
            raise ValueError("Invalid OFP attention specification")
        kind = spec.get("kind", "pruned")  # Older OFP checkpoints omitted this field.
        if kind not in ("dense", "pruned"):
            raise ValueError("Unknown attention kind")
        head_dim = old.qkv.out_features // (3 * old.num_heads)
        if spec["num_heads"] != old.num_heads or any(
            isinstance(spec[key], bool) or not isinstance(spec[key], Integral) or not 1 <= spec[key] <= head_dim
            for key in ("qk_dim", "vo_dim")
        ):
            raise ValueError("OFP attention dimensions disagree with the ViTPose config")
        if spec["qkv_bias"] != (old.qkv.bias is not None) or spec["proj_bias"] != (old.proj.bias is not None):
            raise ValueError("OFP bias settings disagree with the ViTPose config")
        mlp.fc1 = _linear_like(mlp.fc1, mlp.fc1.in_features, width)
        mlp.fc2 = _linear_like(mlp.fc2, width, mlp.fc2.out_features)
        if kind == "dense":
            if spec["qk_dim"] != head_dim or spec["vo_dim"] != head_dim:
                raise ValueError("Dense attention must preserve its original dimensions")
        else:
            block.attn = AttentionPruned(old.qkv.in_features, **{key: spec[key] for key in keys}).to(
                device=old.qkv.weight.device, dtype=old.qkv.weight.dtype)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model


def load_nuwa_model(cfg, checkpoint_path, device="auto"):
    """Reconstruct either an SKP anchor or an OFP checkpoint."""
    from mmpose.models import build_posenet

    payload = torch.load(str(resolve_input_file(checkpoint_path, "checkpoint")), map_location="cpu")
    model = build_posenet(cfg.model)
    if payload.get("stage") == ANCHOR_STAGE:
        model = restore_anchor_state(model, payload)
    elif payload.get("stage") == OFP_STAGE:
        model = restore_ofp_state(model, payload)
    else:
        raise ValueError("Expected a NuWa SKP anchor or OFP checkpoint")
    return model.to(resolve_device(device)).eval(), payload


def geometry(cfg):
    return {"image_size": list(map(int, cfg.data_cfg.image_size)),
            "heatmap_size": list(map(int, cfg.data_cfg.heatmap_size)),
            "num_joints": int(cfg.data_cfg.num_joints)}


def load_pose_checkpoint(cfg, checkpoint_path, device="auto"):
    from mmpose.models import build_posenet

    payload = torch.load(str(resolve_input_file(checkpoint_path, "checkpoint")), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Expected a pose-model state_dict checkpoint")
    if payload.get("stage") == "NuWa-SKP" or "binary_mlp_masks" in payload:
        raise ValueError("This is a mask-only file; export anchor.pth with get_anchor_model.py first")
    if payload.get("geometry") is not None and payload["geometry"] != geometry(cfg):
        raise ValueError("Checkpoint image/keypoint dimensions disagree with the config")
    model = build_posenet(cfg.model)
    stage = payload.get("stage", BASE_STAGE)
    if stage == ANCHOR_STAGE:
        model = restore_anchor_state(model, payload)
        info = payload
    elif stage == OFP_STAGE:
        model = restore_ofp_state(model, payload)
        info = payload
    elif stage == BASE_STAGE:
        model.load_state_dict(payload.get("state_dict", payload), strict=True)
        info = dict(payload, stage=BASE_STAGE) if "state_dict" in payload else {"stage": BASE_STAGE}
    else:
        raise ValueError("Unsupported checkpoint stage: {}".format(stage))
    return model.to(resolve_device(device)).eval(), info


def checkpoint_header(path):
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Expected a state_dict checkpoint")
    keys = ("stage", "species", "split", "base_checkpoint_sha256", "source_anchor_sha256",
            "source_checkpoint_sha256", "origin_checkpoint_sha256", "category_id", "category_ids",
            "smoke_test", "source_smoke_test", "skp_applied", "attention_svd_applied", "mlp_strategy",
            "finetuned", "finetune_step", "finetune_val_ap")
    result = {key: payload[key] for key in keys if key in payload}
    result.setdefault("stage", BASE_STAGE)
    return result


@torch.no_grad()
def export_portable(model, info, cfg, source_path, output_path, base_sha256=None):
    """Export model weights only; strip optimizers and training logs consistently."""
    retained = (
        "species", "category_id", "category_ids", "split", "base_checkpoint_sha256", "source_anchor_sha256",
        "source_masks_sha256", "source_mask_step", "source_mask_val_ap", "baseline_val_ap",
        "target_pruning_rate", "achieved_pruning_rate", "smoke_test", "source_smoke_test",
        "skp_applied", "attention_svd_applied", "mlp_strategy",
        "finetuned", "finetune_step", "finetune_val_ap",
    )
    stage = info.get("stage", BASE_STAGE)
    current = hidden_sizes(model)
    payload = {key: info[key] for key in retained if key in info}
    payload.update({"format_version": 1, "portable_format_version": 1, "stage": stage,
                    "geometry": geometry(cfg), "source_checkpoint_sha256": file_sha256(source_path),
                    "architecture": {
                        "original_hidden_sizes": info.get("architecture", {}).get("original_hidden_sizes", current),
                        "hidden_sizes": current, "attention": attention_specs(model)},
                    "state_dict": {name: value.detach().cpu().clone()
                                   for name, value in model.state_dict().items()}})
    payload["origin_checkpoint_sha256"] = info.get("origin_checkpoint_sha256", payload["source_checkpoint_sha256"])
    if base_sha256 is not None:
        payload["base_checkpoint_sha256"] = base_sha256
    elif stage == BASE_STAGE and "base_checkpoint_sha256" not in payload:
        payload["base_checkpoint_sha256"] = payload["source_checkpoint_sha256"]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".pth", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, str(temporary))
        os.replace(str(temporary), str(path))
    finally:
        temporary.unlink(missing_ok=True)
    return payload


@torch.no_grad()
def verify_reload(model, cfg, checkpoint_path, images):
    expected = forward_heatmaps(model, images)
    expected_shape = (images.shape[0], int(cfg.data_cfg.num_joints),
                      int(cfg.data_cfg.heatmap_size[1]), int(cfg.data_cfg.heatmap_size[0]))
    if tuple(expected.shape) != expected_shape or not torch.isfinite(expected).all().item():
        raise RuntimeError("Invalid pose heatmaps before checkpoint reload")
    restored, _ = load_pose_checkpoint(cfg, checkpoint_path, images.device)
    actual = forward_heatmaps(restored, images)
    if not torch.isfinite(actual).all().item() or not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
        raise RuntimeError("Portable checkpoint reload changed the heatmaps")
    return {"passed": True, "max_abs_error": float((actual - expected).abs().max().item()),
            "heatmap_shape": list(expected.shape), "rtol": 1e-4, "atol": 1e-5}
