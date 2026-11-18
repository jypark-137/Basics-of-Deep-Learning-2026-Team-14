"""Training logs, metric comparison and portable model save/load helpers."""

import json
from pathlib import Path

import torch

from ..model.checkpoint import (
    ANCHOR_STAGE, BASE_STAGE, OFP_STAGE, geometry, restore_anchor_state,
    restore_ofp_state,
)
from ..model.utils import pose_config
from ..model.vit import AttentionPruned, attention_specs, hidden_sizes
from ..paths import DEFAULT_OUTPUT_ROOT, resolve_project_path
from ..utils.io import atomic_torch_save


def better_acc(value, best):
    if best is None:
        return True
    if isinstance(value, dict):
        return sum(value.values()) > sum(best.values())
    return value > best


def get_delta_acc(value, reference):
    if isinstance(value, dict):
        return sum(abs(value[key] - reference[key]) for key in value) / len(value)
    return abs(value - reference)


def log_metrics(directory, event):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, allow_nan=False) + "\n")


def _model_path(args, model_name):
    directory = resolve_project_path(getattr(args, "out_dir", None) or getattr(args, "output_dir", None)
                     or DEFAULT_OUTPUT_ROOT / "training")
    name = Path(model_name or "last.pth")
    return name if name.is_absolute() else directory / name


def save_model(args, model, model_name=None, *, step=None, val_ap=None):
    """Save weights and dimensions together, including variable-width OFP models."""
    info = dict(getattr(args, "_model_info", {}))
    cfg = pose_config(getattr(args, "cfg", None) or getattr(args, "config", None))
    widths = hidden_sizes(model)
    original = info.get("architecture", {}).get("original_hidden_sizes", widths)
    stage = info.get("stage", BASE_STAGE)
    if any(isinstance(block.attn, AttentionPruned) for block in model.backbone.blocks):
        stage = OFP_STAGE
    elif widths != list(original) and stage == BASE_STAGE:
        stage = ANCHOR_STAGE
    # Source weights are replaced; do not retain a second state_dict in memory.
    info.pop("state_dict", None)
    info.update({
        "format_version": 1, "portable_format_version": 1, "stage": stage,
        "geometry": geometry(cfg), "finetuned": True,
        "species": getattr(args, "species", None) or info.get("species"),
        "split": getattr(args, "split_id", None) or info.get("split", 1),
        "finetune_step": step, "finetune_val_ap": val_ap,
        "smoke_test": bool(getattr(args, "smoke_test", False)),
        "architecture": {"original_hidden_sizes": list(original), "hidden_sizes": widths,
                         "attention": attention_specs(model)},
        "state_dict": {name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()},
    })
    path = _model_path(args, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(info, path)
    return path


def load_model(args, model, model_name=None):
    payload = torch.load(str(_model_path(args, model_name)), map_location="cpu")
    stage = payload.get("stage", BASE_STAGE)
    if stage == ANCHOR_STAGE:
        return restore_anchor_state(model, payload)
    if stage == OFP_STAGE:
        return restore_ofp_state(model, payload)
    if stage != BASE_STAGE:
        raise ValueError("Expected a baseline, anchor or OFP model checkpoint")
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    return model
