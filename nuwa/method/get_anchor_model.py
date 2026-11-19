"""NuWa SKP mask learning and physical ViTPose anchor construction."""


import argparse
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
import numpy as np
import torch
from torch import nn
from ..paths import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPECIES, resolve_data_root, resolve_input_file, resolve_project_path,
)
from ..model.vit import get_nuwa_backbone, forward_heatmaps, heatmap_task_loss, hidden_sizes, mlp_layers
from ..model.checkpoint import ANCHOR_STAGE, apply_saved_masks, checkpoint_header, load_anchor_model, prune_skp_mlp, saved_binary_masks
from ..species import add_species_argument, normalize_species, species_slug
from device_utils import add_device_argument, resolve_device
from ..dataset.utils import make_loaders, move_batch, validate_subsets
from ..engine.eval import evaluate_ap
from ..engine.scheduler import (
    SCHEDULER_DEFAULTS, add_scheduler_arguments, make_scheduler, scheduler_settings,
)
from ..utils.io import atomic_torch_save, file_sha256, frozen_state_sha256
from .skp import MaskLinear, TopKBinarizer, get_skp_parameters
from ..utils.console import CommandParser, command_output, detail, display_path
from .utils import export_masks, load_skp_masks, remaining_neurons, save_masks


def attach_skp_masks(model):
    """Freeze the complete pose model and insert only NuWa's MLP gates."""
    backbone = get_nuwa_backbone(model)
    if any(isinstance(block.mlp.fc2, MaskLinear) for block in backbone.blocks):
        raise ValueError("SKP masks are already attached")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in backbone.blocks:
        fc2 = block.mlp.fc2
        mask = nn.Parameter(torch.ones(fc2.in_features, device=fc2.weight.device))
        beta = nn.Parameter(torch.zeros(1, device=fc2.weight.device) + 5.0)
        block.mlp.fc2 = MaskLinear(fc2.weight, fc2.bias, mask, beta, mask_type="mlp")
    return get_skp_parameters(model)


def set_skp_train_mode(model):
    model.train()
    # Frozen parameters alone do not freeze BatchNorm's running statistics.
    model.keypoint_head.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def check_trainable_parameters(model):
    masks, betas = get_skp_parameters(model)
    expected = {id(parameter) for parameter in masks + betas}
    actual = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if actual != expected:
        raise RuntimeError("Trainable parameters must be exactly the MLP masks and betas")
    return sum(parameter.numel() for parameter in masks + betas)


def check_mask_gradients(model):
    masks, betas = get_skp_parameters(model)
    for name, parameters in (("mask", masks), ("beta", betas)):
        for index, parameter in enumerate(parameters):
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError("Invalid or missing {} gradient in block {}".format(name, index))
    if sum(parameter.grad.abs().sum().item() for parameter in masks) == 0:
        raise RuntimeError("All mask gradients are zero")
    trainable_ids = {id(parameter) for parameter in masks + betas}
    if any(parameter.grad is not None for parameter in model.parameters()
           if id(parameter) not in trainable_ids):
        raise RuntimeError("An original model parameter received a gradient")


def self_test():
    """Exercise the original STE and mask save/reload using PyTorch only."""
    scores = torch.tensor([0.1, 0.9, -0.2, 0.5], requires_grad=True)
    beta = torch.tensor([0.0], requires_grad=True)
    gradient = torch.tensor([1.0, 2.0, 3.0, 4.0])
    binary = TopKBinarizer.apply(scores, beta)
    if not torch.equal(binary, torch.tensor([0.0, 1.0, 0.0, 1.0])):
        raise RuntimeError("Vanilla top-k selection check failed")
    (binary * gradient).sum().backward()
    if not torch.equal(scores.grad, gradient) or not torch.equal(beta.grad, torch.tensor([6.0])):
        raise RuntimeError("Vanilla STE gradient check failed")
    for control, expected in ((-20.0, 1), (20.0, 1536), (5.0, 1526)):
        mask = TopKBinarizer.apply(torch.ones(1536), torch.tensor([control]))
        if int(mask.sum().item()) != expected:
            raise RuntimeError("Vanilla retained-neuron count check failed")

    fc2 = nn.Linear(4, 3)
    fc2.requires_grad_(False)
    mask = nn.Parameter(torch.tensor([0.1, 0.9, -0.2, 0.5]))
    beta = nn.Parameter(torch.tensor([0.0]))
    layer = MaskLinear(fc2.weight, fc2.bias, mask, beta)
    inputs = torch.randn(2, 5, 4)
    actual = layer(inputs)
    expected = nn.functional.linear(inputs * torch.tensor([0.0, 1.0, 0.0, 1.0]),
                                    fc2.weight, fc2.bias)
    if not torch.equal(actual, expected):
        raise RuntimeError("Mask placement before fc2 is incorrect")
    actual.square().mean().backward()
    if mask.grad is None or beta.grad is None or fc2.weight.grad is not None:
        raise RuntimeError("MaskLinear gradient isolation check failed")

    # Minimal real modules test the same attachment and checkpoint helpers.
    def make_model():
        model = nn.Module()
        model.backbone = nn.Module()
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.fc1, block.mlp.fc2 = nn.Linear(4, 8), nn.Linear(8, 4)
        block.attn = nn.Module()
        block.attn.num_heads = 2
        block.attn.qkv, block.attn.proj = nn.Linear(4, 12), nn.Linear(4, 4)
        model.backbone.blocks = nn.ModuleList([block])
        model.keypoint_head = nn.BatchNorm2d(4)
        return model
    model = make_model()
    original_keys = list(model.state_dict())
    initial = frozen_state_sha256(model, original_keys)
    attach_skp_masks(model)
    set_skp_train_mode(model)
    if model.keypoint_head.training or check_trainable_parameters(model) != 9:
        raise RuntimeError("Frozen head or trainable-parameter check failed")
    payload = export_masks(model)
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    restored = make_model()
    attach_skp_masks(restored)
    load_skp_masks(restored, torch.load(stream, map_location="cpu"))
    other = export_masks(restored)
    for field in ("mlp_masks", "mlp_betas", "binary_mlp_masks"):
        if not all(torch.equal(a, b) for a, b in zip(payload[field], other[field])):
            raise RuntimeError("Mask checkpoint round trip failed")
    if frozen_state_sha256(model, original_keys) != initial:
        raise RuntimeError("Original weights/buffers changed while attaching masks")
    print("Vanilla mask rule, STE and checkpoint checks: PASS", flush=True)


def compare_heatmaps(actual, expected, label):
    if actual.shape != expected.shape or not torch.isfinite(actual).all().item():
        raise RuntimeError("{} produced invalid heatmaps".format(label))
    error = float((actual - expected).abs().max().item())
    if not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
        raise RuntimeError("{} differs from the saved masked model; max error {}".format(label, error))
    return error


def _options(args, train_loader=None):
    """Fill shared defaults, including NuWa's original training option names."""
    if not hasattr(args, "steps"):
        steps = getattr(args, "num_steps", 10000)
        if steps <= 0 and train_loader is not None:
            steps = getattr(args, "num_epochs", 1) * len(train_loader)
        args.steps = steps
    defaults = {
        "batch_size": getattr(args, "train_batch_size", 1),
        "out_dir": getattr(args, "output_dir", None),
        "config": DEFAULT_CONFIG, "checkpoint": DEFAULT_CHECKPOINT,
        "data_root": DEFAULT_DATA_ROOT, "species": DEFAULT_SPECIES, "split": 1,
        "device": "auto", "mask_lr": 1e-3, "beta_lr": 1e-1,
        "weight_decay": 5e-2, "eval_every": 1000, "log_every": 10,
        "seed": 42, "smoke_test": False, "masks": None, "uniform": False,
        **SCHEDULER_DEFAULTS,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    if args.smoke_test:
        args.steps = 10
    if min(args.steps, args.batch_size, args.eval_every, args.log_every) <= 0:
        raise ValueError("steps, batch-size, eval-every and log-every must be positive")
    if (not all(math.isfinite(value) for value in
                (args.mask_lr, args.beta_lr, args.weight_decay))
            or min(args.mask_lr, args.beta_lr) <= 0 or args.weight_decay < 0):
        raise ValueError("Learning rates must be positive and weight decay nonnegative")
    scheduler_settings(args, args.steps)
    args.device = str(resolve_device(args.device))
    args.species = normalize_species(args.species or DEFAULT_SPECIES)
    args.data_root = str(resolve_data_root(args.data_root))
    if args.out_dir:
        args.out_dir = str(resolve_project_path(args.out_dir))
    return args


def _saved_masks_path(args):
    if args.masks:
        return resolve_input_file(args.masks, "--masks")
    if args.out_dir:
        name = "last_masks.pth" if args.smoke_test else "best_masks.pth"
        candidate = Path(args.out_dir) / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _check_training_output(output_dir):
    """Permit a fresh run or a retry that wrote only its configuration."""
    if output_dir.exists() and (not output_dir.is_dir() or any(
            path.name != "run_config.json" or not path.is_file()
            for path in output_dir.iterdir())):
        raise FileExistsError("Mask training output already contains results; choose another --out-dir")


def _check_completed_training_settings(args, masks):
    """An implicit rerun must identify the same mask-training experiment."""
    saved = masks.get("training_settings", {})
    names = ("species", "split", "config", "data_root", "steps", "batch_size",
             "mask_lr", "beta_lr", "weight_decay", "eval_every", "seed",
             "smoke_test", "uniform", *SCHEDULER_DEFAULTS)
    changed = [name for name in names
               if saved.get(name, SCHEDULER_DEFAULTS.get(name)) != getattr(args, name)]
    if changed:
        raise ValueError("Existing anchor training settings differ ({}); choose another --out-dir "
                         "or select the saved masks explicitly with --masks".format(
                             ", ".join(changed)))


def detach_skp_masks(model):
    """Restore dense fc2 layers before applying the selected saved masks.

    Training has frozen the base weights, so reusing those same parameters
    removes the learned gates without reloading or copying a second model.
    """
    for block in model.backbone.blocks:
        masked = block.mlp.fc2
        if isinstance(masked, MaskLinear):
            dense = nn.Linear(masked.in_features, masked.out_features, bias=False)
            dense.weight, dense.bias = masked.weight, masked.bias
            dense.train(masked.training)
            block.mlp.fc2 = dense
    return model


def train_mask(args, model, train_loader, test_loader, uniform=False):
    """Learn per-block MLP masks with the existing weighted pose task loss."""
    args = _options(args, train_loader)
    if uniform or args.uniform:
        raise ValueError("This ViTPose workflow uses per-block betas, not uniform SKP")
    if train_loader is None or len(train_loader) == 0:
        raise ValueError("Mask training requires a nonempty training loader")
    val_loader = None if args.smoke_test else test_loader
    if not args.smoke_test and (val_loader is None or len(val_loader) == 0):
        raise ValueError("Full mask training requires a nonempty validation loader")
    device = torch.device(args.device)
    model.to(device).eval()
    get_nuwa_backbone(model)
    if getattr(model.backbone, "use_checkpoint", False):
        raise ValueError("Set backbone.use_checkpoint=False in the config for SKP")
    config_path = resolve_input_file(args.config, "--config")
    checkpoint_path = resolve_input_file(args.checkpoint, "--checkpoint")
    data_root = Path(args.data_root)
    slug = species_slug(args.species)
    category_id = getattr(args, "category_id", None)
    original_keys = list(model.state_dict())
    original_digest = frozen_state_sha256(model, original_keys)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S") + ("_smoke" if args.smoke_test else "")
    output_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUTPUT_ROOT / "nuwa_skp" / slug / run_name
    _check_training_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(output_dir)
    settings = {name: str(value) if isinstance(value, (Path, torch.device)) else value
                for name, value in vars(args).items()}
    settings.update({"device": str(device), "config": str(config_path),
                     "checkpoint": str(checkpoint_path), "data_root": str(data_root),
                     "train_annotations": str(getattr(train_loader.dataset, "ann_file", "")),
                     "val_annotations": str(getattr(getattr(val_loader, "dataset", None),
                                                    "ann_file", ""))})
    schedule = scheduler_settings(args, args.steps)
    settings["lr_schedule"] = schedule
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)
        handle.write("\n")
    print("Learning masks: {} | {} | {} updates{}".format(
        ", ".join(args.species), device, args.steps, " | smoke test" if args.smoke_test else ""), flush=True)
    detail("Output:", output_dir, flush=True)

    baseline_ap = None
    if val_loader is not None:
        print("Measuring the unpruned species validation baseline...", flush=True)
        baseline_ap = evaluate_ap(model, val_loader, device, output_dir / "eval" / "unpruned")
    masks, betas = attach_skp_masks(model)
    count = check_trainable_parameters(model)
    detail("Trainable mask/beta parameters:", count, flush=True)
    optimizer = torch.optim.AdamW([
        {"params": masks, "lr": args.mask_lr, "weight_decay": args.weight_decay},
        {"params": betas, "lr": args.beta_lr, "weight_decay": args.weight_decay},
    ], betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    scheduler = make_scheduler(optimizer, schedule)
    metadata = {"species": args.species, "category_id": category_id,
                "category_ids": getattr(args, "category_ids", None), "split": args.split,
                "base_checkpoint": str(checkpoint_path), "base_checkpoint_sha256": file_sha256(checkpoint_path),
                "config_path": str(config_path), "training_settings": settings,
                "original_state_sha256": original_digest}
    initial_parameters = [parameter.detach().cpu().clone() for parameter in masks + betas]
    best_ap, best_kept, best_step = None, None, 0
    last_ap, last_eval_step, step = None, None, 0
    if val_loader is not None:
        print("Measuring the initial vanilla-masked model...", flush=True)
        best_ap = evaluate_ap(model, val_loader, device, output_dir / "eval" / "step_000000")
        initial_info = save_masks(model, output_dir / "best_masks.pth", metadata, 0, best_ap, baseline_ap)
        best_kept, _ = remaining_neurons(initial_info)
        last_ap, last_eval_step = best_ap, 0
    initial_info = export_masks(model)
    kept, total = remaining_neurons(initial_info)
    detail("Initially masked MLP hidden neurons: {} / {} ({:.2f}%)".format(
        total - kept, total, 100 * (1 - kept / total)), flush=True)
    interrupted = False
    start = time.perf_counter()
    iterator = iter(train_loader)
    set_skp_train_mode(model)

    def record(event):
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    record({"event": "initial", "baseline_val_ap": baseline_ap, "masked_val_ap": best_ap,
            "kept_neurons": initial_info["kept_neurons"], "lr_schedule": schedule,
            "mask_lr": optimizer.param_groups[0]["lr"],
            "beta_lr": optimizer.param_groups[1]["lr"]})
    try:
        while step < args.steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            data = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            heatmaps = forward_heatmaps(model, data["img"])
            loss = heatmap_task_loss(model, heatmaps, data["target"], data["target_weight"])
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite pose loss at update {}".format(step + 1))
            loss.backward()
            # Stop on the first bad gradient, before it can corrupt saved masks.
            check_mask_gradients(model)
            mask_lr, beta_lr = [group["lr"] for group in optimizer.param_groups]
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            if any(not torch.isfinite(parameter).all().item() for parameter in masks + betas):
                raise RuntimeError("Nonfinite mask/beta after update {}".format(step))

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                info = export_masks(model)
                kept, total = remaining_neurons(info)
                loss_value = float(loss.detach().item())
                print("Step {}/{} | loss {:.6f} | MLP neurons masked {:.2f}% | "
                      "mask LR {:.2e} | beta LR {:.2e} | {:.1f}s/update".format(
                    step, args.steps, loss_value, 100 * (1 - kept / total),
                    mask_lr, beta_lr,
                    (time.perf_counter() - start) / step), flush=True)
                record({"event": "train", "step": step, "loss": loss_value,
                        "kept_neurons": info["kept_neurons"],
                        "mask_lr": mask_lr, "beta_lr": beta_lr})

            if val_loader is not None and (step % args.eval_every == 0 or step == args.steps):
                last_ap = evaluate_ap(model, val_loader, device,
                                      output_dir / "eval" / "step_{:06d}".format(step))
                last_eval_step = step
                info = export_masks(model)
                kept, total = remaining_neurons(info)
                if last_ap > best_ap or (last_ap == best_ap and kept < best_kept):
                    best_ap, best_kept, best_step = last_ap, kept, step
                    save_masks(model, output_dir / "best_masks.pth", metadata, step, last_ap, baseline_ap)
                save_masks(model, output_dir / "last_masks.pth", metadata, step, last_ap, baseline_ap)
                record({"event": "validation", "step": step, "val_ap": last_ap,
                        "best_val_ap": best_ap, "best_step": best_step})
                set_skp_train_mode(model)
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped; saving the current masks.", flush=True)
    finally:
        # The last state's AP is unknown if it was not evaluated at this step.
        current_ap = last_ap if last_eval_step == step else None
        save_masks(model, output_dir / "last_masks.pth", metadata, step, current_ap, baseline_ap)

    if frozen_state_sha256(model, original_keys) != original_digest:
        raise RuntimeError("Original weights or buffers changed; do not use this run's masks")
    if not any(not torch.equal(old, new.detach().cpu())
               for old, new in zip(initial_parameters, masks + betas)) and step > 0:
        raise RuntimeError("No mask or beta parameter changed")
    detail("Original weights and buffers unchanged: PASS", flush=True)
    if step > 0:
        detail("Mask/beta updates: PASS", flush=True)
    if args.smoke_test:
        detail("{} after {} updates".format("Smoke test stopped" if interrupted else "SKP smoke test passed", step), flush=True)
    elif best_ap is not None:
        print("Best masked validation AP: {:.2f} at step {} | unpruned: {:.2f}".format(
            100 * best_ap, best_step, 100 * baseline_ap), flush=True)
    detail("Masks saved in:", output_dir, flush=True)
    if interrupted:
        raise KeyboardInterrupt
    args.masks = str(output_dir / ("last_masks.pth" if args.smoke_test else "best_masks.pth"))
    return best_ap


def get_mask(args, model, train_loader=None, test_loader=None):
    """Reuse an existing mask checkpoint, or train and load the best one."""
    args = _options(args, train_loader)
    mlp_layers(model)
    path = _saved_masks_path(args)
    if path is None:
        try:
            train_mask(args, model, train_loader, test_loader, args.uniform)
        finally:
            # The current/last gates need not be the validation-best gates.
            detach_skp_masks(model)
        path = resolve_input_file(args.masks, "--masks")
    payload = torch.load(str(path), map_location="cpu")
    if normalize_species(payload.get("species")) != args.species:
        raise ValueError("--species must match the species used to learn these masks")
    saved_binary_masks(payload, model)
    base_path = resolve_input_file(args.checkpoint, "--checkpoint")
    if payload.get("base_checkpoint_sha256") != file_sha256(base_path):
        raise ValueError("The base checkpoint differs from the one used for these masks")
    if payload.get("original_state_sha256") != frozen_state_sha256(model, list(model.state_dict())):
        raise ValueError("The base model state differs from the model used for SKP")
    args.masks = str(path)
    return payload


def mask2anchor(args, model, mask_dict):
    """Slice fc1 rows/bias and fc2 columns using the saved binary selection."""
    return prune_skp_mlp(model, mask_dict)


def head_focus(model, sub_label=None):
    """Species focus comes from the data subset; every keypoint stays active."""
    if not hasattr(model, "keypoint_head"):
        raise ValueError("Expected a ViTPose keypoint head")
    return model


def get_anchor_model(args, model, train_loader=None, test_loader=None, focus=True):
    mask_dict = get_mask(args, model, train_loader, test_loader)
    kept, total = remaining_neurons(mask_dict)
    detail("MLP hidden neurons masked: {:.2f}%".format(100 * (1 - kept / total)))
    model_anchor = mask2anchor(args, model, mask_dict)
    if focus:
        model_anchor = head_focus(model_anchor, getattr(args, "sub_label", None))
    return model_anchor.eval()


@torch.no_grad()
def export_anchor(args, model, masks):
    """Verify and save the portable anchor, or reuse a matching completed one."""
    from mmcv import Config

    args = _options(args)
    config_path = resolve_input_file(args.config, "--config")
    base_path = resolve_input_file(args.checkpoint, "--checkpoint")
    masks_path = resolve_input_file(args.masks, "--masks")
    output_dir = Path(args.out_dir) if args.out_dir else masks_path.parent
    output_path, report_path = output_dir / "anchor.pth", output_dir / "anchor_report.json"
    reuse_existing = output_path.exists() or report_path.exists()
    if reuse_existing and not (output_path.is_file() and report_path.is_file()):
        raise FileExistsError("Anchor export is incomplete; choose another --out-dir")
    cfg = Config.fromfile(str(config_path))
    base_hash = file_sha256(base_path)
    if masks.get("base_checkpoint_sha256") != base_hash:
        raise ValueError("The base checkpoint differs from the one used for these masks")
    if frozen_state_sha256(model, list(model.state_dict())) != masks.get("original_state_sha256"):
        raise ValueError("The base model state differs from the model used for SKP")
    device = torch.device(args.device)
    model.to(device).eval()
    get_nuwa_backbone(model)
    saved_binary_masks(masks, model)
    original_widths = hidden_sizes(model)
    original_count = sum(parameter.numel() for parameter in model.parameters())
    protected_keys = [name for name in model.state_dict()
                      if not any(name == "backbone.blocks.{}.mlp.{}".format(index, suffix)
                                 for index in range(len(original_widths))
                                 for suffix in ("fc1.weight", "fc1.bias", "fc2.weight"))]
    protected_hash = frozen_state_sha256(model, protected_keys)
    generator = torch.Generator().manual_seed(args.seed)
    width, height = map(int, cfg.data_cfg.image_size)
    inputs = [torch.randn(1, 3, height, width, generator=generator).to(device)
              for _ in range(2)]
    with apply_saved_masks(model, masks):
        reference = [forward_heatmaps(model, images).detach().clone() for images in inputs]
    expected_shape = (1, int(cfg.data_cfg.num_joints),
                      int(cfg.data_cfg.heatmap_size[1]), int(cfg.data_cfg.heatmap_size[0]))
    if any(tuple(value.shape) != expected_shape or not torch.isfinite(value).all().item()
           for value in reference):
        raise RuntimeError("The saved masked model produced invalid heatmaps")

    mask2anchor(args, model, masks)
    if frozen_state_sha256(model, protected_keys) != protected_hash:
        raise RuntimeError("A protected attention/head/residual weight or buffer changed")
    errors = [compare_heatmaps(forward_heatmaps(model, images), expected, "Pruned model")
              for images, expected in zip(inputs, reference)]
    reduced_count = sum(parameter.numel() for parameter in model.parameters())
    target_widths = hidden_sizes(model)
    expected_removed = sum((old - new) * (mlp.fc1.in_features + mlp.fc2.out_features
                                         + int(mlp.fc1.bias is not None))
                           for old, new, mlp in zip(original_widths, target_widths,
                                                  [block.mlp for block in model.backbone.blocks]))
    if original_count - reduced_count != expected_removed:
        raise RuntimeError("Physical parameter reduction does not match the selected neurons")
    if any(name.endswith((".mask", ".beta")) for name in model.state_dict()):
        raise RuntimeError("An SKP gate remains in the anchor model")
    source_ap = masks.get("val_ap")
    if source_ap is not None and not math.isfinite(float(source_ap)):
        raise ValueError("The mask checkpoint contains an invalid validation AP")
    report = {
        "stage": ANCHOR_STAGE, "species": normalize_species(masks.get("species")),
        "category_id": masks.get("category_id"), "category_ids": masks.get("category_ids"),
        "split": masks.get("split"),
        "config_path": str(config_path), "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_hash, "source_masks": str(masks_path),
        "source_masks_sha256": file_sha256(masks_path), "source_mask_step": masks.get("step"),
        "source_mask_val_ap": source_ap, "baseline_val_ap": masks.get("baseline_val_ap"),
        "original_hidden_sizes": original_widths, "hidden_sizes": target_widths,
        "parameters_before": original_count, "parameters_after": reduced_count,
        "parameters_removed": original_count - reduced_count,
        "parameter_reduction_percent": 100 * (1 - reduced_count / original_count),
        "masked_vs_anchor_max_abs_error": max(errors), "heatmap_shape": list(expected_shape),
        "protected_state_unchanged": True, "verification_device": str(device),
        "verification_seed": args.seed, "verification_rtol": 1e-4, "verification_atol": 1e-5,
        "source_smoke_test": bool(masks.get("training_settings", {}).get("smoke_test", False)),
    }
    if reuse_existing:
        saved_report = json.loads(report_path.read_text(encoding="utf-8"))
        restored, saved_payload = load_anchor_model(cfg, output_path, device)
        # Both files must describe these exact inputs, not just have familiar names.
        names = ("stage", "species", "split", "config_path", "base_checkpoint_sha256",
                 "source_masks_sha256", "source_mask_step", "original_hidden_sizes",
                 "hidden_sizes", "parameters_before", "parameters_after", "heatmap_shape",
                 "source_smoke_test")
        if any(saved.get(name) != report[name]
               for saved in (saved_report, saved_payload) for name in names):
            raise ValueError("Existing anchor inputs differ; choose another --out-dir")
        keys = list(model.state_dict())
        if (list(restored.state_dict()) != keys
                or frozen_state_sha256(restored, keys) != frozen_state_sha256(model, keys)):
            raise ValueError("Existing anchor weights differ from the saved masks; choose another --out-dir")
        reload_errors = [compare_heatmaps(forward_heatmaps(restored, images), expected, "Existing anchor")
                         for images, expected in zip(inputs, reference)]
        detail("Existing anchor weights and heatmaps: PASS (max error {:.3g})".format(max(reload_errors)))
        print("Reusing verified anchor{}:".format(" (smoke test)" if report["source_smoke_test"] else ""),
              display_path(output_path))
        detail("Report:", display_path(report_path))
        return restored, saved_report

    payload = dict(report)
    payload.update({
        "format_version": 1,
        "architecture": {"original_hidden_sizes": original_widths, "hidden_sizes": target_widths},
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    # Verify serialized weights and architecture before publishing either output.
    with tempfile.TemporaryDirectory(prefix="anchor_check_", dir=str(output_dir)) as temporary:
        candidate = Path(temporary) / "anchor.pth"
        atomic_torch_save(payload, candidate)
        restored, _ = load_anchor_model(cfg, candidate, device)
        reload_errors = [compare_heatmaps(forward_heatmaps(restored, images), expected, "Reloaded anchor")
                         for images, expected in zip(inputs, reference)]
        report["reload_max_abs_error"] = max(reload_errors)
        report["standalone_checkpoint_reload"] = True
        os.replace(str(candidate), str(output_path))
        candidate_report = Path(temporary) / "anchor_report.json"
        candidate_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(str(candidate_report), str(report_path))
    detail("Device:", device)
    for index, (old, new) in enumerate(zip(original_widths, target_widths)):
        detail("Block {}: MLP hidden neurons {} -> {}".format(index, old, new))
    print("Parameters: {:,} -> {:,} ({:.2f}% removed)".format(
        original_count, reduced_count, report["parameter_reduction_percent"]))
    detail("Saved-mask heatmap equivalence: PASS (max error {:.3g})".format(max(errors)))
    detail("Attention, residuals and pose head unchanged: PASS")
    detail("Standalone checkpoint reload: PASS")
    if report["source_smoke_test"]:
        print("Smoke test: AP not evaluated.")
    print("Anchor saved:", display_path(output_path))
    detail("Report:", display_path(report_path))
    return model, report


def get_args_parser():
    parser = CommandParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masks", help="Reuse SKP masks and skip training/data loading")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    add_species_argument(parser)
    parser.add_argument("--split", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", help="Output folder; matching completed anchors are verified and reused")
    add_device_argument(parser)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-lr", type=float, default=1e-3)
    parser.add_argument("--beta-lr", type=float, default=1e-1)
    add_scheduler_arguments(parser)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="10 updates; skip AP evaluation")
    parser.add_argument("--self-test", action="store_true", help="Check masks/STE without ViTPose or data")
    parser.add_argument("--threads", type=int, default=1)
    return parser


@command_output
def run(args, parser=None):
    parser = parser or get_args_parser()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.self_test:
        self_test()
        return
    requested_species = args.species
    explicit_masks = bool(args.masks)
    try:
        args = _options(args)
        args.config = str(resolve_input_file(args.config, "--config"))
        args.checkpoint = str(resolve_input_file(args.checkpoint, "--checkpoint"))
        masks_path = _saved_masks_path(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if masks_path is not None:
        args.masks = str(masks_path)
        if requested_species is None:
            args.species = normalize_species(checkpoint_header(masks_path).get("species"))
    output_dir = Path(args.out_dir) if args.out_dir else (masks_path.parent if masks_path else None)
    reuse_existing = output_dir is not None and any((output_dir / name).exists()
                                                   for name in ("anchor.pth", "anchor_report.json"))
    if reuse_existing:
        if not all((output_dir / name).is_file() for name in ("anchor.pth", "anchor_report.json")):
            parser.error("Anchor export is incomplete; choose another --out-dir")
        if masks_path is None:
            parser.error("Existing anchor needs its source masks for verification; select them with --masks")
        print("Verifying existing anchor...", flush=True)
    if output_dir is not None and masks_path is None:
        try:
            _check_training_output(output_dir)
        except FileExistsError as error:
            parser.error(str(error))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    from mmcv import Config
    from mmpose.models import build_posenet

    cfg = Config.fromfile(args.config)
    train_loader = val_loader = None
    if masks_path is None:
        data_root = resolve_data_root(args.data_root)
        args.data_root = str(data_root.resolve())
        try:
            _, paths, args.category_ids = validate_subsets(data_root, args.species, args.split)
            args.category_id = args.category_ids[0] if len(args.category_ids) == 1 else None
        except ValueError as error:
            parser.error(str(error))
        train_loader, val_loader = make_loaders(cfg, paths, data_root, args)
    model = build_posenet(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    del checkpoint
    model.to(args.device).eval()
    try:
        masks = get_mask(args, model, train_loader, val_loader)
        if reuse_existing and not explicit_masks:
            _check_completed_training_settings(args, masks)
        _, report = export_anchor(args, model, masks)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    return report


def main(argv=None):
    parser = get_args_parser()
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
