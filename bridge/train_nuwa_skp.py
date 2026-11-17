"""Learn vanilla NuWa SKP masks for one AP-10K species.

Keep this file, nuwa_vitpose_bridge.py and bridge_paths.py in
bridge/. Run these examples from the project root.

First check the complete training connection for 10 updates:
    python bridge/train_nuwa_skp.py --smoke-test

Then run mask learning (10,000 updates by default):
    python bridge/train_nuwa_skp.py

The default species is dog; --species can override it. The default data root
is bridge/data/ap10k. Configs use vitpose/configs/, the checkpoint is
bridge/checkpoints/split/ap10k.pth, and outputs use bridge/work_dirs/.

The device is CUDA when available, otherwise CPU. Override with --device cpu
or --device cuda:0. Use --help for the model paths and training settings.
Local mmpose imports work through bridge_paths.py without an editable pip
install. Existing third-party dependencies must still be installed.

Method: NuWa's original TopKBinarizer and MaskLinear below are copied unchanged
from the get_anchor_model.py code you supplied. SKP trains only MLP masks and
their per-block beta parameters. MHA is untouched, as in the paper's SKP.
Scores start at 1 and beta at 5. The original ceil-based rule can mask a few
neurons immediately; initial masked predictions need not equal the baseline.
The optimizer is AdamW with mask LR .001, beta LR .1, and weight decay .05.
The paper's 10,000 updates and batch size 1 are the defaults.

Pose adaptation: keep the existing AP-10K training pipeline and weighted
heatmap loss. Freeze all original parameters and BatchNorm statistics.
Backbone dropout/stochastic depth use training mode, following NuWa.
Species validation AP selects the best masks. The test split is not loaded.

Outputs: last_masks.pth, best_masks.pth (full runs only), run_config.json and
history.jsonl in a new bridge/work_dirs/nuwa_skp/<species>/run_* folder.
Checkpoints contain learned scores, betas, binary masks, base-checkpoint
identity and the validation result measured for that saved state. They are
mask-only files, NOT ordinary ViTPose state_dict checkpoints. The base weights
are still needed. Physical neuron removal is step 4; masks alone do not make
the dense model faster. A smoke run verifies implementation, not AP quality.

Sources:
    https://github.com/CGCL-codes/NuWa
    NuWa.pdf, sections 3.2 and 4.1, supplied with this project
"""

import argparse
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import time

if __package__:
    from .bridge_paths import (
        DEFAULT_CHECKPOINT, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_SPECIES,
        resolve_input_file,
    )
    from .nuwa_vitpose_bridge import (
        DEFAULT_CONFIG, forward_heatmaps, get_nuwa_backbone, heatmap_task_loss,
    )
else:
    from bridge_paths import (
        DEFAULT_CHECKPOINT, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_SPECIES,
        resolve_input_file,
    )
    from nuwa_vitpose_bridge import (
        DEFAULT_CONFIG, forward_heatmaps, get_nuwa_backbone, heatmap_task_loss,
    )

import numpy as np
import torch
import torch.nn as nn

class TopKBinarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs:torch.tensor, threshold):
        threshold = torch.sigmoid(threshold).item()
        mask = inputs.clone()
        _, idx = inputs.sort(descending=True)
        j = math.ceil(threshold * inputs.numel())
        mask[idx[j:]] = 0.
        mask[idx[:j]] = 1.
        ctx.save_for_backward(mask)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output, ((grad_output * mask).sum()).view(-1)

class MaskLinear(nn.Linear):
    def __init__(self, weight, bias, mask, beta, mask_type="mlp", head_dim=64):
        super().__init__(weight.size(1), weight.size(0), bias=False)
        self.weight = weight
        self.bias = bias
        self.mask = mask
        self.beta = beta
        self.mask_type = mask_type
        self.head_dim = head_dim

    def forward(self, x):
        mask = TopKBinarizer.apply(self.mask, self.beta)
        if self.mask_type == "mha":
            mask = mask.repeat_interleave(self.head_dim)
        x = x * mask.view(1,1,-1)
        return nn.functional.linear(x, self.weight, self.bias)


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


def get_skp_parameters(model):
    layers = [block.mlp.fc2 for block in model.backbone.blocks]
    if not all(isinstance(layer, MaskLinear) for layer in layers):
        raise ValueError("Expected an SKP mask in every MLP")
    return [layer.mask for layer in layers], [layer.beta for layer in layers]


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


@torch.no_grad()
def export_masks(model):
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

    Step 4 should use the saved binary_mlp_masks as authoritative for neuron
    indices: sorting tied scores can differ across devices.
    """
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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_state_sha256(model, original_keys):
    """Check original parameters AND buffers, excluding added mask tensors."""
    state = model.state_dict()
    digest = hashlib.sha256()
    for name in original_keys:
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def save_masks(model, path, metadata, step, val_ap, baseline_ap):
    mask_info = export_masks(model)
    payload = dict(metadata)
    payload.update(mask_info)
    payload.update({"format_version": 1, "stage": "NuWa-SKP",
                    "step": step, "val_ap": val_ap, "baseline_val_ap": baseline_ap})
    atomic_torch_save(payload, path)
    return mask_info


def validate_subsets(data_root, species, split):
    """Require the existing step-1 files and guard against mixed-species data."""
    slug = re.sub(r"[^a-z0-9]+", "_", species.strip().casefold()).strip("_")
    if not slug:
        raise ValueError("Use a species name from prepare_species.py --list")
    subset_dir = data_root / "annotations" / "species" / slug
    paths, filenames, category_ids = {}, {}, []
    for part in ("train", "val"):
        path = subset_dir / "ap10k-{}-split{}.json".format(part, split)
        if not path.is_file():
            raise ValueError("Missing step-1 subset: {}. Run prepare_species.py first.".format(path))
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        for key in ("images", "annotations", "categories"):
            if not isinstance(data.get(key), list) or not data[key]:
                raise ValueError("{} needs a nonempty {} list".format(path, key))
        matches = [item["id"] for item in data["categories"]
                   if item["name"].strip().casefold() == species.strip().casefold()]
        if len(matches) != 1:
            raise ValueError("Species name is not unique in {}".format(path))
        category_id = matches[0]
        if any(item["category_id"] != category_id for item in data["annotations"]):
            raise ValueError("{} includes other species' annotations".format(path))
        image_ids = {item["id"] for item in data["images"]}
        if any(item["image_id"] not in image_ids for item in data["annotations"]):
            raise ValueError("{} contains annotations with missing image records".format(path))
        filenames[part] = {item["file_name"] for item in data["images"]}
        for filename in filenames[part]:
            if not (data_root / "data" / filename).is_file():
                raise ValueError("Image not found: {}".format(data_root / "data" / filename))
        paths[part] = path
        category_ids.append(category_id)
    if len(set(category_ids)) != 1:
        raise ValueError("Train and validation category IDs differ")
    if filenames["train"] & filenames["val"]:
        raise ValueError("Species train and validation contain overlapping image filenames")
    return slug, paths, category_ids[0]


def make_loaders(cfg, paths, data_root, args):
    from mmpose.datasets import build_dataset, build_dataloader

    for part in ("train", "val"):
        cfg.data[part].ann_file = str(paths[part])
        cfg.data[part].img_prefix = str(data_root / "data") + os.sep
        cfg.data[part].test_mode = part == "val"
    train_data = build_dataset(cfg.data.train)
    if len(train_data) == 0:
        raise ValueError("No usable training animals remain after dataset filtering")
    train_loader = build_dataloader(
        train_data, samples_per_gpu=args.batch_size, workers_per_gpu=0,
        dist=False, shuffle=True, drop_last=False,
    )
    val_loader = None
    if not args.smoke_test:
        val_data = build_dataset(cfg.data.val)
        if len(val_data) == 0:
            raise ValueError("No usable validation animals remain after dataset filtering")
        val_loader = build_dataloader(
            val_data, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False,
        )
        print("Validation animals:", len(val_data), flush=True)
    print("Training animals:", len(train_data), flush=True)
    return train_loader, val_loader


def move_batch(batch, device):
    from mmcv.parallel import scatter
    destination = -1 if device.type == "cpu" else (device.index or 0)
    return scatter(batch, [destination])[0]


@torch.no_grad()
def evaluate_ap(model, loader, device, output_dir):
    import mmcv
    model.eval()
    outputs = []
    progress = mmcv.ProgressBar(len(loader.dataset))
    for batch in loader:
        data = move_batch(batch, device)
        outputs.append(model(return_loss=False, **data))
        progress.update()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = loader.dataset.evaluate(outputs, res_folder=str(output_dir), metric="mAP")
    ap = float(metrics["AP"])
    if not math.isfinite(ap) or not 0 <= ap <= 1:
        raise RuntimeError("Invalid validation AP: {}".format(ap))
    print("\nValidation AP: {:.2f}".format(100 * ap), flush=True)
    return ap


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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="AP-10K folder; default: bridge/data/ap10k")
    parser.add_argument("--species", default=DEFAULT_SPECIES,
                        help="The same species as step 1; default: dog")
    parser.add_argument("--split", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", help="New output folder; default is a timestamped run")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-lr", type=float, default=1e-3)
    parser.add_argument("--beta-lr", type=float, default=1e-1)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="10 updates; skip AP evaluation")
    parser.add_argument("--self-test", action="store_true", help="Test masks/checkpoints without AP-10K")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.data_root or not args.species:
        parser.error("--data-root and --species cannot be empty")
    if min(args.steps, args.batch_size, args.eval_every, args.log_every) <= 0:
        parser.error("steps, batch-size, eval-every and log-every must be positive")
    if (not all(math.isfinite(value) for value in
                (args.mask_lr, args.beta_lr, args.weight_decay))
            or min(args.mask_lr, args.beta_lr) <= 0 or args.weight_decay < 0):
        parser.error("Learning rates must be positive and weight decay nonnegative")
    if args.smoke_test:
        args.steps = 10
    device_name = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        parser.error("Use cpu or cuda:0 for this MMCV version")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; use --device cpu")
        torch.cuda.set_device(device)
    data_root = Path(args.data_root)
    try:
        config_path = resolve_input_file(args.config, "--config")
        checkpoint_path = resolve_input_file(args.checkpoint, "--checkpoint")
    except FileNotFoundError as error:
        parser.error(str(error))
    slug, paths, category_id = validate_subsets(data_root, args.species, args.split)
    self_test()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    from mmcv import Config
    from mmpose.models import build_posenet
    cfg = Config.fromfile(str(config_path))
    train_loader, val_loader = make_loaders(cfg, paths, data_root, args)
    model = build_posenet(cfg.model)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    del checkpoint
    model.to(device)
    model.eval()
    # Reentrant checkpointing can disconnect gates when all earlier weights
    # are frozen. The working AP-10K config already disables it.
    if getattr(model.backbone, "use_checkpoint", False):
        raise ValueError("Set backbone.use_checkpoint=False in the config for SKP")
    original_keys = list(model.state_dict())
    original_digest = frozen_state_sha256(model, original_keys)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S") + ("_smoke" if args.smoke_test else "")
    output_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUTPUT_ROOT / "nuwa_skp" / slug / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = dict(vars(args))
    settings.update({"device": str(device), "config": str(config_path),
                     "checkpoint": str(checkpoint_path), "data_root": str(data_root),
                     "train_annotations": str(paths["train"]), "val_annotations": str(paths["val"])})
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)
        handle.write("\n")
    print("Device:", device, "| Output:", output_dir, flush=True)

    baseline_ap = None
    if val_loader is not None:
        print("Measuring the unpruned species validation baseline...", flush=True)
        baseline_ap = evaluate_ap(model, val_loader, device, output_dir / "eval" / "unpruned")
    masks, betas = attach_skp_masks(model)
    count = check_trainable_parameters(model)
    print("Trainable mask/beta parameters:", count, flush=True)
    optimizer = torch.optim.AdamW([
        {"params": masks, "lr": args.mask_lr, "weight_decay": args.weight_decay},
        {"params": betas, "lr": args.beta_lr, "weight_decay": args.weight_decay},
    ], betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    metadata = {"species": args.species, "category_id": category_id, "split": args.split,
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
    print("Initially masked MLP hidden neurons: {} / {} ({:.2f}%)".format(
        total - kept, total, 100 * (1 - kept / total)), flush=True)
    interrupted = False
    start = time.perf_counter()
    iterator = iter(train_loader)
    set_skp_train_mode(model)

    def record(event):
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    record({"event": "initial", "baseline_val_ap": baseline_ap, "masked_val_ap": best_ap,
            "kept_neurons": initial_info["kept_neurons"]})
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
            optimizer.step()
            step += 1
            if any(not torch.isfinite(parameter).all().item() for parameter in masks + betas):
                raise RuntimeError("Nonfinite mask/beta after update {}".format(step))

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                info = export_masks(model)
                kept, total = remaining_neurons(info)
                loss_value = float(loss.detach().item())
                print("Step {}/{} | loss {:.6f} | MLP neurons masked {:.2f}% | {:.1f}s/update".format(
                    step, args.steps, loss_value, 100 * (1 - kept / total),
                    (time.perf_counter() - start) / step), flush=True)
                record({"event": "train", "step": step, "loss": loss_value,
                        "kept_neurons": info["kept_neurons"]})

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
    print("Original weights and buffers unchanged: PASS", flush=True)
    if step > 0:
        print("Mask/beta updates: PASS", flush=True)
    if args.smoke_test:
        print("{} after {} updates".format("Smoke test stopped" if interrupted else "SKP smoke test passed", step), flush=True)
    elif best_ap is not None:
        print("Best masked validation AP: {:.2f} at step {} | unpruned: {:.2f}".format(
            100 * best_ap, best_step, 100 * baseline_ap), flush=True)
        print("Use best_masks.pth for step 4; compare its AP with the unpruned baseline.", flush=True)
    print("Masks saved in:", output_dir, flush=True)


if __name__ == "__main__":
    main()
