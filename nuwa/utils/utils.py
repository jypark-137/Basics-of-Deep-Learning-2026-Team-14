"""NuWa pose utilities: reproducibility, matrix costs, timing and neuron scores."""


import argparse
import copy
import math
from numbers import Integral
import os
import random
import statistics
import tempfile
import time
import numpy as np
import torch
from torch import nn
from ..paths import (
    DEFAULT_CONFIG, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_SPECIES,
    resolve_data_root, resolve_project_path,
)
from ..model.vit import forward_heatmaps, heatmap_task_loss, hidden_sizes, mlp_layers
from ..model.utils import pose_config, pose_input
from device_utils import add_device_argument, resolve_device
from ..model.checkpoint import load_pose_checkpoint
from ..dataset.utils import evaluation_paths, move_batch
from .io import atomic_json, atomic_torch_save, state_size_bytes

MAC_CONVENTION = (
    "One multiply-accumulate = one MAC; counts Linear, Conv2d, ConvTranspose2d, "
    "QK and attention-value matrix products in the complete pose forward. "
    "Excludes bias adds, normalization, activations, softmax and interpolation. "
    "One crop, no flip testing. These are operation counts, not measured latency."
)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def profile_pose_macs(model, images, forward_fn):
    """Profile the actual full-pose forward, including the fixed decoder cost."""
    if images.shape[0] != 1:
        raise ValueError("Profile one crop at a time")
    costs, handles = {}, []
    blocks = list(model.backbone.blocks)
    rows = {index: [] for index in range(len(blocks))}
    linear_to_block = {id(block.mlp.fc1): index for index, block in enumerate(blocks)}

    def add(name, amount):
        costs[name] = costs.get(name, 0) + int(amount)

    def weighted_hook(name):
        def hook(module, inputs, output):
            if isinstance(module, nn.Linear):
                add(name, output.numel() * module.in_features)
                if id(module) in linear_to_block:
                    rows[linear_to_block[id(module)]].append(inputs[0].numel() // module.in_features)
            elif isinstance(module, nn.Conv2d):
                kernel = module.kernel_size[0] * module.kernel_size[1]
                add(name, output.numel() * (module.in_channels // module.groups) * kernel)
            elif isinstance(module, nn.ConvTranspose2d):
                kernel = module.kernel_size[0] * module.kernel_size[1]
                add(name, inputs[0].numel() * (module.out_channels // module.groups) * kernel)
        return hook

    def attention_hook(name):
        def hook(module, inputs, output):
            batch, tokens, _ = inputs[0].shape
            dim = module.qkv.out_features // (3 * module.num_heads)
            qk = getattr(module, "qk_dim", dim)
            vo = getattr(module, "vo_dim", dim)
            add(name + ".matrix_products", batch * module.num_heads * tokens**2 * (qk + vo))
        return hook

    try:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                handles.append(module.register_forward_hook(weighted_hook(name)))
            if hasattr(module, "qkv") and hasattr(module, "num_heads") and hasattr(module, "proj"):
                handles.append(module.register_forward_hook(attention_hook(name)))
        output = forward_fn(model, images)
    finally:
        for handle in handles:
            handle.remove()
    if any(len(value) != 1 for value in rows.values()):
        raise ValueError("Expected one visit to every ViTPose MLP per forward")
    per_neuron = [rows[index][0] * (block.mlp.fc1.in_features + block.mlp.fc2.out_features)
                  for index, block in enumerate(blocks)]
    return {"pose_macs": int(sum(costs.values())),
            "backbone_macs": int(sum(value for name, value in costs.items() if name.startswith("backbone."))),
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "per_neuron_macs": per_neuron, "tokens_per_block": [rows[i][0] for i in range(len(blocks))],
            "convention": MAC_CONVENTION}, output


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_sub_task(args):
    """Return the species task; the loader selects animals, not joint channels."""
    species = getattr(args, "species", None) or DEFAULT_SPECIES
    if not isinstance(species, str) or not species.strip():
        raise ValueError("Use a nonempty AP-10K species name, or all")
    return species.strip()


@torch.no_grad()
def get_gflops(model=None, config=None, input_size=None):
    """Return full-pose GMACs; preserves NuWa's historical counting units."""
    if model is None and isinstance(config, dict) and "pose_macs" in config:
        count = float(config["pose_macs"])
        if not math.isfinite(count) or count <= 0:
            raise ValueError("pose_macs must be finite and positive")
        return count / 1e9
    cfg = None
    if model is None:
        from mmpose.models import build_posenet
        cfg = pose_config(config)
        model = build_posenet(cfg.model)
    images = pose_input(model, cfg if cfg is not None else config, input_size)
    model.eval()
    profile, _ = profile_pose_macs(model, images[:1], forward_heatmaps)
    return profile["pose_macs"] / 1e9


def is_identity_layer(attn: torch.nn.Module):
    has_params = any(p.numel() > 0 for p in attn.parameters())
    return not has_params


def get_mparam(model):
    return sum(p.numel() for p in model.parameters())/1e6


def latency_statistics(samples_ms):
    samples = [float(value) for value in samples_ms]
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("Latency samples must be finite positive milliseconds")
    ordered = sorted(samples)
    position = 0.9 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    p90 = ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
    return {"median_ms": float(statistics.median(samples)), "mean_ms": statistics.mean(samples),
            "p90_ms": p90, "std_ms": statistics.pstdev(samples),
            "repeats": len(samples), "samples_ms": samples}


def benchmark_pose(model, images, warmup, repeats):
    """Measure complete heatmap forwards on the model's CPU or CUDA device."""
    device = images.device
    parameter = next(model.parameters(), None)
    if device.type not in ("cpu", "cuda") or (parameter is not None and parameter.device != device):
        raise ValueError("Benchmark inputs and model must use the same CPU or CUDA device")
    if (isinstance(warmup, bool) or not isinstance(warmup, Integral) or warmup < 0
            or isinstance(repeats, bool) or not isinstance(repeats, Integral) or repeats <= 0):
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            forward_heatmaps(model, images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            forward_heatmaps(model, images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    result = latency_statistics(samples)
    result.update({"warmup": warmup, "batch_size": int(images.shape[0]), "CPU_threads": torch.get_num_threads(),
                   "device": str(device),
                   "peak_memory_GiB": (torch.cuda.max_memory_allocated(device) / 1024**3
                                       if device.type == "cuda" else None),
                   "scope": "heatmap forward only; no preprocessing, decoding or flip testing"})
    return result


def benchmark_cpu(model, images, warmup, repeats):
    """Retain the explicitly CPU-only helper for existing callers."""
    if images.device.type != "cpu":
        raise ValueError("benchmark_cpu requires CPU inputs and model")
    return benchmark_pose(model, images, warmup, repeats)


@torch.no_grad()
def profiling(args, model, input_size=None, warm_steps=3, run_steps=10):
    """Time the existing pose forward and count complete-model matrix MACs."""
    if (isinstance(warm_steps, bool) or not isinstance(warm_steps, Integral) or warm_steps < 0
            or isinstance(run_steps, bool) or not isinstance(run_steps, Integral) or run_steps <= 0):
        raise ValueError("warm_steps must be nonnegative and run_steps must be positive")
    device = resolve_device(getattr(args, "device", None), model)
    model.to(device).eval()
    config = getattr(args, "cfg", None) or getattr(args, "config", None)
    images = pose_input(model, config, input_size, getattr(args, "seed", 42))
    profile, heatmaps = profile_pose_macs(model, images[:1], forward_heatmaps)
    if heatmaps.ndim != 4 or not torch.isfinite(heatmaps).all().item():
        raise RuntimeError("Profiling requires finite [batch, joints, height, width] heatmaps")
    timings = benchmark_pose(model, images, warm_steps, run_steps)
    return {
        "inference time (ms)": timings["mean_ms"],
        "throughput (image/s)": images.shape[0] * 1000 / timings["mean_ms"],
        "peak_memory (GB)": timings["peak_memory_GiB"], "#Params (M)": get_mparam(model),
        "FLOPs (G)": profile["pose_macs"] / 1e9,
        "GMACs": profile["pose_macs"] / 1e9,
        "matrix_GFLOPs_2_per_MAC": 2 * profile["pose_macs"] / 1e9,
        "profile": profile, "latency": timings, "device": str(device),
        "state_bytes": state_size_bytes(model),
    }


def _training_loader(args, cfg):
    loader = getattr(args, "train_loader", None)
    if loader is not None:
        return loader
    from mmpose.datasets import build_dataset, build_dataloader
    data_root = resolve_data_root(getattr(args, "data_root", DEFAULT_DATA_ROOT))
    split_id = getattr(args, "split_id", None) or getattr(args, "split", 1)
    path, image_root, _ = evaluation_paths(data_root, get_sub_task(args), "train", split_id)
    dataset_cfg = copy.deepcopy(cfg.data.train)
    dataset_cfg.ann_file, dataset_cfg.img_prefix = str(path), str(image_root) + os.sep
    dataset_cfg.test_mode = False
    dataset = build_dataset(dataset_cfg)
    if len(dataset) == 0:
        raise ValueError("No usable training animals remain")
    return build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                            dist=False, shuffle=False, drop_last=False,
                            pin_memory=resolve_device(getattr(args, "device", "auto")).type == "cuda")


def get_score(args, model, metric="mag"):
    """Compute NuWa's mag/act/grad/taylor neuron scores for the pose task.

    Fresh scores avoid reusing another model's cached values. Gradient scores
    sum training-loss gradients before applying the original absolute means.
    Existing parameter gradients, requires_grad flags and training modes are
    restored; no optimizer runs and model weights are never updated.
    """
    if metric not in ("mag", "act", "grad", "taylor"):
        raise ValueError("metric must be mag, act, grad, or taylor")
    layers = mlp_layers(model)
    if metric == "mag":
        return [(mlp.fc1.weight.detach().abs().mean(1)
                 + mlp.fc2.weight.detach().abs().mean(0)).cpu() for mlp in layers]
    modes = [(module, module.training) for module in model.modules()]
    parameters = [weight for mlp in layers for weight in (mlp.fc1.weight, mlp.fc2.weight)]
    flags = [parameter.requires_grad for parameter in parameters]
    try:
        device = resolve_device(getattr(args, "device", None), model)
        model.to(device).eval()
        cfg = getattr(args, "cfg", None) or pose_config(getattr(args, "config", None))
        if metric == "act":
            from ..method.get_clibration_data import get_activation_and_calibration_features
            options = argparse.Namespace(**vars(args))
            options.strategy, options.device, options.cfg = "prune", str(device), cfg
            with tempfile.TemporaryDirectory(prefix="nuwa_scores_") as directory:
                values, _ = get_activation_and_calibration_features(
                    options, model, train_loader=getattr(args, "train_loader", None),
                    directory=directory, cfg=cfg,
                )
            return [torch.from_numpy(value.copy()) for value in values]
        loader = _training_loader(args, cfg)
        if len(loader) == 0:
            raise ValueError("Gradient scoring requires a nonempty training loader")
        for parameter in parameters:
            parameter.requires_grad_(True)
        totals = [torch.zeros_like(parameter, device="cpu") for parameter in parameters]
        with torch.enable_grad():
            for batch in loader:
                data = move_batch(batch, device)
                heatmaps = forward_heatmaps(model, data["img"])
                loss = heatmap_task_loss(model, heatmaps, data["target"], data["target_weight"])
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Nonfinite pose loss during gradient scoring")
                gradients = torch.autograd.grad(loss, parameters)
                for total, gradient in zip(totals, gradients):
                    if not torch.isfinite(gradient).all().item():
                        raise RuntimeError("Nonfinite neuron-score gradient")
                    total.add_(gradient.detach().cpu())
        result = []
        for index, mlp in enumerate(layers):
            g1, g2 = totals[2 * index:2 * index + 2]
            if metric == "taylor":
                g1, g2 = mlp.fc1.weight.detach().cpu() * g1, mlp.fc2.weight.detach().cpu() * g2
            result.append(g1.abs().mean(1) + g2.abs().mean(0))
        return result
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
        for module, training in modes:
            module.training = training


@torch.no_grad()
def score_based_pruning(args, model, metric="mag"):
    """Zero equal numbers of low-score fc2 columns, retaining dense shapes."""
    rate = float(args.pruning_rate)
    if not math.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("pruning_rate must lie in [0, 1)")
    if rate == 0:
        return model
    model.to(resolve_device(getattr(args, "device", None), model)).eval()
    cfg = getattr(args, "cfg", None) or getattr(args, "config", None)
    images = pose_input(model, cfg)
    profile, _ = profile_pose_macs(model, images, forward_heatmaps)
    costs = profile["per_neuron_macs"]
    if not costs or len(set(costs)) != 1:
        raise ValueError("Uniform scoring requires the same MLP neuron cost per block")
    count = int(rate * profile["pose_macs"] / costs[0])
    per_block = math.ceil(count / len(costs))
    if any(per_block > width for width in hidden_sizes(model)):
        raise ValueError("Uniform masking exceeds an MLP width; lower pruning_rate")
    scores = get_score(args, model, metric)
    for mlp, score in zip(mlp_layers(model), scores):
        indices = torch.topk(score, per_block, largest=False).indices
        mlp.fc2.weight[:, indices.to(mlp.fc2.weight.device)] = 0
    return model


def less_is_more(args, model, test_loader, repeat_num=1000):
    """Repeat the original random-neuron masking experiment using pose AP.

    test_loader is a species validation loader; AP values use 0-100 units.
    Models are copied for each trial. Requested rates describe the masked
    matrix-operation budget, not an actual speedup of the dense model.
    """
    if isinstance(repeat_num, bool) or not isinstance(repeat_num, Integral) or repeat_num <= 0:
        raise ValueError("repeat_num must be a positive integer")
    from ..engine.eval import evaluate
    model.to(resolve_device(getattr(args, "device", None), model)).eval()
    cfg = getattr(args, "cfg", None) or getattr(args, "config", None)
    images = pose_input(model, cfg)
    profile, _ = profile_pose_macs(model, images, forward_heatmaps)
    costs, widths = profile["per_neuron_macs"], hidden_sizes(model)
    if not costs or len(set(costs)) != 1:
        raise ValueError("Random masking requires equal MLP neuron costs")
    baseline = evaluate(model, test_loader)
    results = []
    for _ in range(repeat_num):
        trial = copy.deepcopy(model)
        rate = random.uniform(0.0, 0.2)
        count = min(sum(widths), int(rate * profile["pose_macs"] / costs[0]))
        selection = torch.randperm(sum(widths))[:count]
        mask = torch.ones(sum(widths), dtype=torch.bool)
        mask[selection] = False
        offset = 0
        with torch.no_grad():
            for mlp, width in zip(mlp_layers(trial), widths):
                mlp.fc2.weight.mul_(mask[offset:offset + width].to(mlp.fc2.weight).view(1, -1))
                offset += width
        ap = evaluate(trial, test_loader)
        results.append({"rate": rate, "acc": ap, "state": ap > baseline,
                        "masked_neurons": count, "baseline_ap": baseline})
        print("Masked budget: {:.2f}% | AP: {:.2f} | Improved: {}".format(100 * rate, ap, ap > baseline))
    output = resolve_project_path(getattr(args, "out_dir", None) or getattr(args, "output_dir", None)
                  or DEFAULT_OUTPUT_ROOT) / "motivation" / "less_is_more.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(results, output)
    return results


def get_args_parser():
    from ..paths import DEFAULT_CHECKPOINT
    parser = argparse.ArgumentParser(description="Profile full-pose matrix costs and heatmap-forward latency.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    add_device_argument(parser)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", help="Optional JSON report")
    return parser


def run(args, parser=None):
    import json
    parser = parser or get_args_parser()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    cfg = pose_config(args.config)
    model, _ = load_pose_checkpoint(cfg, args.checkpoint, resolve_device(args.device))
    result = profiling(args, model, warm_steps=args.warmup, run_steps=args.repeats)
    if args.out:
        atomic_json(resolve_project_path(args.out), result)
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = get_args_parser()
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
