"""Training-only calibration collection and NuWa neuron allocation."""


import itertools
import json
from numbers import Integral
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from ..paths import (
    DEFAULT_CONFIG, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_SPECIES,
    resolve_data_root, resolve_input_file, resolve_project_path,
)
from ..dataset.utils import evaluation_paths, move_batch
from ..model.vit import forward_heatmaps, hidden_sizes
from device_utils import resolve_device
from ..utils.io import file_sha256
from ..utils.utils import seed_all
from ..species import normalize_species
from ..utils.console import library_output

def calibration_options(args):
    """Accept the current CLI names and NuWa's calibration_sample_num alias."""
    defaults = {
        "config": DEFAULT_CONFIG, "data_root": DEFAULT_DATA_ROOT,
        "species": DEFAULT_SPECIES, "split_id": getattr(args, "split", 1),
        "device": "auto", "seed": 42, "smoke_test": False,
        "strategy": "optimize", "mode": "random", "target_arch": "uniform",
        "calibration_samples": getattr(args, "calibration_sample_num", 128),
    }
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    if (isinstance(args.calibration_samples, bool)
            or not isinstance(args.calibration_samples, Integral)
            or args.calibration_samples <= 0):
        raise ValueError("calibration-samples must be a positive integer")
    if args.mode != "random":
        raise ValueError("This ViTPose workflow supports random calibration crops")
    if args.target_arch != "uniform":
        raise ValueError("This ViTPose workflow supports uniform MLP allocation")
    if getattr(args, "calibration_batch_size", 1) != 1:
        raise ValueError("This ViTPose activation ranking requires calibration batch size 1")
    if args.strategy not in ("optimize", "prune"):
        raise ValueError("Use the existing optimize or prune MLP strategy")
    args.device = str(resolve_device(args.device))
    args.species = normalize_species(args.species)
    return args


def uniform_widths(widths, prune_count, multiple=8):
    """Upstream block-uniform allocation, followed by its floor-to-eight rule."""
    if not widths or any(isinstance(x, bool) or not isinstance(x, Integral) or x < multiple for x in widths):
        raise ValueError("Every anchor MLP must have at least {} neurons".format(multiple))
    if isinstance(prune_count, bool) or not isinstance(prune_count, Integral) or prune_count < 0:
        raise ValueError("Invalid number of MLP neurons to prune")
    if prune_count > sum(widths) - multiple * len(widths):
        raise ValueError("Target is too small with the pose head and minimum MLP widths preserved; lower pruning-rate")
    if prune_count == 0:
        return list(widths)
    total, left = sum(widths), int(prune_count)
    result = [0] * len(widths)
    order = sorted(range(len(widths)), key=lambda i: widths[i])
    for position, index in enumerate(order):
        target = (total - left) // (len(widths) - position)
        result[index] = min(widths[index], target)
        total -= widths[index]
        left -= widths[index] - result[index]
    result = [(int(value) // multiple) * multiple for value in result]
    if any(value < multiple for value in result):
        raise ValueError("Uniform allocation would remove an entire MLP; lower pruning-rate")
    return result


@torch.no_grad()
def collect_training_features(cfg, model, args, species, split_id, token_counts, directory,
                              train_loader=None):
    """One training pass; store selected post-GELU features on disk per block."""
    args = calibration_options(args)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if len(token_counts) != len(hidden_sizes(model)) or any(
            isinstance(tokens, bool) or not isinstance(tokens, Integral) or tokens <= 0
            for tokens in token_counts):
        raise ValueError("Expected one positive token count per ViTPose block")
    data_root = resolve_data_root(args.data_root)
    if train_loader is None:
        from mmpose.datasets import build_dataset, build_dataloader

        annotation_path, image_root, _ = evaluation_paths(data_root, species, "train", split_id)
        cfg.data.train.ann_file = str(annotation_path)
        cfg.data.train.img_prefix = str(image_root) + os.sep
        cfg.data.train.test_mode = False
        with library_output():
            dataset = build_dataset(cfg.data.train)
        loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                                  dist=False, shuffle=False, drop_last=False,
                                  pin_memory=torch.device(args.device).type == "cuda")
    else:
        loader, dataset = train_loader, train_loader.dataset
        if getattr(loader, "batch_size", 1) != 1 or getattr(loader, "drop_last", False):
            raise ValueError("Supply a training loader with batch size 1 and drop_last=False")
        annotation_file = getattr(dataset, "ann_file", None)
        annotation_path = Path(annotation_file) if annotation_file else None
    if len(dataset) == 0:
        raise ValueError("No usable training crops remain for {}".format(species))
    model.to(args.device)
    count = min(2, len(dataset)) if args.smoke_test else len(dataset)
    capture_features = args.strategy == "optimize"
    samples = min(args.calibration_samples, count) if capture_features else 0
    chosen = sorted(random.Random(args.seed).sample(range(count), samples))
    slots = {index: slot for slot, index in enumerate(chosen)}
    widths = hidden_sizes(model)
    arrays, paths, handles = [], [], []
    scores = [np.zeros(width, dtype=np.float32) for width in widths]
    visits = [0] * len(widths)
    context = {"index": -1}
    seed_all(args.seed)
    model.eval()
    try:
        for index, (block, width, tokens) in enumerate(zip(model.backbone.blocks, widths, token_counts)):
            path = directory / "block_{:02d}.npy".format(index)
            arrays.append(np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                                   shape=(samples, tokens, width)) if capture_features else None)
            paths.append(path if capture_features else None)

            def capture(module, inputs, output, block_id=index, expected_width=width, expected_tokens=tokens):
                if tuple(output.shape) != (1, expected_tokens, expected_width):
                    raise RuntimeError("Unexpected MLP activation shape in block {}".format(block_id))
                if not torch.isfinite(output).all().item():
                    raise RuntimeError("Nonfinite training activations")
                # Preserve the supplied code's abs(sum(tokens)) ranking with batch size 1.
                scores[block_id][:] += output.sum(dim=(0, 1)).abs().detach().float().cpu().numpy()
                slot = slots.get(context["index"])
                if slot is not None:
                    arrays[block_id][slot] = output[0].detach().float().cpu().numpy()
                visits[block_id] += 1

            handles.append(block.mlp.act.register_forward_hook(capture))
        print("Collecting activations: {} training crops, {} calibration crops".format(count, samples), flush=True)
        for index, batch in enumerate(itertools.islice(loader, count)):
            context["index"] = index
            data = move_batch(batch, torch.device(args.device))
            heatmaps = forward_heatmaps(model, data["img"])
            if not torch.isfinite(heatmaps).all().item():
                raise RuntimeError("Nonfinite heatmaps during calibration")
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == count:
                print("Activations {}/{}".format(index + 1, count), flush=True)
        if any(value != count for value in visits):
            raise RuntimeError("Not every block was visited once per training crop")
        if any(not np.isfinite(score).all() for score in scores):
            raise RuntimeError("Nonfinite accumulated activation scores")
    finally:
        for handle in handles:
            handle.remove()
        for array in arrays:
            if array is not None:
                array.flush()
                array._mmap.close()
    return scores, paths, {"species": normalize_species(species),
                           "annotation_file": str(annotation_path) if annotation_path else None,
                           "annotation_sha256": file_sha256(annotation_path) if annotation_path else None,
                           "training_crops_processed": count, "calibration_crops": samples,
                           "calibration_dataset_indices": chosen, "seed": args.seed,
                           "batch_size": 1, "ranking": "sum over crops of abs(sum over patch tokens)",
                           "features": "post-GELU, training split only" if capture_features else "not collected; activation ranking only"}


@torch.no_grad()
def calibration_token_counts(cfg, model, device="auto"):
    """Measure activation token counts without assuming a classifier CLS token."""
    widths = hidden_sizes(model)
    visits, handles = [[] for _ in widths], []
    width, height = map(int, cfg.data_cfg.image_size)
    parameter = next(model.parameters())
    device = resolve_device(device)
    model.to(device).eval()
    images = torch.zeros(1, 3, height, width, device=device, dtype=parameter.dtype)
    try:
        for index, block in enumerate(model.backbone.blocks):
            def capture(module, inputs, output, block_id=index):
                if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != widths[block_id]:
                    raise ValueError("Expected ViTPose MLP activations [1, tokens, hidden]")
                visits[block_id].append(int(output.shape[1]))
            handles.append(block.mlp.act.register_forward_hook(capture))
        forward_heatmaps(model, images)
    finally:
        for handle in handles:
            handle.remove()
    if any(len(value) != 1 for value in visits):
        raise ValueError("Expected one activation visit per ViTPose block")
    return [value[0] for value in visits]


@torch.no_grad()
def get_activation_and_calibration_features(args, model_anchor, train_loader=None,
                                            directory=None, cfg=None, token_counts=None):
    """Return scores and disk-backed feature paths; collect fresh training data."""
    args = calibration_options(args)
    cfg = cfg if cfg is not None else getattr(args, "cfg", None)
    if cfg is None:
        from mmcv import Config
        cfg = Config.fromfile(str(resolve_input_file(args.config, "--config")))
    if token_counts is None:
        token_counts = calibration_token_counts(cfg, model_anchor, args.device)
    if directory is None:
        parent = resolve_project_path(getattr(args, "out_dir", None) or getattr(args, "output_dir", None)
                      or DEFAULT_OUTPUT_ROOT / "nuwa_calibration")
        parent.mkdir(parents=True, exist_ok=True)
        directory = tempfile.mkdtemp(prefix="calibration_", dir=str(parent))
    scores, paths, metadata = collect_training_features(
        cfg, model_anchor, args, args.species, args.split_id, token_counts, directory,
        train_loader=train_loader,
    )
    directory = Path(directory)
    np.savez_compressed(directory / "activation_scores.npz",
                        **{"block_{}".format(i): score for i, score in enumerate(scores)})
    (directory / "calibration_report.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                       encoding="utf-8")
    return scores, paths


def get_config_and_indices(args, model_anchor, prune_neuron_count, A):
    """Apply NuWa's uniform floor-to-eight allocation and activation ranking."""
    if getattr(args, "target_arch", "uniform") != "uniform":
        raise ValueError("This ViTPose workflow supports uniform MLP allocation")
    widths = hidden_sizes(model_anchor)
    if len(A) != len(widths):
        raise ValueError("Expected one activation score vector per ViTPose block")
    checked = []
    for index, (score, width) in enumerate(zip(A, widths)):
        score = score.detach().cpu().numpy() if torch.is_tensor(score) else np.asarray(score)
        if score.shape != (width,) or not np.isfinite(score).all():
            raise ValueError("Invalid activation scores in block {}".format(index))
        checked.append(score)
    targets = uniform_widths(widths, prune_neuron_count)
    removed = [torch.topk(torch.from_numpy(np.ascontiguousarray(score)),
                          k=old - new, largest=False).indices
               for score, old, new in zip(checked, widths, targets)]
    return targets, removed
