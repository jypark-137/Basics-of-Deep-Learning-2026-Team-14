"""NuWa OFP: attention factorization, MLP reconstruction and pruning entry points."""


import argparse
from datetime import datetime
import gc
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import tempfile
import numpy as np
import torch
from torch import nn
from ..paths import (
    DEFAULT_CONFIG, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_SPECIES,
    resolve_input_file, resolve_project_path,
)
from ..model.vit import AttentionPruned, _attention_spec, _linear_like, attention_specs, forward_heatmaps, hidden_sizes
from ..model.checkpoint import ANCHOR_STAGE, BASE_STAGE, OFP_STAGE, load_anchor_model, load_nuwa_model, load_pose_checkpoint
from device_utils import add_device_argument, resolve_device
from ..utils.io import atomic_torch_save, file_sha256, frozen_state_sha256
from ..utils.utils import profile_pose_macs, seed_all
from .get_clibration_data import calibration_options, collect_training_features, get_activation_and_calibration_features, get_config_and_indices, uniform_widths

def energy_ratio(pruning_rate):
    alpha = float(pruning_rate)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("pruning-rate must lie strictly between 0 and 1")
    return -0.41 * alpha**3 + 0.14 * alpha**2 - 0.03 * alpha + 1.0


def product_svd(left, right):
    """Thin SVD of left @ right through reduced QR; retains all nonzero ranks."""
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("Incompatible product shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Nonfinite weights before SVD")
    ql, rl = np.linalg.qr(left, mode="reduced")
    qr, rr = np.linalg.qr(right.T, mode="reduced")
    u, singular, vh = np.linalg.svd(rl @ rr.T, full_matrices=False)
    return ql @ u, singular, vh @ qr.T


def retained_dimension(spectra, original_dim, rho, multiple=8):
    if not 0 < float(rho) <= 1 or not spectra:
        raise ValueError("Invalid retained energy ratio or empty spectra")
    if original_dim < multiple or multiple <= 0:
        raise ValueError("Attention dimensions must be at least {}".format(multiple))
    fractions = []
    for values in spectra:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) < original_dim or not np.isfinite(values).all():
            raise ValueError("Invalid attention singular values")
        powers = values[:original_dim] ** 2
        total = float(np.sum(values**2))
        fractions.append(np.cumsum(powers) / total if total > 0 else np.ones(original_dim))
    average = np.mean(fractions, axis=0)
    raw = min(original_dim, int(np.searchsorted(average, rho, side="left")) + 1)
    # The upstream floor-to-eight can produce zero on a degenerate spectrum.
    # Keep one eight-wide group so that softmax attention remains defined.
    aligned = max(multiple, (raw // multiple) * multiple)
    return aligned, float(average[aligned - 1])


def factor_attention(qkv_weight, qkv_bias, out_weight, out_bias, heads, rho,
                     original_scale=None, dimensions=None):
    """Return packed Q/K/V and output weights using upstream balanced factors."""
    weight, output = np.asarray(qkv_weight), np.asarray(out_weight)
    if weight.ndim != 2 or weight.shape[0] != 3 * weight.shape[1]:
        raise ValueError("Expected ordinary square-width ViTPose QKV weights")
    width = weight.shape[1]
    if not isinstance(heads, Integral) or heads <= 0 or width % heads or output.shape != (width, width):
        raise ValueError("Invalid attention heads or output projection")
    old_dim = width // heads
    has_bias = qkv_bias is not None
    bias = np.zeros(3 * width, dtype=weight.dtype) if qkv_bias is None else np.asarray(qkv_bias)
    if bias.shape != (3 * width,):
        raise ValueError("Invalid QKV bias shape")
    q, k, v = weight.reshape(3, heads, old_dim, width)
    bq, bk, bv = bias.reshape(3, heads, old_dim)
    wo = output.T.reshape(heads, old_dim, width)
    qk, vo = [], []
    for head in range(heads):
        q_hat = np.concatenate((q[head], bq[head, :, None]), axis=1)
        k_hat = np.concatenate((k[head], bk[head, :, None]), axis=1)
        v_hat = np.concatenate((v[head], bv[head, :, None]), axis=1)
        qk.append(product_svd(q_hat.T, k_hat))
        vo.append(product_svd(v_hat.T, wo[head]))
    if dimensions is None:
        q_dim, q_energy = retained_dimension([item[1] for item in qk], old_dim, rho)
        v_dim, v_energy = retained_dimension([item[1] for item in vo], old_dim, rho)
    else:
        q_dim, v_dim = dimensions
        if min(q_dim, v_dim) <= 0 or max(q_dim, v_dim) > old_dim:
            raise ValueError("Invalid explicit attention dimensions")
        def fraction(results, count):
            return float(np.mean([np.sum(s[:count].astype(np.float64)**2) / np.sum(s.astype(np.float64)**2)
                                  if np.sum(s.astype(np.float64)**2) > 0 else 1.0
                                  for _, s, _ in results]))
        q_energy, v_energy = fraction(qk, q_dim), fraction(vo, v_dim)
    scale = old_dim**-0.5 if original_scale is None else float(original_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid attention scale")
    q_scale = scale * math.sqrt(q_dim)
    queries, keys, values, outputs = [], [], [], []
    for (uq, sq, vhq), (uv, sv, vhv) in zip(qk, vo):
        root_q, root_v = np.sqrt(sq[:q_dim]), np.sqrt(sv[:v_dim])
        queries.append((uq[:, :q_dim] * root_q).T * q_scale)
        keys.append(root_q[:, None] * vhq[:q_dim])
        values.append((uv[:, :v_dim] * root_v).T)
        outputs.append(root_v[:, None] * vhv[:v_dim])
    packed = np.concatenate([np.concatenate(rows, axis=0) for rows in (queries, keys, values)], axis=0)
    return {
        "qkv_weight": np.ascontiguousarray(packed[:, :-1], dtype=weight.dtype),
        "qkv_bias": np.ascontiguousarray(packed[:, -1], dtype=weight.dtype) if has_bias else None,
        "out_weight": np.ascontiguousarray(np.concatenate(outputs, axis=0).T, dtype=weight.dtype),
        "out_bias": None if out_bias is None else np.array(out_bias, copy=True),
        "qk_dim": int(q_dim), "vo_dim": int(v_dim), "num_heads": int(heads),
        "original_head_dim": int(old_dim), "qk_energy_retained": q_energy,
        "vo_energy_retained": v_energy,
    }


def compensate_fc2(features, weights, kept):
    """NuWa's optimize strategy: Y.T @ pinv(H[:, kept]).T, with fixed fc2 bias."""
    h, w = np.asarray(features), np.asarray(weights)
    kept = np.asarray(kept, dtype=np.int64)
    if h.ndim != 2 or w.ndim != 2 or h.shape[1] != w.shape[1] or h.shape[0] == 0:
        raise ValueError("Invalid calibration features or fc2 weights")
    if kept.ndim != 1 or kept.size == 0 or np.unique(kept).size != kept.size or np.any(kept < 0) or np.any(kept >= h.shape[1]):
        raise ValueError("Invalid retained neuron indices")
    if not np.isfinite(h).all() or not np.isfinite(w).all():
        raise ValueError("Nonfinite features or MLP weights")
    hr = np.ascontiguousarray(h[:, kept])
    desired = h @ w.T
    # Same dtype-relative rank tolerance used by torch.linalg.pinv. This is
    # a numerical SVD cutoff, not a ridge regularizer or training objective.
    cutoff = max(hr.shape) * np.finfo(hr.dtype).eps
    inverse = np.linalg.pinv(hr, rcond=cutoff)
    result = np.ascontiguousarray(desired.T @ inverse.T, dtype=w.dtype)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite least-squares compensation")
    def mse(prediction):
        difference = prediction.astype(np.float64) - desired.astype(np.float64)
        return float(np.mean(difference**2))
    return result, {"calibration_mse_before": mse(hr @ w[:, kept].T),
                    "calibration_mse_after": mse(hr @ result.T),
                    "pinv_rcond": float(cutoff), "calibration_rows": int(h.shape[0])}


def _as_numpy(value):
    if value is None:
        return None
    value = value.detach().cpu()
    if value.dtype not in (torch.float32, torch.float64):
        value = value.float()
    return value.numpy()


@torch.no_grad()
def prune_attention(model, rho):
    results = []
    for index, block in enumerate(model.backbone.blocks):
        attn = block.attn
        if isinstance(attn, AttentionPruned):
            raise ValueError("OFP expects an SKP anchor, not an already pruned attention module")
        for name in ("q_norm", "k_norm", "norm"):
            module = getattr(attn, name, None)
            if module is not None and not isinstance(module, nn.Identity):
                raise ValueError("This OFP adapter expects ordinary ViTPose attention without {}".format(name))
        if getattr(attn, "use_rel_pos", False) or getattr(attn, "relative_position_bias_table", None) is not None:
            raise ValueError("Relative-position attention is outside this ordinary ViTPose adapter")
        factors = factor_attention(
            _as_numpy(attn.qkv.weight), _as_numpy(attn.qkv.bias),
            _as_numpy(attn.proj.weight), _as_numpy(attn.proj.bias), int(attn.num_heads), rho,
            original_scale=float(attn.scale),
        )
        spec = _attention_spec(attn, factors["qk_dim"], factors["vo_dim"])
        reduced = AttentionPruned(attn.qkv.in_features, **spec).to(
            device=attn.qkv.weight.device, dtype=attn.qkv.weight.dtype)
        for destination, source in ((reduced.qkv.weight, factors["qkv_weight"]),
                                    (reduced.qkv.bias, factors["qkv_bias"]),
                                    (reduced.proj.weight, factors["out_weight"]),
                                    (reduced.proj.bias, factors["out_bias"])):
            if destination is not None:
                destination.copy_(torch.from_numpy(np.ascontiguousarray(source)).to(destination))
        for original, replacement in ((attn.qkv, reduced.qkv), (attn.proj, reduced.proj)):
            replacement.weight.requires_grad_(original.weight.requires_grad)
            if replacement.bias is not None:
                replacement.bias.requires_grad_(original.bias.requires_grad)
        reduced.train(attn.training)
        block.attn = reduced
        result = dict(spec, original_head_dim=factors["original_head_dim"],
                      qk_energy_retained=factors["qk_energy_retained"],
                      vo_energy_retained=factors["vo_energy_retained"])
        results.append(result)
        print("Block {} attention: {} -> QK {}, VO {} per head".format(
            index, result["original_head_dim"], result["qk_dim"], result["vo_dim"]), flush=True)
    return results


@torch.no_grad()
def prune_mlps(model, target_widths, scores, feature_paths, strategy="optimize"):
    widths = hidden_sizes(model)
    if not (len(target_widths) == len(scores) == len(feature_paths) == len(widths)):
        raise ValueError("Expected one target, score vector and feature path per block")
    if strategy not in ("optimize", "prune"):
        raise ValueError("Use the existing optimize or prune MLP strategy")
    for index, (target, score, path, width) in enumerate(zip(target_widths, scores, feature_paths, widths)):
        if isinstance(target, bool) or not isinstance(target, Integral) or not 1 <= target <= width:
            raise ValueError("Invalid target MLP width in block {}".format(index))
        if np.asarray(score).shape != (width,) or not np.isfinite(score).all():
            raise ValueError("Invalid activation scores in block {}".format(index))
        if target < width and strategy == "optimize" and (path is None or not Path(path).is_file()):
            raise ValueError("Missing calibration features in block {}".format(index))
    reports = []
    for index, (block, target, score, path) in enumerate(zip(
        model.backbone.blocks, target_widths, scores, feature_paths
    )):
        mlp = block.mlp
        old_width = mlp.fc1.out_features
        prune_count = old_width - target
        if prune_count == 0:
            reports.append({"block": index, "hidden_before": old_width, "hidden_after": target,
                            "kept_indices": list(range(old_width)), "compensated": False})
            continue
        removed = torch.topk(torch.from_numpy(score), k=prune_count, largest=False).indices
        keep_mask = torch.ones(old_width, dtype=torch.bool)
        keep_mask[removed] = False
        kept = torch.nonzero(keep_mask, as_tuple=False).flatten()
        if strategy == "optimize":
            features = np.load(path, mmap_mode="r")
            try:
                new_weights, diagnostics = compensate_fc2(
                    features.reshape(-1, old_width), mlp.fc2.weight.detach().cpu().numpy(), kept.numpy(),
                )
            finally:
                features._mmap.close()
        elif strategy == "prune":
            new_weights = mlp.fc2.weight.detach().cpu().numpy()[:, kept.numpy()].copy()
            diagnostics = {}
        else:
            raise ValueError("Unknown MLP strategy")
        fc1 = _linear_like(mlp.fc1, mlp.fc1.in_features, target)
        fc2 = _linear_like(mlp.fc2, target, mlp.fc2.out_features)
        device_indices = kept.to(mlp.fc1.weight.device)
        fc1.weight.copy_(mlp.fc1.weight.index_select(0, device_indices))
        if fc1.bias is not None:
            fc1.bias.copy_(mlp.fc1.bias.index_select(0, device_indices))
        fc2.weight.copy_(torch.from_numpy(new_weights).to(fc2.weight))
        if fc2.bias is not None:
            fc2.bias.copy_(mlp.fc2.bias)
        mlp.fc1, mlp.fc2 = fc1, fc2
        reports.append(dict(diagnostics, block=index, hidden_before=old_width,
                            hidden_after=target, kept_indices=kept.tolist(), compensated=strategy == "optimize"))
        print("Block {} MLP: {} -> {} ({})".format(index, old_width, target, strategy), flush=True)
    return reports


@torch.no_grad()
def prune_W2(args, X, W2, pruned_indices):
    """NuWa's fc2 slicing or Moore-Penrose reconstruction, without mutation."""
    if not torch.is_tensor(W2) or W2.ndim != 2:
        raise ValueError("W2 must be a two-dimensional tensor")
    removed = torch.as_tensor(pruned_indices).detach().cpu()
    if removed.ndim != 1 or (removed.numel() and removed.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError("Expected one-dimensional integer neuron indices")
    removed = removed.long()
    if (removed.numel() != removed.unique().numel() or torch.any(removed < 0).item()
            or torch.any(removed >= W2.shape[1]).item()):
        raise ValueError("Invalid removed neuron indices")
    keep_mask = torch.ones(W2.shape[1], dtype=torch.bool)
    keep_mask[removed] = False
    kept = torch.nonzero(keep_mask, as_tuple=False).flatten()
    if kept.numel() == 0:
        raise ValueError("At least one MLP neuron must remain")
    strategy = getattr(args, "strategy", "optimize")
    if strategy not in ("optimize", "prune"):
        raise ValueError("Use the existing optimize or prune MLP strategy")
    if strategy == "prune" or removed.numel() == 0:
        return W2.index_select(1, kept.to(W2.device)).clone()
    features = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    if features.ndim not in (2, 3) or features.shape[-1] != W2.shape[1]:
        raise ValueError("Calibration features must end in the original MLP width")
    weights, _ = compensate_fc2(features.reshape(-1, W2.shape[1]), _as_numpy(W2), kept.numpy())
    return torch.from_numpy(weights).to(W2)


@torch.no_grad()
def prune_mlp(args, model_anchor, prune_neuron_count):
    """Collect training features and physically prune the selected MLP neurons."""
    args = calibration_options(args)
    widths = hidden_sizes(model_anchor)
    # Reject an impossible budget before reading any training images.
    targets = uniform_widths(widths, prune_neuron_count)
    if targets == widths:
        return model_anchor
    with tempfile.TemporaryDirectory(prefix="nuwa_mlp_") as directory:
        scores, paths = get_activation_and_calibration_features(
            args, model_anchor, train_loader=getattr(args, "train_loader", None), directory=directory,
        )
        targets, _ = get_config_and_indices(args, model_anchor, prune_neuron_count, scores)
        prune_mlps(model_anchor, targets, scores, paths, args.strategy)
    return model_anchor


@torch.no_grad()
def prune_mha(args, model_anchor):
    """Prune QK/VO dimensions with NuWa's bias-augmented attention SVD."""
    rho = getattr(args, "energy_rate", None)
    if rho is None:
        rho = energy_ratio(args.pruning_rate)
    return prune_attention(model_anchor, rho)


@torch.no_grad()
def pruning_vit(args, model_anchor, original_size, mha=True, mlp=True):
    """Prune a full pose model using the original NuWa function signature.

    original_size is either a profile_pose_macs result or billions of matrix
    MACs (the original NuWa get_gflops counting convention). Include the pose
    head in that baseline. The CLI uses integer MACs throughout its reports.
    """
    args = calibration_options(args)
    rate = float(args.pruning_rate)
    energy_ratio(rate)  # Validate the budget before changing any layers.
    original_macs = (original_size["pose_macs"] if isinstance(original_size, dict)
                     else float(original_size) * 1e9)
    if not math.isfinite(original_macs) or original_macs <= 0:
        raise ValueError("original_size must describe a positive full-pose MAC count")
    model_anchor.to(args.device).eval()
    probe = getattr(args, "probe", None)
    if probe is None:
        cfg = getattr(args, "cfg", None)
        if cfg is None:
            from mmcv import Config
            cfg = Config.fromfile(str(resolve_input_file(args.config, "--config")))
        width, height = map(int, cfg.data_cfg.image_size)
        probe = torch.zeros(1, 3, height, width)
    probe = probe.to(device=args.device, dtype=next(model_anchor.parameters()).dtype)
    if mha:
        prune_mha(args, model_anchor)
    profile, _ = profile_pose_macs(model_anchor, probe, forward_heatmaps)
    unit_costs = profile["per_neuron_macs"]
    if not unit_costs or len(set(unit_costs)) != 1 or unit_costs[0] <= 0:
        raise ValueError("Uniform OFP requires the same MLP neuron cost in every ViT block")
    target_macs = original_macs * (1 - rate)
    count = max(0, int((profile["pose_macs"] - target_macs) / unit_costs[0]))
    if count and mlp:
        prune_mlp(args, model_anchor, count)
    return model_anchor.eval()


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--anchor", help="SKP anchor.pth; normal full NuWa")
    source.add_argument("--base-checkpoint", help="Unpruned checkpoint; no-SKP ablation")
    parser.add_argument("--pruning-rate", required=True, type=float,
                        help="Target fraction of full-pose matrix MACs removed, e.g. 0.4")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--species", help="Default: the anchor's species (dog)")
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3), help="Default: anchor split, or 1 for a base checkpoint")
    parser.add_argument("--strategy", choices=("optimize", "prune"), default="optimize",
                        help="optimize is vanilla OFP; prune disables reconstruction for ablation")
    parser.add_argument("--skip-attention", action="store_true", help="MLP-only ablation; skip attention SVD")
    parser.add_argument("--calibration-samples", type=int, default=128)
    add_device_argument(parser)
    parser.add_argument("--out-dir", help="Default: an ofp_<rate> folder beside anchor.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Only two training crops; marks output as a smoke test")
    parser.add_argument("--threads", type=int, default=1)
    return parser


def run(args, parser=None):
    parser = parser or get_args_parser()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        rho = energy_ratio(args.pruning_rate)
        config_path = resolve_input_file(args.config, "--config")
        anchor_path = resolve_input_file(args.anchor or args.base_checkpoint, "--anchor/--base-checkpoint")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.calibration_samples <= 0:
        parser.error("calibration-samples must be positive")
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    args.device = str(device)
    name = "ofp_" + format(args.pruning_rate, "g").replace(".", "p")
    if args.base_checkpoint:
        name += "_base"
    if args.skip_attention:
        name += "_mlponly"
    if args.strategy == "prune":
        name += "_prune"
    if args.smoke_test:
        name += "_smoke"
    slug = re.sub(r"[^a-z0-9]+", "_", (args.species or DEFAULT_SPECIES).casefold()).strip("_")
    default_parent = DEFAULT_OUTPUT_ROOT / "nuwa_ablations" / slug if args.base_checkpoint else anchor_path.parent
    output_dir = resolve_project_path(args.out_dir) if args.out_dir else default_parent / name
    if output_dir.exists():
        parser.error("Output folder already exists; choose another --out-dir")
    seed_all(args.seed)

    from mmcv import Config
    from mmpose.models import build_posenet

    cfg = Config.fromfile(str(config_path))
    width, height = map(int, cfg.data_cfg.image_size)
    probe = torch.randn(1, 3, height, width, generator=torch.Generator().manual_seed(args.seed)).to(device)
    print("Profiling the original full-width pose architecture...", flush=True)
    dense = build_posenet(cfg.model).to(device).eval()
    original_hidden = hidden_sizes(dense)
    original_profile, _ = profile_pose_macs(dense, probe, forward_heatmaps)
    del dense
    gc.collect()
    if args.anchor:
        model, anchor = load_anchor_model(cfg, anchor_path, device)
        if anchor.get("stage") != ANCHOR_STAGE:
            parser.error("--anchor must be an SKP anchor.pth")
    else:
        model, source_info = load_pose_checkpoint(cfg, anchor_path, device)
        if source_info.get("stage") != BASE_STAGE:
            parser.error("--base-checkpoint must be the original unpruned split checkpoint")
        anchor = {"stage": BASE_STAGE, "species": args.species or DEFAULT_SPECIES,
                  "split": args.split_id or 1, "base_checkpoint_sha256": file_sha256(anchor_path),
                  "architecture": {"original_hidden_sizes": original_hidden}, "skp_applied": False}
    species = args.species or anchor.get("species")
    if not isinstance(species, str) or not isinstance(anchor.get("species"), str) or species.strip().casefold() != anchor["species"].strip().casefold():
        parser.error("Species must match the anchor's species")
    split_id = anchor.get("split")
    if split_id not in (1, 2, 3):
        parser.error("Invalid split ID in anchor checkpoint")
    if args.split_id is not None and args.split_id != split_id:
        parser.error("split-id must match the anchor's training split")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    anchor_profile, _ = profile_pose_macs(model, probe, forward_heatmaps)
    anchor_widths = hidden_sizes(model)
    protected_keys = [key for key in model.state_dict() if not any(
        key == "backbone.blocks.{}.{}".format(index, suffix)
        for index in range(len(anchor_widths))
        for suffix in ("attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
                       "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight"))]
    protected_hash = frozen_state_sha256(model, protected_keys)
    print("Attention SVD: {} | MLP strategy: {}".format(not args.skip_attention, args.strategy), flush=True)
    attention_report = prune_attention(model, rho) if not args.skip_attention else []
    mha_profile, _ = profile_pose_macs(model, probe, forward_heatmaps)
    unit_costs = mha_profile["per_neuron_macs"]
    if not unit_costs or len(set(unit_costs)) != 1 or unit_costs[0] <= 0:
        parser.error("Uniform OFP requires the same MLP neuron cost in every ViT block")
    target_macs = original_profile["pose_macs"] * (1 - args.pruning_rate)
    # Preserve the supplied pruning_vit's integer budget convention.
    prune_count = max(0, int((mha_profile["pose_macs"] - target_macs) / unit_costs[0]))
    try:
        targets = uniform_widths(anchor_widths, prune_count) if prune_count else anchor_widths
    except ValueError as error:
        parser.error(str(error))
    print("Target: {:.2f}% full-pose matrix MAC reduction".format(100 * args.pruning_rate), flush=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    calibration = {"training_crops_processed": 0, "calibration_crops": 0,
                   "reason": "Attention/SKP already meets the integer MLP budget"}
    mlp_report = []
    with tempfile.TemporaryDirectory(prefix="ofp_work_", dir=str(output_dir)) as temporary:
        temporary_dir = Path(temporary)
        if prune_count:
            scores, paths, calibration = collect_training_features(
                cfg, model, args, species, split_id, mha_profile["tokens_per_block"], temporary_dir,
            )
            np.savez_compressed(output_dir / "activation_scores.npz",
                                **{"block_{}".format(i): score for i, score in enumerate(scores)})
            mlp_report = prune_mlps(model, targets, scores, paths, args.strategy)
        if frozen_state_sha256(model, protected_keys) != protected_hash:
            raise RuntimeError("A protected pose-head, normalization, residual or bias tensor changed")
        final_profile, expected = profile_pose_macs(model, probe, forward_heatmaps)
        expected_shape = (1, int(cfg.data_cfg.num_joints),
                          int(cfg.data_cfg.heatmap_size[1]), int(cfg.data_cfg.heatmap_size[0]))
        if tuple(expected.shape) != expected_shape or not torch.isfinite(expected).all().item():
            raise RuntimeError("The OFP model did not produce finite pose heatmaps")
        predicted_savings = (sum(anchor_widths) - sum(targets)) * unit_costs[0]
        if mha_profile["pose_macs"] - final_profile["pose_macs"] != predicted_savings:
            raise RuntimeError("Measured MLP MAC savings disagree with its physical widths")
        achieved = 1 - final_profile["pose_macs"] / original_profile["pose_macs"]
        report = {
            "stage": OFP_STAGE, "species": species, "category_id": anchor.get("category_id"),
            "split": split_id, "config_path": str(config_path),
            "source_anchor": str(anchor_path) if args.anchor else None,
            "source_anchor_sha256": file_sha256(anchor_path) if args.anchor else None,
            "base_checkpoint_sha256": anchor.get("base_checkpoint_sha256"),
            "skp_applied": bool(args.anchor), "attention_svd_applied": not args.skip_attention,
            "mlp_strategy": args.strategy, "created": datetime.now().isoformat(),
            "target_pruning_rate": args.pruning_rate, "achieved_pruning_rate": achieved,
            "rho": rho, "budget_scope": "complete pose model matrix MACs", "device": str(device),
            "source_mask_val_ap": anchor.get("source_mask_val_ap"),
            "baseline_val_ap": anchor.get("baseline_val_ap"),
            "source_smoke_test": bool(anchor.get("source_smoke_test", False)),
            "smoke_test": args.smoke_test, "calibration": calibration,
            "original_profile": original_profile, "anchor_profile": anchor_profile,
            "after_attention_profile": mha_profile, "final_profile": final_profile,
            "anchor_hidden_sizes": anchor_widths, "hidden_sizes": hidden_sizes(model),
            "attention": attention_report, "mlp": mlp_report,
            "protected_state_unchanged": True, "heatmap_shape": list(expected_shape),
            "edge_val_ap": None,
        }
        payload = dict(report, format_version=1,
                       architecture={"original_hidden_sizes": anchor["architecture"]["original_hidden_sizes"],
                                     "hidden_sizes": hidden_sizes(model), "attention": attention_specs(model)},
                       state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
        candidate = temporary_dir / "edge.pth"
        atomic_torch_save(payload, candidate)
        restored, _ = load_nuwa_model(cfg, candidate, device)
        actual = forward_heatmaps(restored, probe)
        if not torch.isfinite(actual).all().item() or not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
            raise RuntimeError("The reloaded OFP checkpoint changed its heatmaps")
        report["checkpoint_reload_max_abs_error"] = float((actual - expected).abs().max().item())
        report["standalone_checkpoint_reload"] = True
        os.replace(str(candidate), str(output_dir / "edge.pth"))
        (output_dir / "ofp_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Pose matrix MACs: {:.4f}G -> {:.4f}G ({:.2f}% removed; target {:.2f}%)".format(
        original_profile["pose_macs"] / 1e9, final_profile["pose_macs"] / 1e9,
        100 * achieved, 100 * args.pruning_rate), flush=True)
    print("Pose parameters: {:,} -> {:,}".format(original_profile["parameters"], final_profile["parameters"]))
    print("Finite 17-keypoint heatmaps, protected weights and checkpoint reload: PASS")
    if args.smoke_test or report["source_smoke_test"]:
        print("Smoke-test result: use a fully trained anchor and a full OFP run for AP comparisons.")
    print("Checkpoint:", output_dir / "edge.pth")
    print("Report:", output_dir / "ofp_report.json")
    print("OFP passed. Use the existing evaluation scripts with this checkpoint to measure AP.")
    return report


def main(argv=None):
    parser = get_args_parser()
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
