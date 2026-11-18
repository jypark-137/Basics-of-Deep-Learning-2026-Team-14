"""Official AP-10K evaluation for baseline, SKP-anchor and OFP checkpoints."""


import argparse
import math
import os
from pathlib import Path
import tempfile
import torch
from ..paths import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPECIES, resolve_data_root, resolve_input_file, resolve_project_path,
)
from ..dataset.utils import evaluation_paths, move_batch
from ..model.checkpoint import BASE_STAGE, load_pose_checkpoint
from device_utils import add_device_argument, resolve_device
from ..utils.io import atomic_json

@torch.no_grad()
def evaluate_metrics(model, test_loader, device=None, output_dir=None, progress=True, save_predictions=True):
    """Run the existing model inference API and the dataset's official mAP."""
    import mmcv

    if test_loader is None or len(test_loader) == 0 or len(test_loader.dataset) == 0:
        raise ValueError("Evaluation requires a nonempty pose validation/test loader")
    device = resolve_device(device, model)
    model.to(device).eval()
    outputs = []
    indicator = mmcv.ProgressBar(len(test_loader.dataset)) if progress else None
    for batch in test_loader:
        data = move_batch(batch, device)
        outputs.append(model(return_loss=False, **data))
        if indicator is not None:
            indicator.update()

    def score(directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if output_dir is not None and save_predictions:
            mmcv.dump(outputs, str(directory / "predictions.pkl"))
        metrics = test_loader.dataset.evaluate(outputs, res_folder=str(directory), metric="mAP")
        metrics = {name: float(value) for name, value in metrics.items()}
        if "AP" not in metrics or not math.isfinite(metrics["AP"]) or not 0 <= metrics["AP"] <= 1:
            raise RuntimeError("The pose evaluator returned an invalid AP")
        return metrics

    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="nuwa_eval_") as directory:
            return score(directory)
    return score(output_dir)


@torch.no_grad()
def evaluate(model, test_loader, visual_task="pose", sub_label=None, *, device=None,
             output_dir=None, return_metrics=False, progress=True):
    """Return pose AP points, or the complete fractional metric dictionary."""
    if visual_task not in ("pose", "pose_estimation", "keypoint"):
        raise ValueError("This ViTPose evaluator supports pose/keypoint tasks")
    metrics = evaluate_metrics(model, test_loader, device, output_dir, progress)
    return metrics if return_metrics else 100 * metrics["AP"]


@torch.no_grad()
def evaluate_ap(model, loader, device, output_dir):
    """Fractional AP for SKP checkpoint selection, using the common evaluator."""
    metrics = evaluate_metrics(model, loader, device, output_dir, save_predictions=False)
    print("\nValidation AP: {:.2f}".format(100 * metrics["AP"]), flush=True)
    return metrics["AP"]



def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Unpruned split weights, SKP anchor.pth, or OFP edge.pth")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--species", help="Default: checkpoint species, or dog for a baseline")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3))
    parser.add_argument("--ann-file", help="Optional prefiltered annotation file")
    parser.add_argument("--out-dir", help="Default: outputs/nuwa_eval/<checkpoint>/<species>/<split>")
    add_device_argument(parser)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads; default: 1")
    return parser


def run(args, parser=None):
    parser = parser or get_args_parser()
    if args.threads <= 0:
        parser.error("threads must be positive")
    try:
        config_path = resolve_input_file(args.config, "--config")
        checkpoint_path = resolve_input_file(args.checkpoint, "--checkpoint")
        device = resolve_device(args.device)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    from mmcv import Config
    from mmpose.datasets import build_dataset, build_dataloader

    cfg = Config.fromfile(str(config_path))
    model, info = load_pose_checkpoint(cfg, checkpoint_path, device)
    stage = info.get("stage", BASE_STAGE)
    species = args.species or info.get("species") or DEFAULT_SPECIES
    if not isinstance(species, str) or not species.strip():
        parser.error("Use a nonempty species name, or all")
    if stage != BASE_STAGE and str(info.get("species", "")).strip().casefold() != species.strip().casefold():
        parser.error("--species must match the species used to learn this checkpoint")
    split_id = args.split_id or info.get("split", 1)
    if split_id not in (1, 2, 3):
        parser.error("Invalid split ID in checkpoint")
    data_root = resolve_data_root(args.data_root)
    try:
        annotation_path, image_root, slug = evaluation_paths(
            data_root, species, args.split, split_id, args.ann_file,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    dataset_cfg = cfg.data[args.split]
    dataset_cfg.test_mode = True
    dataset_cfg.ann_file, dataset_cfg.img_prefix = str(annotation_path), str(image_root) + os.sep
    dataset = build_dataset(dataset_cfg)
    if len(dataset) == 0:
        parser.error("No usable labeled animals remain after dataset filtering")
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False, pin_memory=device.type == "cuda")
    output_dir = resolve_project_path(args.out_dir) if args.out_dir else (
        DEFAULT_OUTPUT_ROOT / "nuwa_eval" / checkpoint_path.parent.name / checkpoint_path.stem
        / slug / "{}_split{}".format(args.split, split_id)
    )
    print("Species: {} | Split: {} (split {}) | Animals: {}".format(
        species, args.split, split_id, len(dataset)), flush=True)
    print("Checkpoint:", checkpoint_path, "| Device:", device, flush=True)
    metrics = evaluate_metrics(model, loader, device, output_dir)
    report = {
        "checkpoint": str(checkpoint_path), "config": str(config_path),
        "species": species, "split": args.split, "split_id": split_id,
        "annotations": str(annotation_path), "image_root": str(image_root),
        "num_animals": len(dataset), "device": str(device), "stage": stage,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "source_mask_val_ap": info.get("source_mask_val_ap"),
        "baseline_val_ap": info.get("baseline_val_ap"), "metrics": metrics,
        "source_smoke_test": bool(info.get("source_smoke_test", False)),
        "smoke_test": bool(info.get("smoke_test", False)),
        "target_pruning_rate": info.get("target_pruning_rate"),
        "achieved_pruning_rate": info.get("achieved_pruning_rate"),
        "compute_profile": info.get("final_profile"),
    }
    atomic_json(output_dir / "metrics.json", report)
    print("\nAP points: {:.2f}".format(100 * metrics["AP"]))
    if args.split == "val" and split_id == info.get("split") and args.ann_file is None:
        for label, key in (("Saved masked", "source_mask_val_ap"), ("Unpruned baseline", "baseline_val_ap")):
            if report[key] is not None:
                print("{} AP: {:.2f}".format(label, 100 * report[key]))
    print("Results:", output_dir)
    return report


def main(argv=None):
    parser = get_args_parser()
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
