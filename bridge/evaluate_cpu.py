"""Evaluate the unpruned split ViTPose model on AP-10K using CPU.

Keep this file, check_split.py and bridge_paths.py in bridge/.
Run from the project root:

    python bridge/evaluate_cpu.py
    python bridge/evaluate_cpu.py --species all --split test

The default is DOG VALIDATION, using the annotation subset from step 1.
Use --species all for the original complete validation/test annotations.
Species-specific test annotations must be provided separately: step 1 only
creates train/validation subsets. --ann-file can select a custom annotation
file; it must contain only the requested species unless --species all is used.

The default data root is bridge/data/ap10k, containing annotations/ and data/.
Configs use vitpose/configs/. The checkpoint is bridge/checkpoints/split/ap10k.pth.
Local mmpose imports work through bridge_paths.py without an editable install.

The model configuration's preprocessing, flip testing, bounding-box settings
and official dataset.evaluate(..., metric='mAP') scoring are preserved.
Predictions and metrics are saved under bridge/work_dirs/<species>_baseline_cpu/.
This evaluates an ordinary full-weight checkpoint; SKP mask-only files are
not standalone models and are not accepted as --checkpoint.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re

if __package__:
    from .bridge_paths import (
        DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_SPECIES, resolve_input_file,
    )
    from .check_split import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_pose_model
else:
    from bridge_paths import (
        DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_SPECIES, resolve_input_file,
    )
    from check_split import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_pose_model


def evaluation_paths(data_root, species, split, split_id, ann_file=None):
    """Validate the selected dataset without loading images."""
    data_root = Path(data_root)
    species = species.strip()
    if not species:
        raise ValueError("Species cannot be empty; use dog or all")
    all_species = species.casefold() == "all"
    slug = "all" if all_species else re.sub(r"[^a-z0-9]+", "_", species.casefold()).strip("_")
    if not slug:
        raise ValueError("Use a species name from prepare_species.py --list")

    filename = "ap10k-{}-split{}.json".format(split, split_id)
    if ann_file is not None:
        annotation_path = Path(ann_file)
    elif all_species:
        annotation_path = data_root / "annotations" / filename
    else:
        annotation_path = data_root / "annotations" / "species" / slug / filename
    if not annotation_path.is_file():
        if all_species or ann_file is not None:
            raise ValueError("Annotation file not found: {}".format(annotation_path))
        if split == "test":
            raise ValueError(
                "Species test annotations not found: {}. Step 1 creates only "
                "train/val subsets. To reproduce the complete test benchmark, "
                "use --species all --split test.".format(annotation_path)
            )
        raise ValueError(
            "Species subset not found: {}. Run bridge/prepare_species.py "
            "with this --data-root and --species first.".format(annotation_path)
        )

    image_root = data_root / "data"
    if not image_root.is_dir():
        raise ValueError("Image directory not found: {}".format(image_root))
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        annotations = json.load(handle)
    for key in ("images", "annotations", "categories"):
        if not isinstance(annotations.get(key), list) or not annotations[key]:
            raise ValueError("{} needs a nonempty {} list".format(annotation_path, key))
    if not all_species:
        matches = [category["id"] for category in annotations["categories"]
                   if category["name"].strip().casefold() == species.casefold()]
        if len(matches) != 1:
            raise ValueError("Species {!r} is not unique in the annotations".format(species))
        if any(annotation["category_id"] != matches[0]
               for annotation in annotations["annotations"]):
            raise ValueError(
                "The selected annotations include other species. Use a filtered "
                "subset for {!r}, or --species all for all species.".format(species)
            )
    image_ids = {image["id"] for image in annotations["images"]}
    if any(annotation["image_id"] not in image_ids for annotation in annotations["annotations"]):
        raise ValueError("Some annotations refer to missing image records")
    for image in annotations["images"]:
        if not (image_root / image["file_name"]).is_file():
            raise ValueError("Image not found: {}".format(image_root / image["file_name"]))
    return annotation_path, image_root, slug


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="AP-10K folder; default: bridge/data/ap10k")
    parser.add_argument("--species", default=DEFAULT_SPECIES,
                        help="One species, or all; default: dog")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--ann-file", help="Optional prefiltered annotation file")
    parser.add_argument("--out-dir", help="Output folder; relative to the project root")
    args = parser.parse_args()

    try:
        config_path = resolve_input_file(args.config, "--config")
        checkpoint_path = resolve_input_file(args.checkpoint, "--checkpoint")
    except FileNotFoundError as error:
        parser.error(str(error))

    import mmcv
    import torch
    from mmcv import Config
    from mmcv.parallel import scatter
    from mmpose.datasets import build_dataset, build_dataloader

    cfg = Config.fromfile(str(config_path))
    data_root = Path(args.data_root)
    try:
        annotation_path, image_root, slug = evaluation_paths(
            data_root, args.species, args.split, args.split_id, args.ann_file,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    dataset_cfg = cfg.data[args.split]
    dataset_cfg.test_mode = True
    dataset_cfg.ann_file = str(annotation_path)
    dataset_cfg.img_prefix = str(image_root) + os.sep
    dataset = build_dataset(dataset_cfg)
    if len(dataset) == 0:
        parser.error("No usable labeled animals remain after dataset filtering")
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False,
    )
    model = load_pose_model(cfg, checkpoint_path)
    output_dir = Path(args.out_dir) if args.out_dir else (
        DEFAULT_OUTPUT_ROOT / "{}_baseline_cpu".format(slug)
        / "{}_split{}".format(args.split, args.split_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Species: {} | Split: {} (split {}) | Animals: {}".format(
        args.species, args.split, args.split_id, len(dataset)), flush=True)
    print("Annotations:", annotation_path, flush=True)
    print("Checkpoint:", checkpoint_path, flush=True)
    print("Output:", output_dir, flush=True)

    outputs = []
    progress = mmcv.ProgressBar(len(dataset))
    with torch.no_grad():
        for batch in loader:
            batch = scatter(batch, [-1])[0]
            outputs.append(model(return_loss=False, **batch))
            progress.update()

    mmcv.dump(outputs, str(output_dir / "predictions.pkl"))
    metrics = dataset.evaluate(outputs, res_folder=str(output_dir), metric="mAP")
    metrics = {name: float(value) for name, value in metrics.items()}
    if not math.isfinite(metrics["AP"]) or not 0 <= metrics["AP"] <= 1:
        raise RuntimeError("The evaluator returned an invalid AP")
    report = {
        "species": args.species, "split": args.split, "split_id": args.split_id,
        "config": str(config_path), "checkpoint": str(checkpoint_path),
        "annotations": str(annotation_path), "image_root": str(image_root),
        "num_animals": len(dataset), "metrics": metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("\nEvaluation results:")
    for name, value in metrics.items():
        print("{}: {:.4f}".format(name, value))
    print("AP points: {:.2f}".format(100 * metrics["AP"]))


if __name__ == "__main__":
    main()
