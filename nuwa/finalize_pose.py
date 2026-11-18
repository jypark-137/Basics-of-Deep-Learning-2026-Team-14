"""Portable model exports, common AP evaluation, CPU/CUDA timing and comparison reports."""


import argparse
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from .paths import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPECIES, PACKAGE_ROOT, PROJECT_ROOT, VITPOSE_ROOT, resolve_input_file,
    resolve_project_path,
)
from .dataset.utils import evaluation_paths, move_batch
from .model.vit import forward_heatmaps
from .model.utils import config_snapshot
from device_utils import add_device_argument, device_info, resolve_device
from .model.checkpoint import BASE_STAGE, OFP_STAGE, checkpoint_header, export_portable, load_pose_checkpoint, verify_reload
from .utils.utils import MAC_CONVENTION, benchmark_pose, profile_pose_macs
from .utils.io import atomic_json, atomic_text, file_sha256, fingerprint, state_size_bytes
from .utils.pose_results import parse_edges, validate_provenance, write_comparison

def code_fingerprint():
    files = list(PACKAGE_ROOT.glob("*.py"))
    files.append(PROJECT_ROOT / "device_utils.py")
    files.append(PROJECT_ROOT / "main.py")
    for name in ("engine", "method", "model", "dataset", "data", "utils", "tests"):
        files.extend((PACKAGE_ROOT / name).glob("*.py"))
    files.extend(VITPOSE_ROOT.joinpath("mmpose").rglob("*.py"))
    return fingerprint({str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
                        for path in sorted(files)})



def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Original split baseline")
    parser.add_argument("--anchor", required=True, help="SKP anchor anchor.pth")
    parser.add_argument("--edge", required=True, nargs="+", help="One or more PATH or NAME=PATH values")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--species", default=DEFAULT_SPECIES)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3))
    parser.add_argument("--ann-file", help="Optional common prefiltered annotation file")
    add_device_argument(parser)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", help="Default: a new outputs/final_compare run")
    parser.add_argument("--resume", action="store_true", help="Resume matching results in --out-dir")
    parser.add_argument("--smoke-test", action="store_true", help="Skip AP; one warmup and two timings")
    return parser


def run(args, parser=None):
    parser = parser or get_args_parser()
    if args.threads <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("threads/repeats must be positive and warmup nonnegative")
    if args.smoke_test:
        args.warmup, args.repeats = 1, 2
    try:
        device = resolve_device(args.device)
        config_path = resolve_input_file(args.config, "--config")
        cases = [{"label": "baseline", "checkpoint": args.checkpoint, "role": "baseline"},
                 {"label": "skp_anchor", "checkpoint": args.anchor, "role": "anchor"}]
        cases += [dict(case, role="edge") for case in parse_edges(args.edge)]
        for case in cases:
            case["checkpoint"] = str(resolve_input_file(case["checkpoint"], case["label"]))
            case["sha256"] = file_sha256(case["checkpoint"])
    except (OSError, ValueError) as error:
        parser.error(str(error))

    import cv2
    import mmcv
    import numpy as np
    import torch
    from mmcv import Config
    from mmpose.datasets import build_dataset, build_dataloader

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = Config.fromfile(str(config_path))
    base_header = checkpoint_header(cases[0]["checkpoint"])
    anchor_header = checkpoint_header(cases[1]["checkpoint"])
    base_ids = {value for value in (cases[0]["sha256"], base_header.get("base_checkpoint_sha256"),
                base_header.get("source_checkpoint_sha256"), base_header.get("origin_checkpoint_sha256")) if value}
    anchor_ids = {value for value in (cases[1]["sha256"], anchor_header.get("source_checkpoint_sha256"),
                  anchor_header.get("origin_checkpoint_sha256")) if value}
    reference_hash = base_header.get("base_checkpoint_sha256", cases[0]["sha256"])
    split_id = args.split_id or anchor_header.get("split", 1)
    if split_id not in (1, 2, 3):
        parser.error("Invalid split ID")
    for case in cases:
        case["header"] = checkpoint_header(case["checkpoint"])
        try:
            validate_provenance(case["header"], case["role"], args.species, split_id, base_ids, anchor_ids)
        except ValueError as error:
            parser.error("{}: {}".format(case["label"], error))

    dataset, loader, annotation_path = None, None, None
    if args.smoke_test:
        width, height = map(int, cfg.data_cfg.image_size)
        images = torch.randn(1, 3, height, width, generator=torch.Generator().manual_seed(args.seed))
    else:
        try:
            annotation_path, image_root, _ = evaluation_paths(
                Path(args.data_root), args.species, args.split, split_id, args.ann_file,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        data_cfg = cfg.data[args.split]
        data_cfg.test_mode = True
        data_cfg.ann_file = str(annotation_path)
        data_cfg.img_prefix = str(image_root) + os.sep
        dataset = build_dataset(data_cfg)
        if len(dataset) == 0:
            parser.error("No usable validation crops")
        loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                                  dist=False, shuffle=False, pin_memory=device.type == "cuda")
        images = move_batch(next(iter(loader)), device)["img"]
    if not torch.is_tensor(images) or images.shape[0] != 1:
        parser.error("Expected one preprocessed image tensor per batch")
    images = images.detach().to(device).contiguous()
    snapshot = config_snapshot(cfg)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": np.__version__, "mmcv": mmcv.__version__, "opencv": cv2.__version__,
                   "platform": platform.platform(), "processor": platform.processor() or "unreported",
                   "machine": platform.machine(), "logical_CPU_count": os.cpu_count(),
                   "device": device_info(device),
                   "CPU_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()}
    protocol = {"schema_version": 2, "device": str(device), "species": args.species, "split": args.split, "split_id": split_id,
                "annotation_file": str(annotation_path) if annotation_path else None,
                "annotation_sha256": file_sha256(annotation_path) if annotation_path else None,
                "dataset_crops": len(dataset) if dataset is not None else 0,
                "config_sha256": hashlib.sha256(snapshot.encode("utf-8")).hexdigest(), "code_sha256": code_fingerprint(),
                "input_shape": list(images.shape), "image_source": "synthetic" if args.smoke_test else "first validation crop",
                "metric": "official AP-10K mAP", "AP_flip_test": cfg.model.get("test_cfg", {}).get("flip_test", False),
                "latency_flip_test": False, "warmup": args.warmup, "repeats": args.repeats,
                "smoke_test": args.smoke_test, "seed": args.seed, "environment": environment,
                "MAC_convention": MAC_CONVENTION}
    signature = fingerprint({"protocol": protocol, "cases": cases})
    output_dir = resolve_project_path(args.out_dir) if args.out_dir else (
        DEFAULT_OUTPUT_ROOT / "final_compare" / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    manifest = {"schema_version": 1, "fingerprint": signature, "protocol": protocol, "cases": cases}
    if output_dir.exists():
        if not args.resume:
            parser.error("Output folder exists; use --resume with the same settings, or another --out-dir")
        prior = output_dir / "run_manifest.json"
        if not prior.is_file() or json.loads(prior.read_text(encoding="utf-8")).get("fingerprint") != signature:
            parser.error("Resume settings, source files, code or device do not match this run")
    else:
        output_dir.mkdir(parents=True)
        atomic_json(output_dir / "run_manifest.json", manifest)
        atomic_text(output_dir / "config_snapshot.py", snapshot)
    rows = []
    # Profile/time every case before AP so long evaluation passes do not
    # interleave with the benchmark series. Timers never include profiling hooks.
    for case in cases:
        label = case["label"]
        result_path = output_dir / "results" / (label + ".json")
        export_path = output_dir / "models" / (label + ".pth")
        if args.resume and result_path.is_file():
            row = json.loads(result_path.read_text(encoding="utf-8"))
            if row.get("run_fingerprint") != signature or not export_path.is_file() or file_sha256(export_path) != row.get("export_sha256"):
                parser.error("Invalid cached result for {}; choose a new output folder".format(label))
            rows.append(row)
            print("Reusing profile:", label, flush=True)
            continue
        if file_sha256(case["checkpoint"]) != case["sha256"]:
            raise RuntimeError("Source checkpoint changed during the comparison")
        print("Export, reload, profile and {} benchmark:".format(device), label, flush=True)
        model, info = load_pose_checkpoint(cfg, case["checkpoint"], device)
        model.requires_grad_(False)
        export_portable(model, info, cfg, case["checkpoint"], export_path, reference_hash)
        verification = verify_reload(model, cfg, export_path, images)
        profile, _ = profile_pose_macs(model, images, forward_heatmaps)
        timings = benchmark_pose(model, images, args.warmup, args.repeats)
        stage = info.get("stage", BASE_STAGE)
        row = {"label": label, "stage": stage, "phase": "profiled", "ap": None,
               "source_checkpoint": case["checkpoint"], "source_sha256": case["sha256"],
               "source_checkpoint_bytes": Path(case["checkpoint"]).stat().st_size,
               "exported_checkpoint": str(export_path), "export_sha256": file_sha256(export_path),
               "checkpoint_bytes": export_path.stat().st_size, "state_bytes": state_size_bytes(model),
               "parameters": profile["parameters"], "pose_macs": profile["pose_macs"],
               "backbone_macs": profile["backbone_macs"], "latency": timings,
               "reload_verification": verification, "run_fingerprint": signature,
               "target_pruning_rate": info.get("target_pruning_rate"),
               "components": {"skp": bool(info.get("skp_applied", stage != BASE_STAGE)),
                              "attention_svd": bool(info.get("attention_svd_applied", stage == OFP_STAGE)),
                              "reconstruction": stage == OFP_STAGE and info.get("mlp_strategy", "optimize") == "optimize"},
               "smoke_source": bool(info.get("smoke_test", False) or info.get("source_smoke_test", False))}
        atomic_json(result_path, row)
        rows.append(row)
        write_comparison(output_dir, rows, protocol, "profiling")
        del model, info
        gc.collect()
    write_comparison(output_dir, rows, protocol, "smoke check" if args.smoke_test else "evaluating AP")
    if not args.smoke_test:
        for row in rows:
            if args.resume and row.get("phase") == "completed" and row.get("ap") is not None:
                print("Reusing AP:", row["label"], flush=True)
                continue
            print("Official {} AP: {}".format(args.split, row["label"]), flush=True)
            model, _ = load_pose_checkpoint(cfg, row["exported_checkpoint"], device)
            results_dir = output_dir / "evaluation" / row["label"]
            results_dir.mkdir(parents=True, exist_ok=True)
            predictions = []
            progress = mmcv.ProgressBar(len(dataset))
            with torch.no_grad():
                for batch in loader:
                    predictions.append(model(return_loss=False, **move_batch(batch, device)))
                    progress.update()
            mmcv.dump(predictions, str(results_dir / "predictions.pkl"))
            metrics = dataset.evaluate(predictions, res_folder=str(results_dir), metric="mAP")
            metrics = {name: float(value) for name, value in metrics.items()}
            if not math.isfinite(metrics["AP"]) or not 0 <= metrics["AP"] <= 1:
                raise RuntimeError("Invalid AP for {}".format(row["label"]))
            row.update({"ap": metrics["AP"], "metrics": metrics, "phase": "completed"})
            atomic_json(output_dir / "results" / (row["label"] + ".json"), row)
            write_comparison(output_dir, rows, protocol, "evaluating AP")
            del model, predictions
            gc.collect()
    summary = write_comparison(output_dir, rows, protocol, "smoke check complete" if args.smoke_test else "complete")
    for row in summary:
        ap = "AP skipped" if row["AP"] is None else "AP {:.2f}".format(row["AP"])
        print("{} | {} | {:.2f}M params | {:.3f} GMACs | {:.2f} ms | {:.2f}x".format(
            row["model"], ap, row["parameters_M"], row["GMACs"], row["median_ms"], row["speedup"]))
    print("\nPortable weights:", output_dir / "models")
    print("Resolved config:", output_dir / "config_snapshot.py")
    print("Comparison:", output_dir / "comparison.md")
    print("CSV:", output_dir / "comparison.csv")
    print("JSON:", output_dir / "comparison.json")
    return summary


def main(argv=None):
    parser = get_args_parser()
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
