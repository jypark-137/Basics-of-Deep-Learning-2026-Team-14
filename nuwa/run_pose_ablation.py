"""Build the existing controlled OFP ablations and run a common comparison."""


import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from .paths import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPECIES, PROJECT_ROOT, resolve_data_root, resolve_input_file, resolve_project_path,
)
from .dataset.utils import evaluation_paths
from .model.utils import config_snapshot
from device_utils import add_device_argument, device_info, resolve_device
from .utils.io import atomic_json, file_sha256, fingerprint
from .utils.pose_results import ABLATIONS, ablation_cases, validate_provenance
from .species import add_species_argument, normalize_species
from .utils.console import CommandParser, command_output, display_path, is_verbose

def get_args_parser():
    parser = CommandParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rates", nargs="+", required=True, type=float)
    parser.add_argument("--variants", nargs="+", choices=list(ABLATIONS), default=list(ABLATIONS))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    add_species_argument(parser)
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3))
    parser.add_argument("--calibration-samples", type=int, default=128)
    add_device_argument(parser)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", help="Default: a new outputs/ablation_study run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Two training crops per case; no AP")
    return parser


@command_output
def run(args, parser=None):
    parser = parser or get_args_parser()
    args.data_root = str(resolve_data_root(args.data_root))
    if min(args.calibration_samples, args.threads, args.repeats) <= 0 or args.warmup < 0:
        parser.error("Sample/thread/repeat counts must be positive; warmup must be nonnegative")
    try:
        args.device = str(resolve_device(args.device))
        anchor_path = resolve_input_file(args.anchor, "--anchor")
        base_path = resolve_input_file(args.checkpoint, "--checkpoint")
        config_path = resolve_input_file(args.config, "--config")
        cases = ablation_cases(args.rates, args.variants)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    from mmcv import Config
    from .finalize_pose import code_fingerprint
    from .model.checkpoint import checkpoint_header

    anchor_hash, base_hash = file_sha256(anchor_path), file_sha256(base_path)
    header, base_header = checkpoint_header(anchor_path), checkpoint_header(base_path)
    args.species = normalize_species(args.species or header.get("species") or DEFAULT_SPECIES)
    split_id = args.split_id or header.get("split", 1)
    base_ids = {value for value in (base_hash, base_header.get("base_checkpoint_sha256"),
                base_header.get("source_checkpoint_sha256"), base_header.get("origin_checkpoint_sha256")) if value}
    try:
        validate_provenance(base_header, "baseline", args.species, split_id, base_ids, {anchor_hash})
        validate_provenance(header, "anchor", args.species, split_id, base_ids, {anchor_hash})
        train_path, _, _ = evaluation_paths(Path(args.data_root), args.species, "train", split_id)
        val_path = None if args.smoke_test else evaluation_paths(
            Path(args.data_root), args.species, "val", split_id,
        )[0]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output_dir = resolve_project_path(args.out_dir) if args.out_dir else (
        DEFAULT_OUTPUT_ROOT / "ablation_study" / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    settings = {"schema_version": 1, "anchor": str(anchor_path), "anchor_sha256": anchor_hash,
                "checkpoint": str(base_path), "checkpoint_sha256": base_hash,
                "config_sha256": fingerprint(config_snapshot(Config.fromfile(str(config_path)))),
                "code_sha256": code_fingerprint(), "species": args.species, "split_id": split_id,
                "train_annotation_sha256": file_sha256(train_path),
                "val_annotation_sha256": file_sha256(val_path) if val_path else None,
                "data_root": args.data_root, "calibration_samples": args.calibration_samples,
                "seed": args.seed, "threads": args.threads, "warmup": args.warmup,
                "device": device_info(args.device),
                "repeats": args.repeats, "smoke_test": args.smoke_test, "cases": cases}
    signature = fingerprint(settings)
    output_dir = output_dir.resolve()
    plan_path = output_dir / "study_plan.json"
    if output_dir.exists():
        if not args.resume or not plan_path.is_file():
            parser.error("Output exists; use --resume for this study or choose another --out-dir")
        if json.loads(plan_path.read_text(encoding="utf-8")).get("fingerprint") != signature:
            parser.error("Study settings, source weights, data or code differ from this run")
    else:
        output_dir.mkdir(parents=True)
        atomic_json(plan_path, {"fingerprint": signature, "settings": settings})
    completion_path = output_dir / "completed_cases.json"
    completed = json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.is_file() else {}
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[key] = str(args.threads)
    edge_arguments = []
    for case in cases:
        label = case["label"]
        directory = output_dir / "models" / label
        edge = directory / "edge.pth"
        report_path = directory / "ofp_report.json"
        valid = False
        if args.resume and edge.is_file() and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            valid = (report.get("standalone_checkpoint_reload") is True
                     and report.get("target_pruning_rate") == case["rate"]
                     and report.get("skp_applied") == case["skp"]
                     and report.get("attention_svd_applied") == case["attention_svd"]
                     and report.get("mlp_strategy") == ("optimize" if case["reconstruction"] else "prune")
                     and report.get("smoke_test") == args.smoke_test)
            if case["skp"]:
                valid = valid and report.get("source_anchor_sha256") == anchor_hash
            else:
                valid = valid and report.get("base_checkpoint_sha256") == base_hash
            if label in completed and completed[label] != file_sha256(edge):
                parser.error("A completed checkpoint was modified: {}".format(label))
        if valid:
            print("Reusing completed ablation:", label, flush=True)
        else:
            if directory.exists():
                if not args.resume:
                    parser.error("Case output already exists: {}".format(directory))
                # Preserve an interrupted case instead of deleting its files.
                backup = directory.with_name(directory.name + "_partial_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                expected_parent = (output_dir / "models").resolve()
                if directory.resolve().parent != expected_parent or backup.resolve().parent != expected_parent:
                    raise ValueError("Ablation backup must remain inside the study models directory")
                directory.rename(backup)
                print("Preserved incomplete case:", backup, flush=True)
            command = [sys.executable, "-m", "nuwa", "prune",
                       "--anchor" if case["skp"] else "--base-checkpoint",
                       str(anchor_path if case["skp"] else base_path),
                       "--pruning-rate", str(case["rate"]), "--strategy",
                       "optimize" if case["reconstruction"] else "prune",
                       "--config", str(config_path), "--data-root", args.data_root,
                       "--species", *args.species, "--split-id", str(split_id),
                       "--calibration-samples", str(args.calibration_samples),
                       "--seed", str(args.seed), "--device", args.device, "--threads", str(args.threads),
                       "--out-dir", str(directory)]
            if not case["attention_svd"]:
                command.append("--skip-attention")
            if args.smoke_test:
                command.append("--smoke-test")
            if is_verbose():
                command.append("--verbose")
            print("Building ablation:", label, flush=True)
            subprocess.run(command, env=environment, cwd=str(PROJECT_ROOT), check=True)
            if not edge.is_file() or not report_path.is_file():
                raise RuntimeError("Pruning did not produce a complete checkpoint/report")
        completed[label] = file_sha256(edge)
        atomic_json(completion_path, completed)
        edge_arguments.append(label + "=" + str(edge))
    command = [sys.executable, "-m", "nuwa", "compare", "--checkpoint", str(base_path),
               "--anchor", str(anchor_path), "--edge"] + edge_arguments + [
               "--config", str(config_path), "--data-root", args.data_root, "--species", *args.species,
               "--split-id", str(split_id), "--threads", str(args.threads),
               "--device", args.device,
               "--warmup", str(args.warmup), "--repeats", str(args.repeats), "--seed", str(args.seed),
               "--out-dir", str(output_dir / "comparison")]
    if args.smoke_test:
        command.append("--smoke-test")
    if args.resume:
        command.append("--resume")
    if is_verbose():
        command.append("--verbose")
    subprocess.run(command, env=environment, cwd=str(PROJECT_ROOT), check=True)
    print("Ablation results:", display_path(output_dir / "comparison/comparison.md"))
    return output_dir


def main(argv=None):
    parser = get_args_parser()
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
