"""NuWa's class-specific derivation adapted to a species-specific ViTPose model.

The derive command learns/reuses SKP masks, exports an anchor, performs OFP,
and compares the resulting models. Optional fine-tuning is explicit. Resume
reuses verified completed stages; it does not resume an optimizer mid-run.
"""


import argparse
from datetime import datetime
import gc
import json
from pathlib import Path

import torch

from ..dataset.utils import evaluation_paths, get_dataloader
from ..model.checkpoint import (
    BASE_STAGE, OFP_STAGE, checkpoint_header, load_anchor_model,
)
from ..model.utils import config_snapshot, pose_config, pose_input
from device_utils import add_device_argument, resolve_device
from ..model.vit import hidden_sizes
from ..paths import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPECIES, resolve_data_root, resolve_input_file, resolve_project_path,
)
from ..utils.io import atomic_json, file_sha256, fingerprint
from ..utils.pose_results import validate_provenance
from ..utils.utils import profile_pose_macs, set_seed
from ..model.vit import forward_heatmaps
from .get_anchor_model import get_anchor_model
from .pruning import pruning_vit
from ..species import add_species_argument, normalize_species, species_slug


def class_specific_derivation_nuwa(args, model, retrain=False):
    """In-memory equivalent of the original NuWa entry point, using pose loaders."""
    cfg = pose_config(getattr(args, "config", None))
    args.device = str(resolve_device(getattr(args, "device", None), model))
    model.to(args.device).eval()
    original_widths = hidden_sizes(model)
    original_profile, _ = profile_pose_macs(model, pose_input(model, cfg), forward_heatmaps)
    if getattr(args, "anchor", None):
        model_anchor, info = load_anchor_model(cfg, args.anchor, args.device)
        args.species = normalize_species(getattr(args, "species", None) or info.get("species"))
        if args.species != normalize_species(info.get("species")):
            raise ValueError("Species must match the anchor's species")
    else:
        args.species = normalize_species(getattr(args, "species", None) or DEFAULT_SPECIES)
    train_loader, val_loader = get_dataloader(args, cfg)
    if not getattr(args, "anchor", None):
        model_anchor = get_anchor_model(args, model, train_loader, val_loader)
    model_pruned = pruning_vit(args, model_anchor, original_profile)
    if retrain:
        from ..engine.train import train
        options = argparse.Namespace(**vars(args))
        options.steps = getattr(args, "finetune_steps", 1000)
        options.cfg = cfg
        options._model_info = {"stage": OFP_STAGE,
                               "architecture": {"original_hidden_sizes": original_widths}}
        train(options, model_pruned, train_loader, val_loader, model_name="finetuned.pth",
              early_stop=False, best=True)
    return model_pruned


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Original split ViTPose weights")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    add_species_argument(parser)
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3))
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--anchor", help="Reuse an exported anchor; skip mask learning")
    reuse.add_argument("--masks", help="Reuse learned masks; skip mask learning")
    parser.add_argument("--out-dir", "--output_dir", dest="out_dir")
    parser.add_argument("--pruning-rate", "--pruning_rate", dest="pruning_rate", type=float, default=0.4)
    parser.add_argument("--steps", "--num_steps", dest="steps", type=int, default=10000)
    parser.add_argument("--batch-size", "--train_batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--mask-lr", type=float, default=1e-3)
    parser.add_argument("--beta-lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--strategy", choices=("optimize", "prune"), default="optimize")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--retrain", action="store_true", help="Optionally fine-tune the OFP model")
    parser.add_argument("--finetune-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--skip-comparison", action="store_true")
    add_device_argument(parser)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Ten mask updates, two pruning crops, no AP")
    return parser


def _completed(directory, checkpoint_name, report_name, expected, recorded_hash=None):
    checkpoint, report_path = directory / checkpoint_name, directory / report_name
    if not checkpoint.is_file() or not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("standalone_checkpoint_reload") and not report.get("reload_verification", {}).get("passed"):
        return False
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed stage settings differ: {}".format(directory))
    if recorded_hash is not None and file_sha256(checkpoint) != recorded_hash:
        raise ValueError("Completed stage checkpoint changed: {}".format(checkpoint))
    return True


def _fresh_stage_directory(directory, root, resume):
    if not directory.exists():
        return
    if not resume:
        raise FileExistsError("Stage output already exists: {}".format(directory))
    if directory.resolve().parent != root.resolve():
        raise ValueError("A partial stage must stay inside the selected run folder")
    backup = directory.with_name(directory.name + "_partial_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if backup.resolve().parent != root.resolve():
        raise ValueError("A partial-stage backup must stay inside the selected run folder")
    directory.rename(backup)
    print("Preserved incomplete stage:", backup, flush=True)


def run(args, parser=None):
    from . import get_anchor_model as anchor_module
    from . import pruning as pruning_module
    from .. import finalize_pose
    parser = parser or get_args_parser()
    if min(args.threads, args.steps, args.batch_size, args.eval_every, args.log_every,
           args.calibration_samples, args.finetune_steps, args.repeats) <= 0 or args.warmup < 0:
        parser.error("Counts must be positive; warmup may be zero")
    args.device = str(resolve_device(args.device))
    pruning_module.energy_ratio(args.pruning_rate)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    base = resolve_input_file(args.checkpoint, "--checkpoint")
    config = resolve_input_file(args.config, "--config")
    source = resolve_input_file(args.anchor or args.masks, "--anchor/--masks") if args.anchor or args.masks else None
    header = checkpoint_header(source) if args.anchor else {}
    if args.masks:
        masks = torch.load(str(source), map_location="cpu")
        header = {key: masks.get(key) for key in ("species", "split")}
    args.species = normalize_species(args.species or header.get("species") or DEFAULT_SPECIES)
    if args.masks and args.species != normalize_species(header.get("species")):
        parser.error("--species must match the species used to learn these masks")
    args.split_id = args.split_id or header.get("split") or 1
    args.data_root = str(resolve_data_root(args.data_root))
    base_header = checkpoint_header(base)
    if base_header["stage"] != BASE_STAGE:
        parser.error("--checkpoint must contain the unpruned baseline")
    base_hash = file_sha256(base)
    base_ids = {value for value in (base_hash, base_header.get("base_checkpoint_sha256"),
                base_header.get("source_checkpoint_sha256"), base_header.get("origin_checkpoint_sha256")) if value}
    if args.anchor:
        validate_provenance(header, "anchor", args.species, args.split_id, base_ids, {file_sha256(source)})
    annotations = {}
    for split in ("train", "val"):
        path, _, _ = evaluation_paths(Path(args.data_root), args.species, split, args.split_id)
        annotations[split] = file_sha256(path)
    settings = {key: value for key, value in vars(args).items() if key not in ("resume", "out_dir", "command")}
    settings.update({"checkpoint": str(base), "base_sha256": base_hash, "config": str(config),
                     "config_sha256": fingerprint(config_snapshot(config)),
                     "code_sha256": finalize_pose.code_fingerprint(), "annotation_sha256": annotations,
                     "source_sha256": file_sha256(source) if source else None})
    signature = fingerprint(settings)
    directory = resolve_project_path(args.out_dir) if args.out_dir else (
        DEFAULT_OUTPUT_ROOT / "nuwa" / species_slug(args.species) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    )
    manifest_path = directory / "run_manifest.json"
    if directory.exists():
        if not args.resume or not manifest_path.is_file():
            parser.error("Output exists; choose another --out-dir or resume the same completed stages")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != signature:
            parser.error("Resume inputs or settings differ from this run")
    else:
        directory.mkdir(parents=True)
        manifest = {"fingerprint": signature, "settings": settings, "completed": {}}
        atomic_json(manifest_path, manifest)
    common = ["--config", str(config), "--data-root", args.data_root,
              "--device", str(args.device), "--threads", str(args.threads), "--seed", str(args.seed)]
    if args.anchor:
        anchor_path = source
    else:
        anchor_dir = directory / "anchor"
        if not _completed(anchor_dir, "anchor.pth", "anchor_report.json",
                          {"base_checkpoint_sha256": base_hash}, manifest["completed"].get("anchor")):
            # Optimizer resumption is not implemented by the original mask workflow.
            if anchor_dir.exists() and not args.masks:
                parser.error("Incomplete anchor stage: explicitly select saved --masks or use a new run folder")
            _fresh_stage_directory(anchor_dir, directory, args.resume)
            command = common + ["--species", *args.species, "--split", str(args.split_id),
                                "--out-dir", str(anchor_dir), "--steps", str(args.steps),
                                "--batch-size", str(args.batch_size), "--mask-lr", str(args.mask_lr),
                                "--beta-lr", str(args.beta_lr), "--weight-decay", str(args.weight_decay),
                                "--eval-every", str(args.eval_every), "--log-every", str(args.log_every)]
            if args.masks:
                command += ["--masks", str(source)]
            if args.smoke_test:
                command.append("--smoke-test")
            anchor_module.main(command)
        anchor_path = anchor_dir / "anchor.pth"
        manifest["completed"]["anchor"] = file_sha256(anchor_path)
        atomic_json(manifest_path, manifest)
        gc.collect()
    prune_dir = directory / "ofp"
    expected = {"source_anchor_sha256": file_sha256(anchor_path), "target_pruning_rate": args.pruning_rate,
                "mlp_strategy": args.strategy, "attention_svd_applied": not args.skip_attention,
                "smoke_test": args.smoke_test}
    if not _completed(prune_dir, "edge.pth", "ofp_report.json", expected, manifest["completed"].get("ofp")):
        _fresh_stage_directory(prune_dir, directory, args.resume)
        command = common + ["--anchor", str(anchor_path), "--species", *args.species,
                            "--split-id", str(args.split_id), "--pruning-rate", str(args.pruning_rate),
                            "--strategy", args.strategy, "--calibration-samples", str(args.calibration_samples),
                            "--out-dir", str(prune_dir)]
        if args.skip_attention:
            command.append("--skip-attention")
        if args.smoke_test:
            command.append("--smoke-test")
        pruning_module.main(command)
    edge = prune_dir / "edge.pth"
    manifest["completed"]["ofp"] = file_sha256(edge)
    atomic_json(manifest_path, manifest)
    edges = ["nuwa=" + str(edge)]
    selected = edge
    if args.retrain:
        from ..engine import train as training_module
        train_dir = directory / "finetune"
        name = "last.pth" if args.smoke_test else "best.pth"
        if not _completed(train_dir, name, "training_report.json", {"source_checkpoint": str(edge)},
                          manifest["completed"].get("finetune")):
            if train_dir.exists():
                parser.error("Incomplete fine-tuning stage: select its checkpoint with the train command")
            command = common + ["--checkpoint", str(edge), "--species", *args.species,
                                "--split-id", str(args.split_id), "--out-dir", str(train_dir),
                                "--steps", str(args.finetune_steps), "--batch-size", str(args.batch_size),
                                "--learning-rate", str(args.learning_rate), "--weight-decay", str(args.weight_decay)]
            if args.smoke_test:
                command.append("--smoke-test")
            training_module.main(command)
        selected = train_dir / name
        manifest["completed"]["finetune"] = file_sha256(selected)
        atomic_json(manifest_path, manifest)
        edges.append("finetuned=" + str(selected))
    if not args.skip_comparison:
        command = ["--checkpoint", str(base), "--anchor", str(anchor_path), "--edge"] + edges + [
            "--config", str(config), "--data-root", args.data_root, "--species", *args.species,
            "--device", args.device,
            "--split-id", str(args.split_id), "--threads", str(args.threads), "--seed", str(args.seed),
            "--warmup", str(args.warmup), "--repeats", str(args.repeats), "--out-dir", str(directory / "comparison")]
        if args.smoke_test:
            command.append("--smoke-test")
        if args.resume:
            command.append("--resume")
        finalize_pose.main(command)
    result = {"species": args.species, "anchor": str(anchor_path), "edge": str(edge), "selected_model": str(selected),
              "comparison": None if args.skip_comparison else str(directory / "comparison/comparison.md"),
              "smoke_test": args.smoke_test}
    atomic_json(directory / "nuwa_result.json", result)
    print("NuWa outputs:", directory, flush=True)
    return result


def main(argv=None):
    parser = get_args_parser()
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
