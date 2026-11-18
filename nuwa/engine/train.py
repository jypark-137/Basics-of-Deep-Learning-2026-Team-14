"""Optional supervised ViTPose fine-tuning corresponding to NuWa engine/train.py.

SKP mask learning remains in method/get_anchor_model.py. This module updates
model weights only when the train command or --retrain is explicitly selected.
It uses the existing weighted heatmap loss and species validation AP. Model
checkpoints include architecture metadata and can reload independently.
"""


import argparse
from datetime import datetime
import math
from pathlib import Path

import torch

from ..dataset.utils import get_dataloader, move_batch
from ..model.checkpoint import load_pose_checkpoint, verify_reload
from ..model.utils import pose_config, pose_input
from device_utils import add_device_argument, resolve_device
from ..model.vit import forward_heatmaps, heatmap_task_loss
from ..paths import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT,
    resolve_input_file, resolve_project_path,
)
from ..utils.io import atomic_json
from ..utils.utils import set_seed
from .eval import evaluate
from .utils import get_delta_acc, log_metrics, save_model


def train(args, model, train_loader, test_loader, model_name=None, early_stop=True, best=False):
    """Fine-tune in place; return best/final validation AP in 0-100 units.

All model weights are unfrozen for this explicitly requested fine-tuning.
Smoke runs perform ten updates, skip AP, save last.pth and return None.
Validation-best and final weights are separate files. No optimizer state is
claimed to be resumable; completed checkpoints can be used as new inputs.
"""
    if train_loader is None or len(train_loader) == 0:
        raise ValueError("Fine-tuning requires a nonempty training loader")
    smoke = bool(getattr(args, "smoke_test", False))
    if not smoke and (test_loader is None or len(test_loader) == 0):
        raise ValueError("Fine-tuning requires a species validation loader")
    steps = getattr(args, "steps", getattr(args, "num_steps", -1))
    if steps <= 0:
        steps = getattr(args, "num_epochs", 1) * len(train_loader)
    if smoke:
        steps = 10
    every = getattr(args, "eval_every", 1000)
    log_every = getattr(args, "log_every", 10)
    lr, decay = getattr(args, "learning_rate", 1e-4), getattr(args, "weight_decay", 0.05)
    if min(steps, every, log_every) <= 0 or not math.isfinite(lr) or lr <= 0 or not math.isfinite(decay) or decay < 0:
        raise ValueError("Use positive update/evaluation counts and LR, with nonnegative weight decay")
    device = resolve_device(getattr(args, "device", None), model)
    model.to(device)
    model.requires_grad_(True)
    directory = resolve_project_path(getattr(args, "out_dir", None) or getattr(args, "output_dir", None)
                     or DEFAULT_OUTPUT_ROOT / "training")
    directory.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(directory)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay,
                                  betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    model.zero_grad(set_to_none=True)
    best_ap = last_ap = None
    last_eval_step = -1
    stable = 0
    best_name = model_name or "best.pth"
    if Path(best_name).name == "last.pth" and not smoke:
        raise ValueError("The best checkpoint name must differ from last.pth")
    if not smoke:
        best_ap = last_ap = evaluate(model, test_loader, device=device,
                                    output_dir=directory / "eval" / "initial")
        save_model(args, model, best_name, step=0, val_ap=best_ap / 100)
        last_eval_step = 0
    iterator = iter(train_loader)
    update = 0
    completed = 0
    try:
        for update in range(1, steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            model.train()
            data = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            heatmaps = forward_heatmaps(model, data["img"])
            loss = heatmap_task_loss(model, heatmaps, data["target"], data["target_weight"])
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite fine-tuning loss at update {}".format(update))
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
                   for parameter in model.parameters()):
                raise RuntimeError("Nonfinite fine-tuning gradient")
            optimizer.step()
            completed = update
            if update == 1 or update % log_every == 0 or update == steps:
                event = {"event": "train", "step": update, "loss": float(loss.item())}
                log_metrics(directory, event)
                print("Fine-tuning {}/{} | loss {:.6f}".format(update, steps, event["loss"]), flush=True)
            if not smoke and (update % every == 0 or update == steps):
                ap = evaluate(model, test_loader, device=device,
                              output_dir=directory / "eval" / "update_{:06d}".format(update))
                stable = stable + 1 if get_delta_acc(ap, last_ap) < 1.0 else 0
                last_ap, last_eval_step = ap, update
                if ap > best_ap:
                    best_ap = ap
                    save_model(args, model, best_name, step=update, val_ap=ap / 100)
                log_metrics(directory, {"event": "validation", "step": update,
                                        "val_ap": ap / 100, "best_val_ap": best_ap / 100})
                if early_stop and stable >= 3:
                    break
    finally:
        current_ap = last_ap / 100 if last_eval_step == completed and last_ap is not None else None
        save_model(args, model, "last.pth", step=completed, val_ap=current_ap)
    model.eval()
    return best_ap if best else last_ap


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--species")
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3))
    parser.add_argument("--out-dir")
    add_device_argument(parser)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", "--num_steps", dest="steps", type=int, default=1000)
    parser.add_argument("--batch-size", "--train_batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def run(args, parser=None):
    parser = parser or get_args_parser()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    args.device = str(resolve_device(args.device))
    cfg = pose_config(args.config)
    checkpoint = resolve_input_file(args.checkpoint, "--checkpoint")
    model, info = load_pose_checkpoint(cfg, checkpoint, resolve_device(args.device))
    args.species = args.species or info.get("species") or "dog"
    args.split_id = args.split_id or info.get("split", 1)
    if info.get("species") and args.species.casefold() != info["species"].casefold():
        parser.error("species must match the source checkpoint")
    if info.get("split") and args.split_id != info["split"]:
        parser.error("split-id must match the source checkpoint")
    args._model_info = {key: value for key, value in info.items() if key != "state_dict"}
    directory = resolve_project_path(args.out_dir) if args.out_dir else (
        DEFAULT_OUTPUT_ROOT / "finetune" / args.species
        / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    )
    if directory.exists():
        parser.error("Output folder already exists; choose another --out-dir")
    args.out_dir = str(directory)
    train_loader, val_loader = get_dataloader(args, cfg)
    ap = train(args, model, train_loader, val_loader, early_stop=args.early_stop, best=True)
    selected = directory / ("last.pth" if args.smoke_test else "best.pth")
    # Verify the saved selection, which may precede the last optimizer update.
    selected_model, _ = load_pose_checkpoint(cfg, selected, args.device)
    verification = verify_reload(selected_model, cfg, selected, pose_input(selected_model, cfg))
    report = {"checkpoint": str(selected), "source_checkpoint": str(checkpoint),
              "species": args.species, "split_id": args.split_id,
              "best_val_ap": None if ap is None else ap / 100,
              "smoke_test": args.smoke_test, "reload_verification": verification}
    atomic_json(directory / "training_report.json", report)
    print("Fine-tuned checkpoint:", selected, flush=True)
    return report


def main(argv=None):
    parser = get_args_parser()
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
