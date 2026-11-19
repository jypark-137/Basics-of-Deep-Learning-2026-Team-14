"""Optional update-based warmup and cosine learning-rate decay."""

import math
from numbers import Integral

from torch.optim.lr_scheduler import LambdaLR


SCHEDULER_DEFAULTS = {"lr_scheduler": "none", "warmup_steps": 100, "min_lr_ratio": 0.1}


def add_scheduler_arguments(parser):
    group = parser.add_argument_group("Learning-rate schedule")
    group.add_argument("--lr-scheduler", choices=("none", "cosine"),
                       default=SCHEDULER_DEFAULTS["lr_scheduler"],
                       help="Optional per-update schedule; default: constant learning rates")
    group.add_argument("--warmup-steps", type=int, default=SCHEDULER_DEFAULTS["warmup_steps"],
                       help="Linear warmup updates for cosine; capped to steps - 1 (default: 100)")
    group.add_argument("--min-lr-ratio", type=float, default=SCHEDULER_DEFAULTS["min_lr_ratio"],
                       help="Final / peak learning rate for each parameter group (default: 0.1)")


def scheduler_settings(args, steps):
    """Validate requested options and describe the effective update schedule."""
    name = getattr(args, "lr_scheduler", SCHEDULER_DEFAULTS["lr_scheduler"])
    warmup = getattr(args, "warmup_steps", SCHEDULER_DEFAULTS["warmup_steps"])
    minimum = float(getattr(args, "min_lr_ratio", SCHEDULER_DEFAULTS["min_lr_ratio"]))
    if name not in ("none", "cosine"):
        raise ValueError("lr-scheduler must be none or cosine")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0:
        raise ValueError("A learning-rate schedule requires a positive number of updates")
    if isinstance(warmup, bool) or not isinstance(warmup, Integral) or warmup < 0:
        raise ValueError("warmup-steps must be a nonnegative integer")
    if not math.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError("min-lr-ratio must be finite and between 0 and 1")
    return {"name": name, "total_steps": int(steps),
            "warmup_steps": min(int(warmup), int(steps) - 1) if name == "cosine" else 0,
            "min_lr_ratio": minimum if name == "cosine" else 1.0}


def lr_multiplier(update, settings):
    """Scale for a zero-based optimizer update, including the first and last."""
    if settings["name"] == "none":
        return 1.0
    steps, warmup = settings["total_steps"], settings["warmup_steps"]
    update = min(max(update, 0), steps - 1)
    if update < warmup:
        return (update + 1) / warmup
    if steps == 1:
        return 1.0
    # The last warmup update reaches the peak; the following update decays.
    progress = ((update - warmup + 1) / (steps - warmup) if warmup
                else update / (steps - 1))
    minimum = settings["min_lr_ratio"]
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def make_scheduler(optimizer, settings):
    """Initialize the first update's LR; call step() AFTER optimizer.step()."""
    if settings["name"] == "none":
        return None
    return LambdaLR(optimizer, lr_lambda=lambda update: lr_multiplier(update, settings))
