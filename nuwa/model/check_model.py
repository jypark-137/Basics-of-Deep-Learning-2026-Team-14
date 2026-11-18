"""Verify checkpoint loading, ViTPose heatmaps and a gradient through the pose loss."""


import argparse

import torch

from ..paths import DEFAULT_CHECKPOINT, DEFAULT_CONFIG
from .utils import get_model, pose_config, pose_input
from device_utils import add_device_argument, resolve_device
from .vit import AttentionPruned, forward_heatmaps, get_nuwa_backbone, heatmap_task_loss, hidden_sizes
from ..utils.console import CommandParser, command_output


def get_args_parser():
    parser = CommandParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    add_device_argument(parser)
    parser.add_argument("--threads", type=int, default=1)
    return parser


@command_output
def run(args, parser=None):
    parser = parser or get_args_parser()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    cfg = pose_config(args.config)
    model, info = get_model(cfg, args.checkpoint, resolve_device(args.device), return_info=True)
    if not any(isinstance(block.attn, AttentionPruned) for block in model.backbone.blocks):
        get_nuwa_backbone(model)
    model.requires_grad_(False)
    images = pose_input(model, cfg)
    with torch.no_grad():
        reference = forward_heatmaps(model, images)
    expected = (1, int(cfg.data_cfg.num_joints), int(cfg.data_cfg.heatmap_size[1]),
                int(cfg.data_cfg.heatmap_size[0]))
    if tuple(reference.shape) != expected or not torch.isfinite(reference).all().item():
        raise RuntimeError("Expected finite heatmaps with shape {}".format(expected))
    images.requires_grad_(True)
    heatmaps = forward_heatmaps(model, images)
    if not torch.allclose(reference, heatmaps.detach(), rtol=1e-4, atol=1e-5):
        raise RuntimeError("The repeated evaluation changed its heatmaps beyond tolerance")
    weights = torch.ones(1, expected[1], 1, device=images.device, dtype=images.dtype)
    loss = heatmap_task_loss(model, heatmaps, torch.zeros_like(heatmaps), weights)
    loss.backward()
    if (images.grad is None or not torch.isfinite(images.grad).all().item()
            or images.grad.abs().sum().item() == 0):
        raise RuntimeError("Expected a finite nonzero gradient through the pose model")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("A frozen model weight received a gradient")
    result = {"stage": info.get("stage"), "device": str(images.device), "heatmap_shape": list(expected),
              "hidden_sizes": hidden_sizes(model), "gradient_check": True}
    print("Model check passed ({}, heatmaps {}).".format(images.device, " x ".join(map(str, expected))))
    return result


def main(argv=None):
    parser = get_args_parser()
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
