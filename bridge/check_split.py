"""Load the split AP-10K checkpoint and run one dummy image on CUDA or CPU.

Keep this file with bridge_paths.py in bridge/.
Run from the project root:

    python bridge/check_split.py

The config is under vitpose/configs/ and the checkpoint is
bridge/checkpoints/split/ap10k.pth.
Existing third-party dependencies still need to be installed.
No dataset, training, or checkpoint modification is involved in this check.
"""

import argparse

if __package__:
    from .bridge_paths import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, add_device_argument, resolve_device, resolve_input_file
else:
    from bridge_paths import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, add_device_argument, resolve_device, resolve_input_file


def load_pose_model(cfg, checkpoint_path, device="auto"):
    """Build the configured pose model and require an exact weight match."""
    import torch
    from mmpose.models import build_posenet

    device = resolve_device(device)
    model = build_posenet(cfg.model)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("stage") == "NuWa-SKP":
        raise ValueError(
            "This is a NuWa mask-only file. Use the original split ap10k.pth "
            "checkpoint for check_split.py and the unpruned baseline evaluator."
        )
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    add_device_argument(parser)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    try:
        config_path = resolve_input_file(args.config, "--config")
        checkpoint_path = resolve_input_file(args.checkpoint, "--checkpoint")
        device = resolve_device(args.device)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    import torch
    from mmcv import Config

    cfg = Config.fromfile(str(config_path))
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    model = load_pose_model(cfg, checkpoint_path, device)
    print("Device:", device, flush=True)
    print("Checkpoint loaded successfully", flush=True)

    width, height = map(int, cfg.data_cfg.image_size)
    heatmap_width, heatmap_height = map(int, cfg.data_cfg.heatmap_size)
    expected = (1, int(cfg.data_cfg.num_joints), heatmap_height, heatmap_width)
    torch.manual_seed(0)
    with torch.no_grad():
        image = torch.randn(1, 3, height, width, device=device)
        features = model.backbone(image)
        if getattr(model, "with_neck", False):
            features = model.neck(features)
        heatmaps = model.keypoint_head(features)

    if not torch.is_tensor(heatmaps) or tuple(heatmaps.shape) != expected:
        raise RuntimeError("Expected heatmaps with shape {}, got {}".format(
            expected, getattr(heatmaps, "shape", type(heatmaps).__name__)))
    if not torch.isfinite(heatmaps).all().item():
        raise RuntimeError("Heatmaps contain NaN or infinity")

    print("Heatmap shape:", tuple(heatmaps.shape))
    print("{} test passed".format(device))


if __name__ == "__main__":
    main()
