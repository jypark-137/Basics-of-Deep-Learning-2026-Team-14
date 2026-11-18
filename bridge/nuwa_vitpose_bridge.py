"""Connect vanilla NuWa to an existing ordinary ViTPose model.

Keep this file and bridge_paths.py in bridge/, then run from the
project root:

    python bridge/nuwa_vitpose_bridge.py

The defaults use the same AP-10K config and split checkpoint as check_split.py.
Configs use vitpose/configs/. The checkpoint is bridge/checkpoints/split/ap10k.pth.
Override them with --config and --checkpoint if needed. No dataset is needed.
Keep your existing third-party dependencies installed.

This module provides only model access and the existing pose loss. It does NOT
implement or change NuWa's masks, STE, neuron selection, SVD, or compensation.
It does not train, prune, save, or overwrite a checkpoint.

Usage with NuWa:

    backbone = get_nuwa_backbone(pose_model)
    # NuWa's block operations use backbone.blocks, not pose_model.blocks.
    # Keep pose_model, including its existing 17-keypoint head, for pose loss.

The original NuWa classification driver still needs pose-specific data,
evaluation and FLOP accounting. This bridge does not make that driver directly
runnable on AP-10K.
"""

import argparse

if __package__:
    from .bridge_paths import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, add_device_argument, resolve_device, resolve_input_file
else:
    from bridge_paths import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, add_device_argument, resolve_device, resolve_input_file

import torch
from torch import nn


def get_nuwa_backbone(pose_model):
    """Return the ViT backbone with the attention metadata NuWa expects.

    ViTPose stores transformer blocks in pose_model.backbone.blocks.
    Its Attention uses a local head_dim variable instead of storing it as an
    attribute. Derive that same number from qkv's actual output width.
    Adding this integer attribute does not change any tensor or forward pass.

    This returns the existing backbone, not a copy. Later changes to its
    blocks will therefore also be used by the complete pose model.
    """
    backbone = getattr(pose_model, "backbone", None)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("Expected an ordinary ViTPose model with backbone.blocks")

    metadata = []
    for index, block in enumerate(blocks):
        name = "backbone.blocks.{}".format(index)
        mlp = getattr(block, "mlp", None)
        attn = getattr(block, "attn", None)
        if mlp is None or attn is None or hasattr(mlp, "experts"):
            raise ValueError("{} must contain ordinary ViT MLP/attention layers".format(name))
        for module, attribute in ((mlp, "fc1"), (mlp, "fc2"),
                                  (attn, "qkv"), (attn, "proj")):
            if not isinstance(getattr(module, attribute, None), nn.Linear):
                raise ValueError("{}: {} must be a Linear layer".format(name, attribute))

        heads = getattr(attn, "num_heads", 0)
        if not isinstance(heads, int) or heads <= 0:
            raise ValueError("{}: invalid attention head count".format(name))
        if attn.qkv.out_features % (3 * heads):
            raise ValueError("{}: qkv width is incompatible with head count".format(name))
        head_dim = attn.qkv.out_features // (3 * heads)
        if hasattr(attn, "head_dim") and attn.head_dim != head_dim:
            raise ValueError("{}: existing head_dim disagrees with qkv".format(name))
        width = mlp.fc1.in_features
        if not (
            mlp.fc1.out_features == mlp.fc2.in_features
            and mlp.fc2.out_features == width
            and attn.qkv.in_features == width
            and attn.proj.in_features == heads * head_dim
            and attn.proj.out_features == width
        ):
            raise ValueError("{}: inconsistent residual/MLP/attention dimensions".format(name))
        metadata.append((attn, head_dim))

    # Validate every block before adding any metadata.
    for attn, head_dim in metadata:
        attn.head_dim = head_dim
    return backbone


def forward_heatmaps(pose_model, images):
    """Use the existing backbone and pose head, keeping autograd enabled."""
    features = pose_model.backbone(images)
    if getattr(pose_model, "with_neck", False):
        features = pose_model.neck(features)
    heatmaps = pose_model.keypoint_head(features)
    if not torch.is_tensor(heatmaps) or heatmaps.ndim != 4:
        raise ValueError("Expected one heatmap tensor with shape [batch, joints, height, width]")
    return heatmaps


def heatmap_task_loss(pose_model, heatmaps, targets, target_weights):
    """Use the pose head's configured loss, including keypoint visibility.

    For the AP-10K config this is JointsMSELoss(use_target_weight=True).
    This replaces classification cross-entropy as NuWa's original task loss.
    """
    losses = pose_model.keypoint_head.get_loss(heatmaps, targets, target_weights)
    terms = []
    for name, value in losses.items():
        if "loss" not in name:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        terms.extend(item.mean() for item in values if torch.is_tensor(item))
    if not terms:
        raise ValueError("The pose head did not return a tensor loss")
    return sum(terms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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

    # Use the already installed ViTPose environment. No NuWa-only dependencies
    # (such as MMDetection, Transformers or wandb) are imported here.
    from mmcv import Config
    from mmpose.models import build_posenet

    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = Config.fromfile(str(config_path))
    model = build_posenet(cfg.model).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    model.eval()
    print("Device:", device)
    print("Checkpoint loaded successfully")

    # Freeze parameters for this diagnostic. Gradients will be checked through
    # the full model using the input image, without updating any model weights.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    width, height = map(int, cfg.data_cfg.image_size)
    image = torch.randn(1, 3, height, width, device=device)
    with torch.no_grad():
        reference = forward_heatmaps(model, image)

    backbone = get_nuwa_backbone(model)
    first = backbone.blocks[0]
    print("Transformer blocks:", len(backbone.blocks))
    print("First MLP: {} -> {} -> {}".format(
        first.mlp.fc1.in_features, first.mlp.fc1.out_features,
        first.mlp.fc2.out_features,
    ))
    print("Attention heads: {}; dimension per head: {}".format(
        first.attn.num_heads, first.attn.head_dim,
    ))

    image.requires_grad_(True)
    heatmaps = forward_heatmaps(model, image)
    expected = (1, int(cfg.data_cfg.num_joints),
                int(cfg.data_cfg.heatmap_size[1]), int(cfg.data_cfg.heatmap_size[0]))
    if tuple(heatmaps.shape) != expected:
        raise RuntimeError("Expected heatmaps {}, got {}".format(expected, tuple(heatmaps.shape)))
    if not torch.isfinite(heatmaps).all().item():
        raise RuntimeError("Heatmaps contain NaN or infinity")
    if not torch.allclose(reference, heatmaps.detach(), rtol=1e-5, atol=1e-6):
        raise RuntimeError("Adding the backbone metadata changed the heatmaps")
    print("Heatmap shape:", tuple(heatmaps.shape))
    print("Heatmaps unchanged: PASS")

    # Synthetic targets only verify that the original weighted pose loss is
    # connected to the full model. This number is not an accuracy measurement.
    targets = torch.zeros_like(heatmaps)
    target_weights = torch.ones(1, expected[1], 1, device=device)
    loss = heatmap_task_loss(model, heatmaps, targets, target_weights)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Pose loss contains NaN or infinity")
    loss.backward()
    gradient = image.grad
    if (gradient is None or not torch.isfinite(gradient).all().item()
            or gradient.abs().sum().item() == 0):
        raise RuntimeError("No finite, nonzero gradient through the complete pose model")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("An original model parameter unexpectedly received a gradient")
    print("Pose loss and backward pass: PASS")
    print("Connection check passed")


if __name__ == "__main__":
    main()
