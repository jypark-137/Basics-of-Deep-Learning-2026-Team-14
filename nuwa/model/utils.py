"""Model construction, configuration, devices and synthetic profiling inputs."""


from numbers import Integral
from pathlib import Path
import pprint
import torch
from ..paths import DEFAULT_CONFIG, resolve_input_file
from device_utils import resolve_device


def pose_config(config=None):
    """Accept a resolved ViTPose config, a mapping, or a config-file path."""
    from mmcv import Config
    if config is None or isinstance(config, (str, Path)):
        return Config.fromfile(str(resolve_input_file(config or DEFAULT_CONFIG, "--config")))
    if isinstance(config, dict):
        return Config(config)
    if not hasattr(config, "model") and not hasattr(config, "data_cfg"):
        raise ValueError("Expected a ViTPose config, not a classifier FLOP specification")
    return config


def config_snapshot(config=None):
    """Serialize a resolved MMCV config without depending on YAPF's API."""
    cfg = pose_config(config)
    values = cfg._cfg_dict.to_dict()
    text = "\n\n".join("{} = {}".format(name, pprint.pformat(value, width=100, sort_dicts=False))
                         for name, value in values.items()) + "\n"
    compile(text, "config_snapshot.py", "exec")
    return text


def pose_input(model, config=None, input_size=None, seed=42):
    if input_size is None:
        cfg = pose_config(config)
        width, height = map(int, cfg.data_cfg.image_size)
        input_size = (1, 3, height, width)
    if (len(input_size) != 4 or any(isinstance(size, bool) or not isinstance(size, Integral)
                                  or size <= 0 for size in input_size) or input_size[1] != 3):
        raise ValueError("input_size must be positive [batch, 3, height, width]")
    parameter = next(model.parameters())
    images = torch.randn(tuple(input_size), generator=torch.Generator().manual_seed(seed))
    return images.to(device=parameter.device, dtype=parameter.dtype)


def get_model(config=None, checkpoint=None, device="auto", *, return_info=False):
    """Build ordinary ViTPose, or reload baseline/anchor/OFP weights strictly."""
    cfg = pose_config(config)
    device = resolve_device(device)
    if checkpoint is None:
        from mmpose.models import build_posenet
        model, info = build_posenet(cfg.model).to(device).eval(), {"stage": "ViTPose-base"}
    else:
        from .checkpoint import load_pose_checkpoint
        path = resolve_input_file(checkpoint, "checkpoint")
        model, info = load_pose_checkpoint(cfg, path, device)
    return (model, info) if return_info else model
