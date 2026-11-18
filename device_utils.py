"""Shared CPU/CUDA selection for the NuWa and bridge entry points."""

def add_device_argument(parser):
    return parser.add_argument(
        "--device", default="auto",
        help="auto (default: CUDA if available, otherwise CPU), cpu, cuda, or cuda:N")


def resolve_device(value="auto", model=None):
    """Select a visible CUDA device or CPU; None preserves an existing model's device.

    CUDA_VISIBLE_DEVICES also applies to MIG instances: indices here refer to
    devices visible to this process, not physical GPU numbers on the host.
    An explicitly requested CUDA device never silently falls back to CPU.
    """
    import torch

    if value is None:
        parameter = next(model.parameters(), None) if model is not None else None
        value = parameter.device if parameter is not None else "auto"
    if str(value) == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("Use --device auto, cpu, cuda, or cuda:N") from error
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use --device auto, cpu, cuda, or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu or auto")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError("CUDA device {} is not visible to this process".format(index))
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
        # Masked/dense and physically pruned layers must agree in FP32 during
        # export checks, including on Ampere GPUs with TF32-capable kernels.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


def device_info(device="auto"):
    """Record the selected device, including the memory exposed by a MIG slice."""
    import torch

    device = resolve_device(device)
    result = {"device": str(device), "type": device.type, "name": "CPU"}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update({"name": properties.name, "total_memory_bytes": properties.total_memory,
                       "compute_capability": [properties.major, properties.minor],
                       "torch_cuda": torch.version.cuda,
                       "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32})
    return result
