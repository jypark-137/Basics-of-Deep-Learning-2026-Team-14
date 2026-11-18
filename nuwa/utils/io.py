"""Shared atomic I/O and checkpoint fingerprints."""


import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_state_sha256(model, original_keys):
    """Check original parameters AND buffers, excluding added mask tensors."""
    state = model.state_dict()
    digest = hashlib.sha256()
    for name in original_keys:
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=str(path.parent), delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def state_size_bytes(model):
    return sum(value.numel() * value.element_size() for value in model.state_dict().values())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                      allow_nan=False).encode("utf-8")).hexdigest()
