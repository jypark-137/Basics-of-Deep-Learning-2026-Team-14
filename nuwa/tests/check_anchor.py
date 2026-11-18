"""Numerical, checkpoint and reporting checks for the adapted package."""


import copy
import io
import torch
from torch import nn
from ..model.checkpoint import (
    ANCHOR_STAGE, apply_saved_masks, prune_skp_mlp, restore_anchor_state,
)
from ..model.vit import hidden_sizes

class TinyMlp(nn.Module):
    def __init__(self, width, hidden, bias):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden, bias=bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden, width, bias=bias)

    def forward(self, value):
        return self.drop(self.fc2(self.act(self.fc1(value))))


class TinyBlock(nn.Module):
    def __init__(self, hidden, bias):
        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.mlp = TinyMlp(4, hidden, bias)

    def forward(self, value):
        value = value + self.attn(value)
        return value + self.mlp(value)


class TinyPose(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([TinyBlock(7, bias), TinyBlock(5, bias)])
        self.keypoint_head = nn.Linear(4, 17)

    def forward(self, value):
        for block in self.backbone.blocks:
            value = block(value)
        return self.keypoint_head(value)


def mask_payload(masks):
    return {"format_version": 1, "stage": "NuWa-SKP", "binary_mlp_masks": masks,
            "hidden_sizes": [int(mask.numel()) for mask in masks],
            "kept_neurons": [int(mask.sum().item()) for mask in masks]}


@torch.no_grad()
def main():
    torch.manual_seed(7)
    for bias in (True, False):
        for dtype in (torch.float32, torch.float64):
            model = TinyPose(bias).to(dtype=dtype).eval()
            base_state = {key: value.clone() for key, value in model.state_dict().items()}
            masks = mask_payload([torch.tensor([1., 0., 1., 0., 0., 1., 0.]),
                                  torch.tensor([0., 1., 0., 0., 0.])])
            images = torch.randn(2, 11, 4, dtype=dtype)
            with apply_saved_masks(model, masks):
                expected = model(images)
            before = sum(parameter.numel() for parameter in model.parameters())
            prune_skp_mlp(model, masks)
            assert hidden_sizes(model) == [3, 1]
            assert model(images).shape == (2, 11, 17)
            assert torch.allclose(model(images), expected, rtol=1e-5, atol=1e-6)
            removed = 8 * (4 + 4 + int(bias))
            assert sum(parameter.numel() for parameter in model.parameters()) == before - removed
            for key, value in model.state_dict().items():
                if ".mlp.fc1." not in key and not key.endswith("mlp.fc2.weight"):
                    assert torch.equal(value, base_state[key]), key
            assert torch.equal(model.backbone.blocks[1].mlp.fc1.weight, base_state[
                "backbone.blocks.1.mlp.fc1.weight"][[1]])
            assert all(parameter.dtype == dtype for parameter in model.parameters())
            assert not any(name.endswith((".mask", ".beta")) for name in model.state_dict())
            payload = {"format_version": 1, "stage": ANCHOR_STAGE,
                       "architecture": {"original_hidden_sizes": [7, 5], "hidden_sizes": [3, 1]},
                       "state_dict": model.state_dict()}
            stream = io.BytesIO()
            torch.save(payload, stream)
            stream.seek(0)
            restored = restore_anchor_state(TinyPose(bias).to(dtype=dtype),
                                            torch.load(stream, map_location="cpu")).eval()
            assert torch.equal(restored(images), model(images))
    print("Slicing, one-neuron blocks, biases, dtypes, heatmaps and reload: PASS")

    model = TinyPose().eval()
    reference = copy.deepcopy(model.state_dict())
    masks = mask_payload([torch.ones(7), torch.ones(5)])
    prune_skp_mlp(model, masks)
    assert all(torch.equal(value, reference[key]) for key, value in model.state_dict().items())
    # Invalid masks in a later block must not partially prune earlier blocks.
    for bad in (torch.tensor([0.5, 1., 0., 0., 0.]), torch.tensor([float("nan"), 1., 0., 0., 0.]),
                torch.ones(4), torch.zeros(5)):
        model = TinyPose().eval()
        bad_payload = mask_payload([torch.tensor([1., 0., 1., 0., 0., 0., 0.]), torch.ones(5)])
        bad_payload["binary_mlp_masks"][1] = bad
        try:
            prune_skp_mlp(model, bad_payload)
        except ValueError:
            assert hidden_sizes(model) == [7, 5]
        else:
            raise AssertionError("Accepted an invalid binary mask")
    # Hook cleanup also holds on an interrupted reference forward.
    model = TinyPose().eval()
    try:
        with apply_saved_masks(model, mask_payload([torch.ones(7), torch.ones(5)])):
            raise RuntimeError("synthetic interruption")
    except RuntimeError:
        pass
    assert all(not mlp.fc2._forward_pre_hooks for mlp in
               [block.mlp for block in model.backbone.blocks])
    print("All-kept case, invalid-mask rejection and hook cleanup: PASS")


if __name__ == "__main__":
    main()
