"""Shared SKP gates used by mask training and checkpoint helpers.

Keep these definitions outside the CLI module so ``python -m`` and package
imports use the same MaskLinear class for attachment and type checks.
"""

import math

import torch
from torch import nn


class TopKBinarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, threshold):
        threshold = torch.sigmoid(threshold).item()
        mask = inputs.clone()
        _, idx = inputs.sort(descending=True)
        j = math.ceil(threshold * inputs.numel())
        mask[idx[j:]] = 0.
        mask[idx[:j]] = 1.
        ctx.save_for_backward(mask)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output, ((grad_output * mask).sum()).view(-1)


class MaskLinear(nn.Linear):
    def __init__(self, weight, bias, mask, beta, mask_type="mlp", head_dim=64):
        super().__init__(weight.size(1), weight.size(0), bias=False)
        self.weight = weight
        self.bias = bias
        self.mask = mask
        self.beta = beta
        self.mask_type = mask_type
        self.head_dim = head_dim

    def forward(self, x):
        mask = TopKBinarizer.apply(self.mask, self.beta)
        if self.mask_type == "mha":
            mask = mask.repeat_interleave(self.head_dim)
        x = x * mask.view(1, 1, -1)
        return nn.functional.linear(x, self.weight, self.bias)


def get_skp_parameters(model):
    layers = [block.mlp.fc2 for block in model.backbone.blocks]
    if not all(isinstance(layer, MaskLinear) for layer in layers):
        raise ValueError("Expected an SKP mask in every MLP")
    return [layer.mask for layer in layers], [layer.beta for layer in layers]
