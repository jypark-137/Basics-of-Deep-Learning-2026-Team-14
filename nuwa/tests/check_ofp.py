"""Numerical, checkpoint and reporting checks for the adapted package."""


import argparse
import io
import numpy as np
from ..method.pruning import (
    compensate_fc2, energy_ratio, factor_attention, product_svd,
    retained_dimension, uniform_widths,
)

def numpy_attention(x, weight, bias, output, output_bias, heads, qdim, vdim):
    packed = x @ weight.T
    if bias is not None:
        packed += bias
    q, k, v = np.split(packed, [heads * qdim, 2 * heads * qdim], axis=-1)
    def split(value, dim):
        return value.reshape(x.shape[0], x.shape[1], heads, dim).transpose(0, 2, 1, 3)
    q, k, v = split(q, qdim), split(k, qdim), split(v, vdim)
    logits = (q * qdim**-0.5) @ k.transpose(0, 1, 3, 2)
    exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
    value = (probabilities @ v).transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], heads * vdim)
    result = value @ output.T
    return result if output_bias is None else result + output_bias


def check_math():
    rng = np.random.default_rng(5)
    for dtype, tolerance in ((np.float32, 2e-4), (np.float64, 1e-10)):
        left = rng.normal(size=(17, 4)).astype(dtype)
        right = rng.normal(size=(4, 13)).astype(dtype)
        u, singular, vh = product_svd(left, right)
        np.testing.assert_allclose((u * singular) @ vh, left @ right, rtol=tolerance, atol=tolerance)
        # Full-rank bias-augmented factorization must preserve complete attention.
        width, heads = 8, 2
        wqkv = (rng.normal(size=(3 * width, width)) * 0.2).astype(dtype)
        bqkv = rng.normal(size=3 * width).astype(dtype)
        wo = (rng.normal(size=(width, width)) * 0.2).astype(dtype)
        bo = rng.normal(size=width).astype(dtype)
        x = rng.normal(size=(2, 9, width)).astype(dtype)
        factored = factor_attention(wqkv, bqkv, wo, bo, heads, 1.0, dimensions=(4, 4))
        expected = numpy_attention(x, wqkv, bqkv, wo, bo, heads, 4, 4)
        actual = numpy_attention(x, factored["qkv_weight"], factored["qkv_bias"],
                                 factored["out_weight"], factored["out_bias"], heads, 4, 4)
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        # Truncated QK factors must preserve the truncated bilinear product with
        # the old attention scale, including all affine bias terms.
        reduced = factor_attention(wqkv, bqkv, wo, bo, heads, 0.9, dimensions=(2, 3))
        q, k, v = wqkv.reshape(3, heads, 4, width)
        bq, bk, bv = bqkv.reshape(3, heads, 4)
        packed = np.concatenate((reduced["qkv_weight"], reduced["qkv_bias"][:, None]), axis=1)
        for head in range(heads):
            original_product = np.concatenate((q[head], bq[head, :, None]), axis=1).T @ np.concatenate((k[head], bk[head, :, None]), axis=1)
            uu, ss, vv = np.linalg.svd(original_product, full_matrices=False)
            best = (uu[:, :2] * ss[:2]) @ vv[:2]
            query = packed[head * 2:(head + 1) * 2]
            key = packed[heads * 2 + head * 2:heads * 2 + (head + 1) * 2]
            np.testing.assert_allclose((query.T @ key) / np.sqrt(2), best / np.sqrt(4),
                                       rtol=tolerance, atol=tolerance)
            vbar = np.concatenate((v[head], bv[head, :, None]), axis=1)
            old_vo = vbar.T @ wo.T.reshape(heads, 4, width)[head]
            uv, sv, vhv = np.linalg.svd(old_vo, full_matrices=False)
            best_vo = (uv[:, :3] * sv[:3]) @ vhv[:3]
            value = packed[2 * heads * 2 + head * 3:2 * heads * 2 + (head + 1) * 3]
            output = reduced["out_weight"][:, head * 3:(head + 1) * 3]
            np.testing.assert_allclose(value.T @ output.T, best_vo, rtol=tolerance, atol=tolerance)
        h = rng.normal(size=(80, 7)).astype(dtype)
        weights = rng.normal(size=(4, 7)).astype(dtype)
        result, diagnostics = compensate_fc2(h, weights, np.array([0, 2, 5]))
        direct = np.linalg.lstsq(h[:, [0, 2, 5]], h @ weights.T, rcond=diagnostics["pinv_rcond"])[0].T
        np.testing.assert_allclose(result, direct, rtol=tolerance, atol=tolerance)
        assert diagnostics["calibration_mse_after"] <= diagnostics["calibration_mse_before"] + tolerance
        # Rank-deficient calibration must remain finite without a ridge penalty.
        h[:, 2] = h[:, 0]
        rank_deficient, _ = compensate_fc2(h, weights, np.array([0, 2, 5]))
        assert np.isfinite(rank_deficient).all()
    assert abs(energy_ratio(0.4) - 0.98416) < 1e-12
    assert uniform_widths([16, 32, 48], 24) == [16, 24, 24]
    assert uniform_widths([16, 32, 48], 0) == [16, 32, 48]
    dimension, fraction = retained_dimension([np.array([4.] + [1.] * 15)] * 2, 16, 0.75)
    assert dimension == 8 and 0 < fraction <= 1
    assert retained_dimension([np.zeros(16)], 16, 0.9)[0] == 8
    for _ in range(30):
        widths = rng.integers(8, 100, size=5).tolist()
        prune = int(rng.integers(1, sum(widths) - 8 * len(widths) + 1))
        targets = uniform_widths(widths, prune)
        assert all(8 <= new <= old and new % 8 == 0 for old, new in zip(widths, targets))
        assert prune <= sum(widths) - sum(targets) < prune + 8 * len(widths)
    aligned = factor_attention(rng.normal(size=(96, 32)), rng.normal(size=96),
                               rng.normal(size=(32, 32)), rng.normal(size=32), 2, energy_ratio(0.4))
    assert aligned["qk_dim"] % 8 == aligned["vo_dim"] % 8 == 0
    assert aligned["qkv_weight"].shape == (2 * (2 * aligned["qk_dim"] + aligned["vo_dim"]), 32)
    zero = factor_attention(np.zeros((96, 32)), np.zeros(96), np.zeros((32, 32)), np.ones(32), 2, 0.9)
    assert zero["qk_dim"] == zero["vo_dim"] == 8
    assert np.isfinite(zero["qkv_weight"]).all()
    for invalid in (-0.1, 0, 1, float("nan")):
        try:
            energy_ratio(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted an invalid pruning rate")
    try:
        uniform_widths([16, 32], 40)
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted an infeasible MLP budget")
    print("SVD, affine biases, attention scaling, energy ranks, uniform widths and compensation: PASS")


def check_torch():
    import torch
    from torch import nn
    from ..model.checkpoint import OFP_STAGE, restore_ofp_state
    from ..model.vit import AttentionPruned, attention_specs
    from ..utils.utils import profile_pose_macs

    class TinyPose(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            block = nn.Module()
            block.attn = AttentionPruned(8, 2, 4, 4)
            block.mlp = nn.Module()
            block.mlp.fc1 = nn.Linear(8, 16)
            block.mlp.act = nn.GELU()
            block.mlp.fc2 = nn.Linear(16, 8)
            self.backbone.blocks = nn.ModuleList([block])
            self.keypoint_head = nn.Linear(8, 17)

        def forward(self, x):
            block = self.backbone.blocks[0]
            x = x + block.attn(x)
            return self.keypoint_head(x + block.mlp.fc2(block.mlp.act(block.mlp.fc1(x))))

    torch.manual_seed(5)
    model = TinyPose().double().eval()
    x = torch.randn(1, 9, 8, dtype=torch.float64)
    attn = model.backbone.blocks[0].attn
    reference = numpy_attention(x.numpy(), attn.qkv.weight.detach().numpy(), attn.qkv.bias.detach().numpy(),
                                 attn.proj.weight.detach().numpy(), attn.proj.bias.detach().numpy(), 2, 4, 4)
    np.testing.assert_allclose(attn(x).detach().numpy(), reference, rtol=1e-10, atol=1e-10)
    profile, _ = profile_pose_macs(model, x, lambda module, image: module(image))
    expected_macs = 9 * 8 * 24 + 9 * 8 * 8 + 2 * 9 * 9 * 8 + 2 * 9 * 8 * 16 + 9 * 8 * 17
    assert profile["pose_macs"] == expected_macs
    assert profile["per_neuron_macs"] == [2 * 9 * 8]
    # Restore unequal QK/VO dimensions and smaller MLPs from only the state dict.
    block = model.backbone.blocks[0]
    block.attn = AttentionPruned(8, 2, 2, 3).double()
    block.mlp.fc1 = nn.Linear(8, 8).double()
    block.mlp.fc2 = nn.Linear(8, 8).double()
    model.eval()
    payload = {"format_version": 1, "stage": OFP_STAGE,
               "architecture": {"original_hidden_sizes": [16], "hidden_sizes": [8],
                                "attention": attention_specs(model)},
               "state_dict": model.state_dict()}
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    restored = restore_ofp_state(TinyPose().double(), torch.load(stream, map_location="cpu")).eval()
    assert torch.equal(restored(x), model(x))
    assert model(x).shape == (1, 9, 17)
    print("PyTorch attention, matrix MACs, unequal widths and standalone reload: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-only", action="store_true")
    args = parser.parse_args()
    check_math()
    if not args.math_only:
        check_torch()


if __name__ == "__main__":
    main()
