"""Numerical, checkpoint and reporting checks for the adapted package."""


import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from ..utils.pose_results import (
    ABLATIONS, ablation_cases, comparison_rows, parse_edges, validate_provenance, write_comparison,
)
from ..utils.io import atomic_json, atomic_text, fingerprint
from ..utils.utils import latency_statistics

def must_fail(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, TypeError):
        return
    raise AssertionError("Expected invalid input to be rejected")


def check_reports():
    timings = latency_statistics([1, 2, 3, 4])
    assert timings["median_ms"] == 2.5 and abs(timings["p90_ms"] - 3.7) < 1e-12
    for values in ([], [0], [-1], [float("nan")], [float("inf")]):
        must_fail(latency_statistics, values)
    baseline = {"label": "baseline", "ap": 0.8, "parameters": 1000, "pose_macs": 10000,
                "checkpoint_bytes": 8192, "state_bytes": 4096,
                "latency": latency_statistics([4, 4, 4])}
    slower = dict(baseline, label="smaller_but_slower", ap=0.79, parameters=500,
                  pose_macs=4000, checkpoint_bytes=4096, state_bytes=2048,
                  latency=latency_statistics([5, 5, 5]))
    result = comparison_rows([baseline, slower])[1]
    assert abs(result["delta_AP_points"] + 1) < 1e-12
    assert result["MAC_reduction_percent"] == 60 and result["parameter_reduction_percent"] == 50
    assert result["speedup"] == 0.8 and result["device"] == "cpu"
    assert abs(result["matrix_GFLOPs_2_per_MAC"] - 2 * result["GMACs"]) < 1e-12
    assert comparison_rows([dict(baseline, ap=0), dict(slower, ap=0.1)])[1]["delta_AP_points"] == 10
    assert comparison_rows([dict(baseline, ap=None), dict(slower, ap=None)])[1]["AP"] is None
    must_fail(comparison_rows, [dict(baseline, ap=float("nan"))])
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        write_comparison(directory, [baseline, slower], {"smoke_test": False}, "test fixture")
        saved = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
        assert saved["summary"][1]["speedup"] == 0.8
        assert "speedup" in (directory / "comparison.csv").read_text(encoding="utf-8")
        assert "0.80x" in (directory / "comparison.md").read_text(encoding="utf-8")
        gpu_rows = [dict(row, latency=dict(row["latency"], device="cuda:0"))
                    for row in (baseline, slower)]
        write_comparison(directory, gpu_rows, {"device": "cuda:0"}, "CUDA fixture")
        markdown = (directory / "comparison.md").read_text(encoding="utf-8")
        assert "Device: cuda:0" in markdown and "CPU speedup" not in markdown
        must_fail(comparison_rows, [baseline, gpu_rows[1]])
        path = directory / "atomic.txt"
        atomic_text(path, "keep this\n")
        must_fail(atomic_text, path, None)
        assert path.read_text(encoding="utf-8") == "keep this\n"
        atomic_json(directory / "unicode.json", {"species": "dog", "note": "한글"})
        assert json.loads((directory / "unicode.json").read_text(encoding="utf-8"))["note"] == "한글"
    cases = ablation_cases([0.4, 0.6], list(ABLATIONS))
    assert len(cases) == 10
    assert ABLATIONS["full_nuwa"] == {"skp": True, "attention_svd": True, "reconstruction": True}
    assert ABLATIONS["activation"] == {"skp": False, "attention_svd": False, "reconstruction": False}
    assert ABLATIONS["reconstruction"] == {"skp": False, "attention_svd": False, "reconstruction": True}
    assert ABLATIONS["ofp_without_skp"]["skp"] is False
    assert ABLATIONS["without_compensation"]["reconstruction"] is False
    must_fail(ablation_cases, [0.4, 0.4], ["full_nuwa"])
    must_fail(ablation_cases, [1.0], ["full_nuwa"])
    assert parse_edges(["full=outputs/run/edge.pth"])[0]["label"] == "full"
    must_fail(parse_edges, ["A=one.pth", "a=two.pth"])
    must_fail(parse_edges, ["baseline=one.pth"])
    must_fail(parse_edges, ["../bad=one.pth"])
    anchor = {"stage": "NuWa-SKP-anchor", "species": "dog", "split": 1,
              "base_checkpoint_sha256": "base"}
    edge = {"stage": "NuWa-OFP", "species": "dog", "split": 1, "source_anchor_sha256": "anchor"}
    validate_provenance(anchor, "anchor", "dog", 1, {"base"}, {"anchor"})
    validate_provenance(edge, "edge", "dog", 1, {"base"}, {"anchor"})
    validate_provenance(dict(edge, source_anchor_sha256=None, base_checkpoint_sha256="base"),
                        "edge", "dog", 1, {"base"}, {"anchor"})
    for bad in (dict(edge, species="cat"), dict(edge, split=2),
                dict(edge, source_anchor_sha256="other"), dict(edge, base_checkpoint_sha256="other")):
        must_fail(validate_provenance, bad, "edge", "dog", 1, {"base"}, {"anchor"})
    print("AP deltas, missing AP, actual speedups, units, atomic reports, provenance and ablation plans: PASS")


def check_torch():
    import types
    import torch
    from torch import nn
    from ..model.checkpoint import ANCHOR_STAGE
    from ..model.checkpoint import OFP_STAGE
    from ..model.vit import AttentionPruned, attention_specs
    from ..model.checkpoint import (
        BASE_STAGE, checkpoint_header, export_portable, load_pose_checkpoint, verify_reload,
    )

    class DenseAttention(nn.Module):
        # A distinct native type checks that skipped SVD remains native on reload.
        def __init__(self):
            super().__init__()
            self.num_heads, self.scale = 2, 4**-0.5
            self.qkv, self.proj = nn.Linear(8, 24), nn.Linear(8, 8)
            self.attn_drop, self.proj_drop = nn.Dropout(0), nn.Dropout(0)

        def forward(self, x):
            batch, tokens, width = x.shape
            q, k, v = self.qkv(x).reshape(batch, tokens, 3, 2, 4).permute(2, 0, 3, 1, 4)
            attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            value = (attention @ v).transpose(1, 2).reshape(batch, tokens, width)
            return self.proj(value)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = nn.Conv2d(3, 8, 1)
            block = nn.Module()
            block.attn = DenseAttention()
            block.mlp = nn.Module()
            block.mlp.fc1, block.mlp.fc2 = nn.Linear(8, 16), nn.Linear(16, 8)
            block.mlp.act = nn.GELU()
            self.blocks = nn.ModuleList([block])

        def forward(self, images):
            value = self.patch(images)
            batch, width, height, columns = value.shape
            value = value.flatten(2).transpose(1, 2)
            block = self.blocks[0]
            value = value + block.attn(value)
            value = value + block.mlp.fc2(block.mlp.act(block.mlp.fc1(value)))
            return value.transpose(1, 2).reshape(batch, width, height, columns)

    class Pose(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.keypoint_head = nn.Conv2d(8, 17, 1)

    cfg = types.SimpleNamespace(model={}, data_cfg=types.SimpleNamespace(
        image_size=[4, 4], heatmap_size=[4, 4], num_joints=17))
    parent, child = types.ModuleType("mmpose"), types.ModuleType("mmpose.models")
    parent.__path__ = []
    child.build_posenet = lambda config: Pose()
    parent.models = child
    original = {key: sys.modules.get(key) for key in ("mmpose", "mmpose.models")}
    sys.modules.update({"mmpose": parent, "mmpose.models": child})
    torch.manual_seed(9)
    images = torch.randn(1, 3, 4, 4)
    try:
        with tempfile.TemporaryDirectory() as directory:
            for kind in ("base", "anchor", "ofp_dense", "ofp_pruned"):
                model = Pose().eval()
                if kind != "base":
                    model.backbone.blocks[0].mlp.fc1 = nn.Linear(8, 8)
                    model.backbone.blocks[0].mlp.fc2 = nn.Linear(8, 8)
                if kind == "ofp_pruned":
                    model.backbone.blocks[0].attn = AttentionPruned(8, 2, 2, 3)
                model.eval()
                info = {"stage": BASE_STAGE if kind == "base" else ANCHOR_STAGE if kind == "anchor" else OFP_STAGE,
                        "format_version": 1, "architecture": {"original_hidden_sizes": [16],
                        "hidden_sizes": [16 if kind == "base" else 8], "attention": attention_specs(model)}}
                if kind == "ofp_dense":
                    info["architecture"]["attention"][0]["kind"] = "dense"
                info["state_dict"] = model.state_dict()
                source, output = Path(directory) / (kind + "_source.pth"), Path(directory) / (kind + ".pth")
                torch.save(model.state_dict() if kind == "base" else info, source)
                restored, metadata = load_pose_checkpoint(cfg, source, "cpu")
                export_portable(restored, metadata, cfg, source, output)
                source.unlink()  # The exported checkpoint must work independently.
                assert verify_reload(restored, cfg, output, images)["passed"]
                assert checkpoint_header(output)["stage"] == info["stage"]
                reloaded, _ = load_pose_checkpoint(cfg, output, "cpu")
                if kind == "ofp_dense":
                    assert isinstance(reloaded.backbone.blocks[0].attn, DenseAttention)
                wrong = copy.deepcopy(cfg)
                wrong.data_cfg.num_joints = 16
                must_fail(load_pose_checkpoint, wrong, output, "cpu")
    finally:
        for key, module in original.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module
    print("PyTorch base/anchor/OFP portable reload, dense attention and geometry checks: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-only", action="store_true")
    args = parser.parse_args()
    check_reports()
    if not args.math_only:
        check_torch()


if __name__ == "__main__":
    main()
