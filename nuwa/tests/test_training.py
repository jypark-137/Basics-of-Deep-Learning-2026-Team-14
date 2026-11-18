"""Verify optional pose fine-tuning and validation-best checkpoint selection."""

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..engine.train import train
from ..model.checkpoint import checkpoint_header, export_portable


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.qkv, self.proj = nn.Linear(8, 24), nn.Linear(8, 8)

    def forward(self, value):
        return self.proj(self.qkv(value)[..., :8])


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 8, 1)
        block = nn.Module()
        block.attn, block.mlp = TinyAttention(), nn.Module()
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


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 17, 1)

    def forward(self, value):
        return self.conv(value)

    def get_loss(self, actual, target, weights):
        return {"heatmap_loss": ((actual - target) * weights.unsqueeze(-1)).square().mean()}


class TinyPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.keypoint_head = TinyBackbone(), TinyHead()


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = TinyPose().requires_grad_(False)
        self.loader = DataLoader([{"img": torch.randn(3, 2, 2), "target": torch.randn(17, 2, 2),
                                   "target_weight": torch.ones(17, 1)}], batch_size=1)
        self.cfg = SimpleNamespace(model={}, data_cfg=SimpleNamespace(
            image_size=[2, 2], heatmap_size=[2, 2], num_joints=17))

    def options(self, directory, smoke=False):
        return argparse.Namespace(cfg=self.cfg, out_dir=directory, device="cpu", steps=2,
                                  learning_rate=1e-3, weight_decay=0.05, eval_every=1,
                                  log_every=10, smoke_test=smoke, species="dog", split_id=1)

    def test_best_and_last_are_distinct(self):
        with tempfile.TemporaryDirectory(prefix="pose_training_") as directory:
            with patch("nuwa.engine.train.evaluate", side_effect=[80.0, 90.0, 70.0]):
                ap = train(self.options(directory), self.model, self.loader, self.loader,
                           early_stop=False, best=True)
            self.assertEqual(ap, 90.0)
            best = torch.load(str(Path(directory) / "best.pth"), map_location="cpu")
            last = torch.load(str(Path(directory) / "last.pth"), map_location="cpu")
            self.assertEqual((best["finetune_step"], last["finetune_step"]), (1, 2))
            self.assertEqual((best["finetune_val_ap"], last["finetune_val_ap"]), (0.9, 0.7))
            self.assertTrue(any(not torch.equal(value, last["state_dict"][key])
                                for key, value in best["state_dict"].items()))
            restored = TinyPose()
            restored.load_state_dict(best["state_dict"], strict=True)
            portable = Path(directory) / "portable.pth"
            export_portable(restored, best, self.cfg, Path(directory) / "best.pth", portable)
            header = checkpoint_header(portable)
            self.assertTrue(header["finetuned"])
            self.assertEqual(header["finetune_step"], 1)
            self.assertEqual(header["finetune_val_ap"], 0.9)
            self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))

    def test_smoke_does_not_measure_ap(self):
        with tempfile.TemporaryDirectory(prefix="pose_training_smoke_") as directory:
            with patch("nuwa.engine.train.evaluate") as evaluator:
                result = train(self.options(directory, smoke=True), self.model, self.loader, None)
                evaluator.assert_not_called()
            self.assertIsNone(result)
            payload = torch.load(str(Path(directory) / "last.pth"), map_location="cpu")
            self.assertTrue(payload["smoke_test"])
            self.assertEqual(payload["finetune_step"], 10)
            self.assertIsNone(payload["finetune_val_ap"])


if __name__ == "__main__":
    unittest.main()
