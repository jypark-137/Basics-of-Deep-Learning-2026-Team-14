"""CPU fallback, visible CUDA devices, batches, gradients and portable reload."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import main
from ..dataset.utils import move_batch
from ..method.get_anchor_model import attach_skp_masks, check_mask_gradients, detach_skp_masks
from ..method.utils import export_masks
from ..model.checkpoint import ANCHOR_STAGE, export_portable, load_pose_checkpoint, prune_skp_mlp, verify_reload
from device_utils import resolve_device
from ..model.vit import forward_heatmaps, heatmap_task_loss
from ..utils.utils import benchmark_pose
from .test_training import TinyPose


class DeviceChecks(unittest.TestCase):
    def test_auto_falls_back_and_explicit_cuda_fails_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.set_device") as select:
            self.assertEqual(resolve_device(), torch.device("cpu"))
            self.assertEqual(resolve_device(None), torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                resolve_device("cuda")
            select.assert_not_called()

    def test_visible_indices_and_auto_cuda(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device_count", return_value=2), patch("torch.cuda.set_device") as select:
            self.assertEqual(resolve_device(), torch.device("cuda:0"))
            self.assertEqual(resolve_device("cuda"), torch.device("cuda:0"))
            self.assertEqual(resolve_device("cuda:1"), torch.device("cuda:1"))
            select.assert_called_with(torch.device("cuda:1"))
            with self.assertRaisesRegex(ValueError, "not visible"):
                resolve_device("cuda:2")

    def test_explicit_cpu_preserves_cpu_model(self):
        with patch("torch.cuda.is_available", side_effect=AssertionError("CPU must remain usable")):
            self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(resolve_device(None, TinyPose()), torch.device("cpu"))
            with self.assertRaises(ValueError):
                resolve_device("mps")

    def test_all_model_commands_default_to_auto_and_accept_cpu(self):
        required = {"prune": ["--anchor", "anchor.pth", "--pruning-rate", "0.4"],
                    "compare": ["--anchor", "anchor.pth", "--edge", "edge=edge.pth"],
                    "ablate": ["--anchor", "anchor.pth", "--rates", "0.4"]}
        parser = main.get_args_parser()
        for command in main.COMMANDS:
            if command in ("prepare", "self-test"):
                continue
            arguments = [command] + required.get(command, [])
            with self.subTest(command=command):
                self.assertEqual(parser.parse_args(arguments).device, "auto")
                self.assertEqual(parser.parse_args(arguments + ["--device", "cpu"]).device, "cpu")

    def check_pose_pipeline(self, selected):
        from mmcv.parallel import DataContainer, collate

        torch.set_num_threads(1)
        device = resolve_device(selected)
        model = TinyPose().to(device).eval()
        batch = collate([{"img": torch.randn(3, 2, 2), "target": torch.randn(17, 2, 2),
                          "target_weight": torch.ones(17, 1),
                          "img_metas": DataContainer({"image_id": 7}, cpu_only=True)}], samples_per_gpu=1)
        data = move_batch(batch, device)
        self.assertEqual(data["img"].device, device)
        self.assertEqual(data["target"].device, device)
        self.assertEqual(data["img_metas"], [{"image_id": 7}])
        masks, betas = attach_skp_masks(model)
        with torch.no_grad():
            masks[0].copy_(torch.arange(16, device=device))
            betas[0].zero_()
        heatmaps = forward_heatmaps(model, data["img"])
        heatmap_task_loss(model, heatmaps, data["target"], data["target_weight"]).backward()
        check_mask_gradients(model)
        self.assertTrue(all(parameter.grad.device == device for parameter in masks + betas))
        payload = dict(export_masks(model), format_version=1, stage="NuWa-SKP")
        expected = heatmaps.detach()
        model = prune_skp_mlp(detach_skp_masks(model), payload)
        self.assertTrue(all(parameter.device == device for parameter in model.parameters()))
        self.assertTrue(torch.allclose(forward_heatmaps(model, data["img"]), expected, rtol=1e-4, atol=1e-5))
        timing = benchmark_pose(model, data["img"], 1, 2)
        self.assertEqual(timing["device"], str(device))
        self.assertGreater(timing["median_ms"], 0)
        self.assertEqual(timing["peak_memory_GiB"] is not None, device.type == "cuda")
        cfg = SimpleNamespace(model={}, data_cfg=SimpleNamespace(
            image_size=[2, 2], heatmap_size=[2, 2], num_joints=17))
        info = {"stage": ANCHOR_STAGE, "architecture": {"original_hidden_sizes": [16]}}
        with tempfile.TemporaryDirectory(prefix="nuwa_device_") as directory:
            source, output = Path(directory) / "source.pth", Path(directory) / "portable.pth"
            torch.save(model.state_dict(), source)
            portable = export_portable(model, info, cfg, source, output)
            self.assertTrue(all(value.device.type == "cpu" for value in portable["state_dict"].values()))
            with patch("mmpose.models.build_posenet", side_effect=lambda config: TinyPose()):
                self.assertTrue(verify_reload(model, cfg, output, data["img"])["passed"])
                restored, _ = load_pose_checkpoint(cfg, output, "cpu")
            self.assertEqual(next(restored.parameters()).device.type, "cpu")
            self.assertTrue(torch.allclose(forward_heatmaps(restored, data["img"].cpu()),
                                           expected.cpu(), rtol=1e-4, atol=1e-5))

    def test_cpu_pose_pipeline(self):
        self.check_pose_pipeline("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA device is available")
    def test_cuda_pose_pipeline_and_cpu_reload(self):
        self.check_pose_pipeline("cuda:0")


if __name__ == "__main__":
    unittest.main()
