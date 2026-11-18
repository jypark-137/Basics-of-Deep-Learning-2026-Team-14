"""Regressions for module execution and retries after mask initialization fails."""

from contextlib import redirect_stderr
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from ..method.get_anchor_model import get_args_parser, get_mask, run
from ..method.skp import MaskLinear
from ..paths import PROJECT_ROOT
from ..utils.io import file_sha256
from .test_training import TinyPose


class AnchorTrainingChecks(unittest.TestCase):
    def test_module_entry_point_self_test(self):
        result = subprocess.run(
            [sys.executable, "-m", "nuwa.method.get_anchor_model", "--self-test",
             "--device", "cpu", "--threads", "2"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Vanilla mask rule, STE and checkpoint checks: PASS", result.stdout)

    def options(self, root, model):
        config, checkpoint = root / "config.py", root / "base.pth"
        config.write_text(
            "model = {}\n"
            "data_cfg = dict(image_size=[2, 2], heatmap_size=[2, 2], num_joints=17)\n",
            encoding="utf-8")
        torch.save(model.state_dict(), checkpoint)
        return get_args_parser().parse_args([
            "--config", str(config), "--checkpoint", str(checkpoint),
            "--species", "dog", "--device", "cpu", "--threads", "2",
            "--smoke-test", "--out-dir", str(root / "masks")])

    def completed_anchor(self, root):
        torch.manual_seed(17)
        args = self.options(root, TinyPose())
        loader = DataLoader([{
            "img": torch.randn(3, 2, 2), "target": torch.randn(17, 2, 2),
            "target_weight": torch.ones(17, 1)}], batch_size=1)
        with patch("mmpose.models.build_posenet", side_effect=lambda config: TinyPose()), \
                patch("nuwa.method.get_anchor_model.validate_subsets", return_value=(None, {}, [8])), \
                patch("nuwa.method.get_anchor_model.make_loaders", return_value=(loader, None)):
            report = run(copy.deepcopy(args))
        return args, report

    def output_snapshot(self, directory):
        return {path.name: (file_sha256(path), path.stat().st_mtime_ns)
                for path in directory.iterdir() if path.is_file()}

    def test_completed_anchor_is_verified_and_reused_without_writes(self):
        with tempfile.TemporaryDirectory(prefix="nuwa_anchor_complete_") as temporary:
            args, report = self.completed_anchor(Path(temporary))
            output = Path(args.out_dir)
            before = self.output_snapshot(output)
            with patch("mmpose.models.build_posenet", side_effect=lambda config: TinyPose()), \
                    patch("nuwa.method.get_anchor_model.make_loaders") as loaders, \
                    patch("nuwa.method.get_anchor_model.train_mask") as train:
                reused_report = run(copy.deepcopy(args))
            self.assertEqual(reused_report, report)
            self.assertEqual(self.output_snapshot(output), before)
            loaders.assert_not_called()
            train.assert_not_called()

    def test_completed_anchor_rejects_different_training_settings(self):
        with tempfile.TemporaryDirectory(prefix="nuwa_anchor_mismatch_") as temporary:
            args, _ = self.completed_anchor(Path(temporary))
            output = Path(args.out_dir)
            before = self.output_snapshot(output)
            for name, value in (("split", 2), ("mask_lr", 0.01)):
                changed = copy.deepcopy(args)
                setattr(changed, name, value)
                error = io.StringIO()
                with self.subTest(setting=name), \
                        patch("mmpose.models.build_posenet", side_effect=lambda config: TinyPose()), \
                        redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                    run(changed)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("training settings differ ({})".format(name), error.getvalue())
                self.assertEqual(self.output_snapshot(output), before)

    def test_completed_anchor_rejects_modified_weights(self):
        with tempfile.TemporaryDirectory(prefix="nuwa_anchor_modified_") as temporary:
            args, _ = self.completed_anchor(Path(temporary))
            output = Path(args.out_dir)
            anchor = output / "anchor.pth"
            payload = torch.load(str(anchor), map_location="cpu")
            payload["state_dict"]["keypoint_head.conv.bias"][0] += 1
            torch.save(payload, str(anchor))
            before = self.output_snapshot(output)
            error = io.StringIO()
            with patch("mmpose.models.build_posenet", side_effect=lambda config: TinyPose()), \
                    redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                run(copy.deepcopy(args))
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Existing anchor weights differ", error.getvalue())
            self.assertEqual(self.output_snapshot(output), before)

    def test_smoke_training_retries_empty_or_config_only_output(self):
        torch.set_num_threads(1)
        for previous_config in (False, True):
            with self.subTest(previous_config=previous_config), \
                    tempfile.TemporaryDirectory(prefix="nuwa_anchor_retry_") as temporary:
                torch.manual_seed(17)
                root = Path(temporary)
                model = TinyPose().eval()
                args = self.options(root, model)
                output = Path(args.out_dir)
                output.mkdir()
                if previous_config:
                    (output / "run_config.json").write_text("{}\n", encoding="utf-8")
                loader = DataLoader([{
                    "img": torch.randn(3, 2, 2), "target": torch.randn(17, 2, 2),
                    "target_weight": torch.ones(17, 1)}], batch_size=1)
                with patch("nuwa.method.get_anchor_model.evaluate_ap") as evaluate:
                    payload = get_mask(args, model, loader, None)
                evaluate.assert_not_called()
                self.assertEqual(payload["step"], 10)
                self.assertEqual(payload["stage"], "NuWa-SKP")
                self.assertTrue((output / "last_masks.pth").is_file())
                self.assertTrue((output / "history.jsonl").is_file())
                self.assertFalse(isinstance(model.backbone.blocks[0].mlp.fc2, MaskLinear))

    def test_existing_results_are_rejected_before_loading_data(self):
        with tempfile.TemporaryDirectory(prefix="nuwa_anchor_existing_") as temporary:
            root = Path(temporary)
            args = self.options(root, TinyPose())
            output = Path(args.out_dir)
            output.mkdir()
            history = output / "history.jsonl"
            history.write_text("previous training results\n", encoding="utf-8")
            error = io.StringIO()
            with patch("nuwa.method.get_anchor_model.validate_subsets") as validate, \
                    redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                run(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("choose another --out-dir", error.getvalue())
            self.assertEqual(history.read_text(encoding="utf-8"), "previous training results\n")
            validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
