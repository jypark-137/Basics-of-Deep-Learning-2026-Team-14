"""Verify scheduler endpoints and the rates actually used by both optimizers."""

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

import main
from ..engine.scheduler import (
    SCHEDULER_DEFAULTS, lr_multiplier, make_scheduler, scheduler_settings,
)
from ..engine.train import train
from ..method.get_anchor_model import (
    _check_completed_training_settings, get_args_parser, get_mask,
)
from .test_training import TinyPose


def recording_adamw():
    observed = []

    class RecordingAdamW(torch.optim.AdamW):
        def step(self, *args, **kwargs):
            observed.append([group["lr"] for group in self.param_groups])
            return super().step(*args, **kwargs)

    return RecordingAdamW, observed


class SchedulerChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def options(self, **overrides):
        return SimpleNamespace(**dict(SCHEDULER_DEFAULTS, **overrides))

    def test_warmup_decay_and_parameter_group_ratio(self):
        optimizer = torch.optim.SGD([
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 2e-4},
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 2e-3},
        ])
        settings = scheduler_settings(self.options(lr_scheduler="cosine", warmup_steps=2), 6)
        scheduler = make_scheduler(optimizer, settings)
        expected = [0.5, 1.0, 0.55 + 0.45 / math.sqrt(2), 0.55,
                    0.55 - 0.45 / math.sqrt(2), 0.1]
        for scale in expected:
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2e-4 * scale)
            self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 2e-3 * scale)
            optimizer.step()
            scheduler.step()
        # Further calls stay at the floor rather than restarting cosine.
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2e-5)

    def test_no_warmup_and_constant_mode(self):
        settings = scheduler_settings(self.options(lr_scheduler="cosine", warmup_steps=0), 3)
        self.assertEqual([lr_multiplier(i, settings) for i in range(3)], [1.0, 0.55, 0.1])
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
        self.assertIsNone(make_scheduler(optimizer, scheduler_settings(self.options(), 3)))
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)

    def test_short_runs_cap_warmup_and_one_update_is_nonzero(self):
        settings = scheduler_settings(self.options(lr_scheduler="cosine"), 3)
        self.assertEqual(settings["warmup_steps"], 2)
        self.assertEqual([lr_multiplier(i, settings) for i in range(3)], [0.5, 1.0, 0.1])
        settings = scheduler_settings(self.options(lr_scheduler="cosine"), 1)
        self.assertEqual(settings["warmup_steps"], 0)
        self.assertEqual(lr_multiplier(0, settings), 1.0)

    def test_invalid_settings_are_rejected(self):
        for change in ({"lr_scheduler": "invalid"}, {"warmup_steps": -1},
                       {"warmup_steps": True}, {"warmup_steps": 0.5},
                       {"min_lr_ratio": -0.1}, {"min_lr_ratio": 1.1},
                       {"min_lr_ratio": float("nan")}, {"min_lr_ratio": float("inf")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                scheduler_settings(self.options(**change), 10)
        for steps in (0, -1, True, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                scheduler_settings(self.options(), steps)

    def test_all_training_commands_accept_schedule_options(self):
        for command in ("derive", "anchor", "train"):
            args = main.get_args([command, "--lr-scheduler", "cosine", "--warmup-steps", "5",
                                  "--min-lr-ratio", "0.2"])
            self.assertEqual((args.lr_scheduler, args.warmup_steps, args.min_lr_ratio),
                             ("cosine", 5, 0.2))

    def test_legacy_constant_masks_reuse_but_different_schedule_is_rejected(self):
        args = self.options(species=["dog"], split=1, config="config.py", data_root="data",
                            steps=10, batch_size=1, mask_lr=1e-4, beta_lr=1e-2,
                            weight_decay=0.05, eval_every=1, seed=42,
                            smoke_test=False, uniform=False)
        saved = {key: value for key, value in vars(args).items() if key not in SCHEDULER_DEFAULTS}
        _check_completed_training_settings(args, {"training_settings": saved})
        args.lr_scheduler = "cosine"
        with self.assertRaisesRegex(ValueError, "lr_scheduler"):
            _check_completed_training_settings(args, {"training_settings": saved})

    def tiny_data(self):
        model = TinyPose()
        loader = DataLoader([{"img": torch.randn(3, 2, 2), "target": torch.randn(17, 2, 2),
                              "target_weight": torch.ones(17, 1)}], batch_size=1)
        return model, loader

    def test_finetuning_uses_and_logs_scheduled_rates_and_saves_settings(self):
        model, loader = self.tiny_data()
        cfg = SimpleNamespace(model={}, data_cfg=SimpleNamespace(
            image_size=[2, 2], heatmap_size=[2, 2], num_joints=17))
        with tempfile.TemporaryDirectory(prefix="nuwa_finetune_schedule_") as temporary:
            args = self.options(lr_scheduler="cosine", warmup_steps=2, cfg=cfg,
                                out_dir=temporary, device="cpu", steps=4, learning_rate=1e-3,
                                eval_every=2, log_every=1, species="dog", split_id=1)
            optimizer_class, observed = recording_adamw()
            with patch("torch.optim.AdamW", optimizer_class), \
                    patch("nuwa.engine.train.evaluate", side_effect=[74.0, 75.0, 76.0]):
                train(args, model, loader, loader, early_stop=False, best=True)
            self.assertEqual(len(observed), 4)
            for actual, expected in zip(observed, [5e-4, 1e-3, 5.5e-4, 1e-4]):
                self.assertAlmostEqual(actual[0], expected)
            history = [json.loads(line) for line in
                       (Path(temporary) / "history.jsonl").read_text().splitlines()]
            self.assertEqual([event["learning_rate"] for event in history if event["event"] == "train"],
                             [rates[0] for rates in observed])
            best = torch.load(str(Path(temporary) / "best.pth"), map_location="cpu")
            self.assertEqual(best["finetune_step"], 4)
            self.assertEqual(best["finetune_training_settings"]["lr_schedule"]["warmup_steps"], 2)

    def test_mask_training_uses_and_logs_both_scheduled_rates(self):
        model, loader = self.tiny_data()
        with tempfile.TemporaryDirectory(prefix="nuwa_mask_schedule_") as temporary:
            root = Path(temporary)
            config, checkpoint = root / "config.py", root / "base.pth"
            config.write_text("model = {}\n", encoding="utf-8")
            torch.save(model.state_dict(), str(checkpoint))
            args = get_args_parser().parse_args([
                "--config", str(config), "--checkpoint", str(checkpoint),
                "--device", "cpu", "--smoke-test", "--log-every", "1",
                "--lr-scheduler", "cosine", "--warmup-steps", "2",
                "--out-dir", str(root / "masks")])
            optimizer_class, observed = recording_adamw()
            with patch("torch.optim.AdamW", optimizer_class):
                get_mask(args, model, loader, None)
            self.assertEqual(len(observed), 10)
            self.assertAlmostEqual(observed[0][0], args.mask_lr / 2)
            self.assertAlmostEqual(observed[0][1], args.beta_lr / 2)
            self.assertAlmostEqual(observed[1][0], args.mask_lr)
            self.assertAlmostEqual(observed[-1][0], args.mask_lr * 0.1)
            self.assertAlmostEqual(observed[-1][1], args.beta_lr * 0.1)
            history = [json.loads(line) for line in
                       (root / "masks/history.jsonl").read_text().splitlines()]
            self.assertEqual([[event["mask_lr"], event["beta_lr"]]
                              for event in history if event["event"] == "train"], observed)
            last = torch.load(str(root / "masks/last_masks.pth"), map_location="cpu")
            self.assertEqual(last["training_settings"]["lr_schedule"]["name"], "cosine")


if __name__ == "__main__":
    unittest.main()
