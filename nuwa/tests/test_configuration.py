"""Resolved configuration snapshots remain reloadable without a formatter dependency."""

from pathlib import Path
import importlib
import os
import pkgutil
import tempfile
import unittest
from unittest.mock import PropertyMock, patch

from ..model.utils import config_snapshot, pose_config
from ..paths import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, PROJECT_ROOT, resolve_data_root, resolve_input_file, resolve_project_path


class ConfigurationChecks(unittest.TestCase):
    def test_project_paths_are_independent_of_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="nuwa_cwd_") as temporary:
            directory = Path(temporary)
            (directory / "main.py").write_text("# unrelated file\n", encoding="utf-8")
            try:
                os.chdir(directory)
                self.assertEqual(resolve_input_file("main.py", "entry point"), PROJECT_ROOT / "main.py")
                self.assertEqual(resolve_data_root("bridge/data/ap-10k"), Path(DEFAULT_DATA_ROOT))
                self.assertEqual(resolve_project_path("nuwa/work_dirs"), DEFAULT_OUTPUT_ROOT)
                from bridge import bridge_paths
                self.assertEqual(Path(bridge_paths.DEFAULT_DATA_ROOT), Path(DEFAULT_DATA_ROOT))
                self.assertEqual(bridge_paths.resolve_input_file("main.py", "entry point"), PROJECT_ROOT / "main.py")
            finally:
                os.chdir(previous)

    def test_all_package_modules_import(self):
        import nuwa
        for module in pkgutil.walk_packages(nuwa.__path__, nuwa.__name__ + "."):
            with self.subTest(module=module.name):
                importlib.import_module(module.name)

    def test_inherited_config_round_trip_without_pretty_text(self):
        with tempfile.TemporaryDirectory(prefix="nuwa_config_") as temporary:
            directory = Path(temporary)
            (directory / "base.py").write_text(
                "model = dict(backbone=dict(depth=12, width=384))\n"
                "data_cfg = dict(image_size=[192, 256], heatmap_size=[48, 64], num_joints=17)\n",
                encoding="utf-8")
            source = directory / "source.py"
            source.write_text("_base_ = 'base.py'\nmodel = dict(backbone=dict(depth=2))\n", encoding="utf-8")
            cfg = pose_config(source)
            with patch.object(type(cfg), "pretty_text", new_callable=PropertyMock,
                              side_effect=TypeError("incompatible formatter API")):
                snapshot = config_snapshot(cfg)
            output = directory / "resolved.py"
            output.write_text(snapshot, encoding="utf-8")
            restored = pose_config(output)
            self.assertEqual(restored._cfg_dict.to_dict(), cfg._cfg_dict.to_dict())
            self.assertEqual(restored.model.backbone.depth, 2)
            self.assertEqual(restored.model.backbone.width, 384)


if __name__ == "__main__":
    unittest.main()
