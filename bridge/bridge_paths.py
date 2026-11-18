"""Use the project's shared paths while keeping legacy bridge output folders."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nuwa.paths import (
    BRIDGE_ROOT, DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATA_ROOT,
    DEFAULT_SPECIES, VITPOSE_ROOT, resolve_data_root, resolve_input_file,
    resolve_project_path,
)
from device_utils import add_device_argument, resolve_device

DEFAULT_OUTPUT_ROOT = BRIDGE_ROOT / "work_dirs"

__all__ = [
    "PROJECT_ROOT", "VITPOSE_ROOT", "BRIDGE_ROOT", "DEFAULT_SPECIES",
    "DEFAULT_CONFIG", "DEFAULT_CHECKPOINT", "DEFAULT_DATA_ROOT", "DEFAULT_OUTPUT_ROOT",
    "resolve_project_path", "resolve_data_root", "resolve_input_file",
    "add_device_argument", "resolve_device",
]


if __name__ == "__main__":
    print("ViTPose:", VITPOSE_ROOT)
    print("Checkpoint:", DEFAULT_CHECKPOINT)
    print("Dataset:", DEFAULT_DATA_ROOT)
    print("Outputs:", DEFAULT_OUTPUT_ROOT)
