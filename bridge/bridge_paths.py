"""Shared defaults for scripts run from the project root.

Layout: vitpose/ and bridge/ are siblings.
Checkpoints, data and work_dirs are inside bridge/.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(".")
VITPOSE_ROOT = Path("vitpose")
BRIDGE_ROOT = Path("bridge")

DEFAULT_SPECIES = "dog"
DEFAULT_CONFIG = (
    "vitpose/configs/animal/2d_kpt_sview_rgb_img/"
    "topdown_heatmap/ap10k/ViTPose_small_ap10k_256x192.py"
)
DEFAULT_CHECKPOINT = "bridge/checkpoints/split/ap10k.pth"
DEFAULT_DATA_ROOT = "bridge/data/ap-10k"
DEFAULT_OUTPUT_ROOT = BRIDGE_ROOT / "work_dirs"


if not (VITPOSE_ROOT / "mmpose").is_dir():
    raise RuntimeError(
        "Run commands from the project root containing vitpose/ and bridge/."
    )

if str(VITPOSE_ROOT) in sys.path:
    sys.path.remove(str(VITPOSE_ROOT))
sys.path.insert(0, str(VITPOSE_ROOT))


def resolve_input_file(value, option):
    """Check one configured file."""
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError("File not found for {}: {}".format(option, path))
    return path


if __name__ == "__main__":
    print("ViTPose:", VITPOSE_ROOT)
    print("Checkpoint:", DEFAULT_CHECKPOINT)
    print("Dataset:", DEFAULT_DATA_ROOT)
    print("Outputs:", DEFAULT_OUTPUT_ROOT)
