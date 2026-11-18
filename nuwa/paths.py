"""Shared paths for root-level data, checkpoints and experiment outputs."""


from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
VITPOSE_ROOT = PROJECT_ROOT / "vitpose"
DEFAULT_SPECIES = "dog"
DEFAULT_CONFIG = str(VITPOSE_ROOT / "configs/animal/2d_kpt_sview_rgb_img/"
                     "topdown_heatmap/ap10k/ViTPose_small_ap10k_256x192.py")
DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "checkpoints/split/ap10k.pth")
DEFAULT_DATA_ROOT = str(PROJECT_ROOT / "data/ap-10k")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"

if (VITPOSE_ROOT / "mmpose").is_dir():
    if str(VITPOSE_ROOT) not in sys.path:
        sys.path.insert(0, str(VITPOSE_ROOT))


def resolve_project_path(value):
    """Resolve user paths consistently relative to the project directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_data_root(value=DEFAULT_DATA_ROOT):
    return resolve_project_path(value)


def resolve_input_file(value, option):
    path = resolve_project_path(value)
    if not path.is_file():
        raise FileNotFoundError("File not found for {}: {}".format(option, path))
    return path
