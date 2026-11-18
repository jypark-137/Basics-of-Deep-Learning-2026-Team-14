# Project setup: CPU and NVIDIA GPU

These instructions target 64-bit Windows and Linux on x86-64. Run commands
from the project root, which contains `main.py`, `nuwa/`, `vitpose/`, and
`bridge/`. Commands below work in PowerShell and Bash unless marked otherwise.

NuWa runs directly from this checkout. It loads the bundled
`vitpose/mmpose` and `vitpose/mmcv_custom` source automatically; keep the local
ViTPose checkout and configs with the project. Install the third-party
dependencies described below.

## Choose an environment

If the existing environment is working, activate it and continue
with [data and checkpoint setup](#data-and-checkpoint-setup):

```console
conda activate [ENVIRONMENT_NAME]
python -c "import sys; print(sys.executable)"
```

The server environment named `dk14_sjk_torch` is described in
[WORKFLOW.md](WORKFLOW.md). Activate that environment inside your allocated GPU
session and run the verification commands before changing packages. Use a
separate environment for the fresh installation recipes below.

This checkout uses the older MMPose/MMCV APIs. The reference versions are:

| Component | Version / source |
| --- | --- |
| Python | 3.9 |
| PyTorch | 1.11.0, with the CPU or CUDA 11.3 build |
| torchvision | 0.12.0, matching the PyTorch build |
| MMCV distribution | `mmcv-full==1.4.8`, matching PyTorch and CPU/CUDA |
| MMPose | 0.24.0, from this project's `vitpose/mmpose/` |
| NumPy / SciPy | 1.23.5 / 1.10.1 |
| timm / einops | 0.4.9 / 0.6.1 |
| OpenCV / xtcocotools | 4.8.1.78 / 1.14.3 |

The bundled MMPose checks for MMCV versions from 1.3.8 through 1.5.0.
MMCV 2.x and current MMPose releases use different APIs. These recipes pin
the legacy stack rather than upgrading it.

## CPU environment

Create a new environment and install the CPU builds:

```console
conda create -n nuwa_cpu python=3.9 -y
conda activate nuwa_cpu
python -m pip install "pip==24.0" "setuptools==65.5.1" "wheel==0.43.0" "numpy==1.23.5" "opencv-python==4.8.1.78"
python -m pip install "torch==1.11.0+cpu" "torchvision==0.12.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install "mmcv-full==1.4.8" --only-binary=mmcv-full --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch1.11.0/index.html
```

These PyTorch builds follow the
[official PyTorch 1.11 installation instructions](https://pytorch.org/get-started/previous-versions/#v1110).
The [CPU MMCV wheel index](https://download.openmmlab.com/mmcv/dist/cpu/torch1.11.0/index.html)
contains Python 3.9 wheels for Windows and Linux x86-64.

Continue with [shared Python dependencies](#shared-python-dependencies).
This environment needs no NVIDIA GPU, driver, or CUDA toolkit. CUDA
availability should be `False`:

```console
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

## GPU environment

### Obtain access to the GPU

On the server described in [WORKFLOW.md](WORKFLOW.md), request the GPU session
using the site's allocation command before running Python:

```bash
ssai_agpu -g 1
```

On a workstation, use a terminal with access to the installed NVIDIA driver.
Check the GPU visible in your session:

```console
nvidia-smi
nvidia-smi -L
```

Keep the scheduler's `CUDA_VISIBLE_DEVICES` setting. Device indices in this
project refer to the GPUs visible to the Python process. A single allocated
MIG device is normally selected as `cuda:0`.

### Install the CUDA build

For a fresh environment:

```console
conda create -n nuwa_gpu python=3.9 -y
conda activate nuwa_gpu
python -m pip install "pip==24.0" "setuptools==65.5.1" "wheel==0.43.0" "numpy==1.23.5" "opencv-python==4.8.1.78"
python -m pip install "torch==1.11.0+cu113" "torchvision==0.12.0+cu113" --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install "mmcv-full==1.4.8" --only-binary=mmcv-full --find-links https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

The [PyTorch version guide](https://pytorch.org/get-started/previous-versions/#v1110)
provides this CUDA 11.3 build, and the
[matching MMCV wheel index](https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html)
provides Python 3.9 binaries for Windows and Linux x86-64. The binary-only
MMCV option avoids an unintended local C++/CUDA build.

The supplied server screenshot reports driver 535.104.05 and CUDA 12.2.
That driver can run the older CUDA 11.3 binaries through
[NVIDIA driver backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html).
`torch.version.cuda` should report `11.3` for this recipe. Installing these
prebuilt wheels does not require a separate CUDA toolkit or `nvcc`.

Continue with [shared Python dependencies](#shared-python-dependencies), then
verify CUDA explicitly:

```console
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is unavailable in this environment/session'"
python -c "import torch; p = torch.cuda.get_device_properties(0); print('GPU:', p.name); print('Visible VRAM (GiB):', round(p.total_memory / 1024**3, 2))"
```

For the provided A100 MIG allocation, plan around approximately **9.5 GiB**
and 14 SMs. The physical card's 80 GB is shared between MIG allocations.
Start with the project's default batch size of **1**. After upgrading to a
full GPU, rerun the CUDA probe and model check, then increase `--batch-size`
for mask training or fine-tuning as memory permits. The code does not select
devices by GPU model name.

## Shared Python dependencies

Run this after completing either the CPU or GPU installation above:

```console
python -m pip install "numpy==1.23.5" "scipy==1.10.1" "opencv-python==4.8.1.78" "Pillow==11.3.0" "matplotlib==3.9.4" "timm==0.4.9" "einops==0.6.1" "xtcocotools==1.14.3" "json-tricks==3.17.3" "munkres==1.1.4" "addict==2.4.0" "PyYAML==6.0.3" "yapf==0.43.0"
python -m pip check
```

NumPy is pinned for the older MMPose code and compiled evaluation package.
The timm pin preserves the layer APIs used by this ViTPose checkout.
The project uses its own config snapshot function, which works with the
listed YAPF version.

The `mmcv-full` distribution provides the Python module named `mmcv`.
Use one MMCV distribution in an environment; the upstream
[MMCV installation guide](https://mmcv.readthedocs.io/en/v1.4.8/get_started/installation.html)
explains the conflict when `mmcv` and `mmcv-full` are installed together.

The current NuWa workflow uses FP32 on CPU and GPU. Apex is optional and is
not required for these commands. TF32 is disabled during CUDA selection to
preserve the masked/pruned model equivalence checks. NumPy SVD and
least-squares calculations run on CPU; calibration features are stored on
disk instead of retaining the complete calibration set in GPU memory.

## Data and checkpoint setup

The dataset and model weights are excluded from Git. Copy the existing data
and supplied split checkpoint into this layout, or select other locations
with `--data-root` and `--checkpoint`:

```text
project-root/
  main.py
  device_utils.py
  nuwa/
  vitpose/
    mmpose/
    mmcv_custom/
    configs/
  bridge/
  checkpoints/
    vitpose-plus-small.pth         # source checkpoint, if conversion is needed
    split/
      ap10k.pth
  data/
    ap-10k/
      annotations/
        ap10k-train-split1.json
        ap10k-val-split1.json
        ap10k-test-split1.json     # needed only for test-set evaluation
      data/
        <images at the paths in the annotations' file_name fields>
  outputs/                       # generated models, masks, logs and results
```

Obtain AP-10K using the
[official dataset instructions](https://github.com/AlexTheBad/AP-10K).
Preserve its annotation IDs and train/validation/test splits. Split 1 is the
default; use the corresponding annotation files when selecting split 2 or 3.

The default checkpoint is the **ordinary, split AP-10K ViTPose-S** model:
384 embedding dimensions, 12 transformer blocks, 12 attention heads, and
17 keypoint channels. Its config is:

```text
vitpose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_small_ap10k_256x192.py
```

Use the supplied `ap10k.pth`. If obtaining another checkpoint from
[upstream ViTPose](https://github.com/ViTAE-Transformer/ViTPose), ensure it has
the same ordinary architecture and AP-10K head. A raw ViTPose++ multi-expert
checkpoint or a ViTPose-B checkpoint requires the appropriate conversion or
config before it can be loaded by this default setup.

To split the supplied ViTPose++ source checkpoint, run:

```console
python vitpose/tools/model_split.py
```

The splitter reads `checkpoints/vitpose-plus-small.pth` and writes the task
checkpoints, including `ap10k.pth`, into `checkpoints/split/`. It creates the
output directory when needed. Use `--source` and `--target` to select other
paths; relative paths are resolved from the project root.

Relative config, checkpoint, annotation, data and output paths are resolved
from the project root. Absolute paths are also accepted. Outputs default to
`outputs/`.

This output folder collects evaluation predictions/metrics, training logs,
learned masks, generated model checkpoints, calibration scores, and profiling
and comparison reports. Pretrained source/split weights stay in `checkpoints/`;
dataset files and prepared species annotations stay in `data/`.

## Verify the installation

First check the active environment and bundled source imports:

```console
python -c "import nuwa.paths; import mmcv, mmpose, torch; print('torch:', torch.__version__); print('mmcv:', mmcv.__version__); print('mmpose:', mmpose.__version__); print('source:', mmpose.__file__)"
python main.py --help
python main.py self-test
```

The MMPose source path should end in this checkout's
`vitpose/mmpose/__init__.py`. Self-tests need the dependencies and code, but
do not need the AP-10K dataset or pretrained checkpoint.

After placing the checkpoint, check model loading, finite 17-joint heatmaps,
and gradients on your chosen device:

```console
python main.py check-model --device cpu
```

In a GPU environment, also run:

```console
python main.py check-model --device cuda:0
```

The explicit CUDA selection reports an error if the GPU cannot be used.
Ordinary model commands default to `--device auto`, which selects the first
visible CUDA device or falls back to CPU. Other supported selections are
`--device cpu`, `--device cuda`, and `--device cuda:N`.

Prepare the default dog subset using the existing train/validation files:

```console
python main.py prepare --list
python main.py prepare --species dog
```

Species preparation writes filtered annotation files under
`data/ap-10k/annotations/species/dog/` and reuses the original images.

### CPU smoke run

```console
python main.py derive --species dog --pruning-rate 0.4 --device cpu --smoke-test --out-dir outputs/dog_cpu_smoke
```

### GPU smoke run

```console
python main.py derive --species dog --pruning-rate 0.4 --device cuda:0 --smoke-test --out-dir outputs/dog_gpu_smoke
```

Each smoke run performs ten mask updates, anchor export, OFP using two
training crops, checkpoint reload checks, and comparison. AP is skipped.
Add `--retrain` to include ten fine-tuning updates in the smoke check. Use a
new output directory for each fresh run.

After a successful smoke run, start a full run in another directory. This
command uses CUDA automatically when available:

```console
python main.py derive --species dog --pruning-rate 0.4 --out-dir outputs/dog_full
```

Add `--device cpu` for a full CPU run. Full runs default to 10,000 mask
updates and include validation AP and final comparison. Optional weight
fine-tuning is enabled with `--retrain --finetune-steps 1000`.

To evaluate a generated model:

```console
python main.py evaluate --checkpoint outputs/dog_full/ofp/edge.pth
```

Checkpoints can move between CPU and GPU. Start a new comparison output
directory after changing hardware; benchmark resume checks include the
device and environment. See [README.md](README.md) for individual
stages, model reuse, profiling, ablation, and resume options.

## Troubleshooting

| Symptom | Check / action |
| --- | --- |
| `torch.cuda.is_available()` is `False` | Check `nvidia-smi` inside the allocated session, the active Python environment, and that PyTorch has the `+cu113` build. Preserve the scheduler's GPU visibility setting. CPU execution remains available through `--device cpu`. |
| `No matching distribution` for MMCV | Confirm Python 3.9, 64-bit Windows/Linux x86-64, and the CPU or `cu113` wheel index matching your PyTorch build. |
| `mmcv._ext`, DLL, or undefined-symbol error | Check the PyTorch/MMCV binary pairing and whether both MMCV distributions were installed. Recreate the separate environment with one matching `mmcv-full` wheel. |
| NumPy binary/ABI error from `xtcocotools` | Check that NumPy is 1.23.5 and xtcocotools is 1.14.3 in the selected environment; reinstall their matching wheels there. |
| `libGL.so.1` or `libgthread-2.0.so.0` missing on Linux | Install the OpenCV system libraries; on Debian/Ubuntu these are `libgl1` and `libglib2.0-0`. A headless alternative is to replace `opencv-python` with `opencv-python-headless==4.8.1.78`. Keep one OpenCV distribution. |
| Dataset/subset not found | Check the `data/ap-10k/annotations/` and `data/` layout, or supply `--data-root` consistently to preparation and model commands. Run species preparation before derivation. |
| Missing/unexpected checkpoint keys or size mismatch | Check that the ordinary ViTPose-S config and split AP-10K checkpoint match. SKP mask-only files must first be exported as `anchor.pth`. |
| CUDA out of memory | Keep training batch size at 1 on the MIG allocation, check other GPU processes, and use the small config. The project also supports explicit CPU execution. |
| Output directory already exists | Choose another `--out-dir`, or use the documented `--resume` option with matching settings and completed stages. |

The fresh CPU/CUDA installation recipes use published matching wheel
indexes; they were not installed into new environments while writing this
guide. Run the checks above on the actual A100 MIG or replacement GPU before
starting a full experiment.
