# Setup

Use this guide to get from a fresh environment to a working smoke test on
64-bit Windows or Linux x86-64. Run project commands from the folder containing
`main.py`. Installation commands below work in PowerShell and Bash.

If you already have a working environment, activate it and continue with
[2. Verify the environment](#2-verify-the-environment).

## 1. Install dependencies

This checkout uses the legacy ViTPose/MMPose APIs. Use the pinned versions below
for a fresh installation, and choose **one** PyTorch/MMCV build: CPU or CUDA.

| Component | Version |
| --- | --- |
| Python | 3.9 |
| PyTorch / torchvision | 1.11.0 / 0.12.0 |
| MMCV | `mmcv-full==1.4.8` |
| MMPose | 0.24.0, bundled under `vitpose/mmpose/` |
| NumPy / SciPy | 1.23.5 / 1.10.1 |
| timm / einops | 0.4.9 / 0.6.1 |

Keep the `vitpose/` source and configs in this checkout. NuWa finds the bundled
MMPose automatically. The bundled code accepts MMCV 1.3.8 through 1.5.0;
MMCV 2.x and MMPose 1.x use different APIs.

### CPU installation

```console
conda create -n nuwa_cpu python=3.9 -y
conda activate nuwa_cpu
python -m pip install "pip==24.0" "setuptools==65.5.1" "wheel==0.43.0" "numpy==1.23.5" "opencv-python==4.8.1.78"
python -m pip install "torch==1.11.0+cpu" "torchvision==0.12.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install "mmcv-full==1.4.8" --only-binary=mmcv-full --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch1.11.0/index.html
```

Continue with [shared dependencies](#shared-dependencies).

### NVIDIA GPU installation

Check GPU access in your terminal first:

```console
nvidia-smi
```

On a shared server, obtain your GPU allocation before running the check. Keep
the scheduler's `CUDA_VISIBLE_DEVICES` setting; the first allocated GPU or MIG
device is normally `cuda:0` inside that session. Site-specific server notes are
in [WORKFLOW.md](WORKFLOW.md).

Install the CUDA 11.3 builds in a separate environment:

```console
conda create -n nuwa_gpu python=3.9 -y
conda activate nuwa_gpu
python -m pip install "pip==24.0" "setuptools==65.5.1" "wheel==0.43.0" "numpy==1.23.5" "opencv-python==4.8.1.78"
python -m pip install "torch==1.11.0+cu113" "torchvision==0.12.0+cu113" --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install "mmcv-full==1.4.8" --only-binary=mmcv-full --find-links https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

These recipes use the [official PyTorch 1.11 builds](https://pytorch.org/get-started/previous-versions/#v1110)
and matching MMCV [CPU](https://download.openmmlab.com/mmcv/dist/cpu/torch1.11.0/index.html)
or [CUDA 11.3](https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html)
wheels. The CUDA wheels need a compatible NVIDIA driver; a separate CUDA
toolkit or `nvcc` is not needed to use the prebuilt packages.

### Shared dependencies

Run these in the environment you chose above:

```console
python -m pip install "scipy==1.10.1" "Pillow==11.3.0" "matplotlib==3.9.4"
python -m pip install "timm==0.4.9" "einops==0.6.1" "xtcocotools==1.14.3" "json-tricks==3.17.3" "munkres==1.1.4"
python -m pip install "addict==2.4.0" "PyYAML==6.0.3" "yapf==0.43.0"
python -m pip check
```

The `mmcv-full` distribution supplies the `mmcv` Python module. Keep one MMCV
distribution installed; see the [MMCV installation guide](https://mmcv.readthedocs.io/en/v1.4.8/get_started/installation.html)
for binary compatibility details. NumPy and timm are pinned for the bundled
code. Apex is not required.

## 2. Verify the environment

Check versions and confirm that Python uses the bundled MMPose:

```console
python -c "import nuwa.paths, torch, mmcv, mmpose; print('torch:', torch.__version__); print('mmcv:', mmcv.__version__); print('mmpose:', mmpose.__version__); print(mmpose.__file__)"
python main.py self-test
```

The MMPose path should end in this checkout's `vitpose/mmpose/__init__.py`.
Self-test should finish with `Checks passed (...)`. It needs dependencies but
no dataset or pretrained checkpoint. Add `--verbose` for detailed output.

For a GPU environment, also confirm that PyTorch can see CUDA:

```console
python -c "import torch; print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is unavailable in this environment/session'"
```

For the GPU recipe, expect runtime `11.3` and availability `True`. A CPU
installation should report availability `False`.

## 3. Add the dataset and checkpoint

Data and pretrained weights are excluded from Git. Arrange them as follows:

```text
project-root/
  main.py
  nuwa/
  vitpose/                       # bundled code and model configs
  checkpoints/
    split_base/
      ap10k.pth                  # pretrained AP-10K ViTPose-B weights
  data/
    ap-10k/
      annotations/
        ap10k-train-split1.json
        ap10k-val-split1.json
        ap10k-test-split1.json    # only needed for test-set evaluation
      data/
        ...                      # images, matching annotation file_name paths
  outputs/                       # created by project commands
```

### Dataset

Get the images and annotations from the [official AP-10K repository](https://github.com/AlexTheBad/AP-10K#download).
Preserve the original IDs and official splits. Split 1 is the default; include
the corresponding annotation files when using split 2 or 3.

For another dataset location, pass `--data-root path/to/ap-10k` to both
`prepare` and subsequent model commands.

### Checkpoint

Use `checkpoints/split_base/ap10k.pth` with the
[default ViTPose-B AP-10K config](vitpose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_base_ap10k_256x192.py).
The model has 768 embedding dimensions, 12 transformer blocks, 12 attention
heads and 17 keypoint channels. Other checkpoints need a matching `--config`.

If you have the supplied multi-task source checkpoint instead, place it at
`checkpoints/vitpose-plus-base.pth` and export its AP-10K task:

```console
python vitpose/tools/model_split.py --source checkpoints/vitpose-plus-base.pth --target checkpoints/split_base --tasks ap10k
```

This writes `ap10k.pth` into `checkpoints/split_base/`. The `--tasks` option
selects which task checkpoints to export; omitting it exports all tasks.

Relative file paths are resolved from the project root. Absolute paths work
too. Generated models, logs, configs and reports go under `outputs/` by default;
source weights stay in `checkpoints/` and prepared annotations in `data/`.

## 4. Check the model and prepare a species

```console
python main.py check-model
python main.py prepare --species dog
```

Model commands select CUDA when available and otherwise use CPU. Add
`--device cpu` to force CPU execution, or `--device cuda:0` to require the first
visible GPU. An unavailable explicit CUDA selection produces an error.

Preparation creates dog train/validation annotations under
`data/ap-10k/annotations/species/dog/` and a config with paths already applied at
`outputs/nuwa_prepare/dog/split1/config.py`. No Python assignments need to be
copied. Use `python main.py prepare --list` to see other species.

## 5. Run a smoke test

```console
python main.py derive --species dog --pruning-rate 0.1 --smoke-test --out-dir outputs/dog_smoke
```

This performs ten mask updates, prunes using two training crops, verifies model
reloads, and runs a short comparison. AP evaluation is skipped. Choose a fresh
output folder if you repeat the test.

Training batch size defaults to 1. Start there, especially on a MIG allocation;
increase `--batch-size` for mask learning or fine-tuning only as memory permits.
Calibration and evaluation use one crop at a time.

After the smoke test succeeds, follow [Run the full experiment](README.md#3-run-the-full-experiment)
in the README. It also covers multiple species, custom paths, saved models,
comparison and resume options.

## Troubleshooting

Add `--verbose` to the failing command for diagnostic details.

| Problem | What to check |
| --- | --- |
| Python uses the wrong environment | Run `python -c "import sys; print(sys.executable)"` and activate the intended conda environment. |
| CUDA is unavailable | Check `nvidia-smi` inside your allocated session and verify the `+cu113` PyTorch build. Preserve the scheduler's GPU visibility setting. Use `--device cpu` for CPU execution. |
| No matching MMCV wheel | Use Python 3.9 on Windows/Linux x86-64 and the CPU or CUDA wheel index matching your PyTorch build. |
| `mmcv._ext`, DLL or undefined-symbol error | Check the PyTorch/MMCV pairing and that only one MMCV distribution is installed. Recreate the environment with the matching wheel if needed. |
| NumPy ABI error from `xtcocotools` | Check NumPy 1.23.5 and xtcocotools 1.14.3; reinstall matching wheels in the active environment. |
| Missing `libGL.so.1` or `libgthread-2.0.so.0` on Linux | Install `libgl1` and `libglib2.0-0` on Debian/Ubuntu, or use `opencv-python-headless==4.8.1.78` in place of `opencv-python`. Keep one OpenCV distribution. |
| Dataset or subset not found | Check the layout above, use the same `--data-root` throughout, and run `prepare` for the selected species and split. |
| Checkpoint keys or tensor sizes do not match | Pair the checkpoint with its model config. Export mask-only files through `anchor` before using them as models. |
| CUDA runs out of memory | Use batch size 1, check other GPU processes, and confirm the selected config fits the visible GPU allocation. |
| Output folder already exists | Use a new `--out-dir` for a fresh run. See [Reuse work](README.md#reuse-work) for matching completed runs and interrupted stages. |
