# NuWa for ViTPose / AP-10K

This is an independent project package arranged like the original NuWa. It
uses the ordinary ViTPose backbone, weighted pose loss, and official AP-10K
evaluator. Keep the existing `vitpose/` checkout and configured `idl_proj`
environment. Dataset files live in root `data/`, pretrained weights in root
`checkpoints/`, and experiment outputs in root `outputs/`.

## Layout

```text
main.py                      common CLI; complete derivation is the default
device_utils.py              shared CPU/CUDA selection
checkpoints/                 source and split pretrained weights
data/                        AP-10K data and annotations
outputs/                   generated models, logs and reports
nuwa/
  __main__.py                python -m nuwa entry point
  paths.py                   shared paths and local ViTPose discovery
  engine/
    eval.py                  official evaluation; AP fractions and AP points
    train.py                 optional supervised fine-tuning
    utils.py                 training logs and portable best/last weights
  method/
    get_anchor_model.py      SKP masks and anchor creation
    get_clibration_data.py   training activations, calibration and allocation
    pruning.py               attention SVD and MLP reconstruction
    nuwa.py                  complete species-specific derivation
    utils.py                 mask I/O and normalization
  dataset/
    utils.py                 AP-10K dataset/loaders and batch transfer
  data/
    prepare_species.py       generate species train/validation annotations
  model/
    vit.py                   pose access, layer metadata and pruned attention
    utils.py                 model/config/device construction
    checkpoint.py            portable baseline/anchor/OFP reload and export
    check_model.py           checkpoint, heatmap and gradient diagnostic
  utils/
    utils.py                 seeding, costs, timing and neuron-score utilities
    io.py                    atomic files and fingerprints
    pose_results.py          reports, provenance and ablation definitions
  finalize_pose.py           common comparison and portable exports
  run_pose_ablation.py       controlled component ablations
  tests/                    numerical, reload, reporting and training checks
```

The root `device_utils.py` shares CPU/CUDA selection with the legacy bridge
scripts.

The spelling `get_clibration_data.py` matches upstream. Classifier/Swin
branches and class-index task files are replaced by species selection through
the existing AP-10K pipelines. The keypoint head retains all 17 channels.

## Commands

Run these from the project root using the existing environment:

```console
conda activate idl_proj
python main.py --help
python main.py self-test
python main.py check-model
python main.py prepare --list
python main.py prepare --species dog
```

Defaults are `vitpose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_small_ap10k_256x192.py`,
`checkpoints/split/ap10k.pth`, and `data/ap-10k`. Outputs default
to `outputs/`. Override paths with `--config`, `--checkpoint`,
`--data-root`, and `--out-dir`. Use `python main.py ...` or
`python -m nuwa ...`. Modules use ordinary package imports; run a specific
module with `python -m nuwa.engine.eval ...` when needed.

Relative configuration, checkpoint, annotation, data and output paths are
resolved from the project root, even when a command is started elsewhere.

`outputs/` holds artifacts from every experiment stage: evaluation predictions
and metrics; learned SKP masks; anchor, pruned and fine-tuned model weights;
training logs; calibration scores; profiling and comparison reports; and run
configs/manifests. Source pretrained weights live in `checkpoints/`, and
dataset images and species annotations live in `data/`.

## CPU and GPU execution

All model commands default to `--device auto`: use the first visible CUDA
device when PyTorch reports CUDA available, otherwise use CPU. Override this
with `--device cpu`, `--device cuda`, or `--device cuda:N` on any model command,
including `compare` and `ablate`. Child stages inherit the selected device.
The scripts in `bridge/` use the same defaults; `evaluate_cpu.py` keeps its
legacy filename and supports both devices.

```console
python main.py check-model
python main.py derive --species dog --pruning-rate 0.4
python main.py check-model --device cpu
python main.py evaluate --checkpoint PATH/edge.pth --device cuda:0
```

Your A100 MIG allocation exposes about 9.5 GiB, so batch size remains 1 by
default. Only devices exposed by the GPU allocation / `CUDA_VISIBLE_DEVICES`
are selected; the assigned MIG device is normally `cuda:0` within the process.
No GPU model or memory size is hardcoded. A larger GPU uses the same commands;
increase `--batch-size` explicitly for mask training or fine-tuning as memory
permits. Calibration and evaluation use one crop at a time.

Model forward/backward and activation collection use the selected device.
OFP's NumPy SVD and least-squares calculations use CPU, with calibration arrays
stored on disk. Saved weights remain on CPU so every checkpoint can be loaded
on either device. Execution uses FP32 without Apex; TF32 is disabled during
CUDA selection to preserve the masked/pruned heatmap equivalence checks.

Comparison reports record the selected device, visible GPU name/memory and
CUDA peak allocation. CUDA timings synchronize around each measured forward.
The CSV/JSON speedup field is `speedup` on both devices; compare runs on the same
hardware and device. Resume checks include device and hardware information.

A complete connection check performs ten mask updates, prunes using two
training crops, reloads the checkpoints, and benchmarks synthetic heatmaps:

```console
python main.py derive --species dog --pruning-rate 0.4 --smoke-test --out-dir outputs/dog_smoke
```

Run the complete experiment without `--smoke-test` in a new folder:

```console
python main.py derive --species dog --pruning-rate 0.4 --out-dir outputs/dog_full
```

The default derivation learns masks for 10,000 updates, exports the anchor,
performs OFP, and compares baseline/anchor/OFP models. Mask selection uses
species validation AP, then fewer neurons on ties. Calibration reads training
images only. Fine-tuning is optional: add `--retrain --finetune-steps 1000`.
Use `--skip-comparison` when only the generated models are needed.

Reuse completed work by selecting its masks or anchor explicitly:

```console
python main.py anchor --masks PATH/best_masks.pth --out-dir PATH/exported_anchor
python main.py derive --anchor PATH/anchor.pth --pruning-rate 0.4 --out-dir PATH/derived_model
python main.py prune --anchor PATH/anchor.pth --pruning-rate 0.4 --out-dir PATH/ofp
python main.py evaluate --checkpoint PATH/ofp/edge.pth
python main.py profile --checkpoint PATH/ofp/edge.pth --out PATH/profile.json
python main.py train --checkpoint PATH/ofp/edge.pth --steps 1000 --out-dir PATH/finetune
```

`derive --resume` reuses verified completed stages with matching settings and
inputs. It does not restore a partially completed training optimizer. An
interrupted mask/fine-tuning run preserves its checkpoints; select a saved
checkpoint explicitly to start another run. Comparison and ablation commands
retain their own fingerprint-checked resume support.

## Comparison and ablation

```console
python main.py compare --anchor PATH/anchor.pth --edge full40=PATH/edge.pth --out-dir PATH/comparison
python main.py ablate --anchor PATH/anchor.pth --rates 0.4 0.6 --out-dir PATH/ablation
```

Comparison exports independent models, verifies their heatmaps after reload,
and writes `comparison.md`, CSV/JSON, configuration snapshots and provenance.
Ablations retain the existing activation, reconstruction, without-SKP,
without-compensation and full-NuWa cases. `--smoke-test` skips AP; smoke-source
models remain marked in the reports.

## Units and supported behavior

`engine.eval.evaluate` returns AP points (0–100), matching the original NuWa
interface. `evaluate_ap`, `evaluate_metrics`, checkpoint fields and JSON metrics
use fractional AP (0–1). Baseline, anchor and OFP all use the same pose decoder,
configured flip testing and official evaluator.

`get_gflops` retains NuWa's historical one-MAC-as-one-operation convention and
returns GMACs for the complete pose model. Profiling/reports explicitly also
show matrix GFLOPs at two FLOPs per MAC. Timing covers heatmap forward only;
it excludes data loading, preprocessing, decoding and flip testing. Reduced
matrix counts do not imply a measured latency improvement.

SKP uses the original mask/gradient rule, per-block betas and AdamW settings.
OFP retains the existing uniform/random workflow, eight-wide rounding,
bias-augmented attention factorization and pseudoinverse reconstruction.
Optional neuron-scoring experiments preserve dense shapes when zeroing
columns. Full-dataset AP and training quality must be measured on real runs;
smoke runs check implementation and reload behavior.
