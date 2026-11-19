# NuWa for ViTPose / AP-10K

Prune a ViTPose model for one or more animal species, then compare its accuracy,
size and inference speed with the original model. The model keeps all 17 AP-10K
keypoints and runs on CPU or NVIDIA GPU.

The default model is ViTPose-B (Base), using the AP-10K task checkpoint at
`checkpoints/split_base/ap10k.pth`.

**First time here?** Follow [SETUP.md](SETUP.md) to install dependencies and add
the dataset and checkpoint. For the research plan, see [WORKFLOW.md](WORKFLOW.md).

## Quick start

Activate your Python environment and run these commands from the project root
(the folder containing `main.py`).

### 1. Check the model and prepare your species

```console
python main.py check-model
python main.py prepare --species dog
```

Preparation filters the annotations, reuses the original images, and saves a
config with the dataset paths filled in. No Python code needs to be edited.

### 2. Try a smoke test

```console
python main.py derive --species dog --pruning-rate 0.1 --smoke-test --out-dir outputs/dog_smoke
```

This runs ten mask updates, prunes using two training crops, verifies checkpoint
reloads, and benchmarks the models. Accuracy evaluation is skipped.

Smoke checkpoints test execution and conversion, not accuracy. Their AP can be
near zero: with the default config, two crops provide only 384 token rows to
reconstruct each pruned MLP, which can have more retained neurons than calibration
rows. A tiny reconstruction error on those crops does not establish validation
accuracy. Use the full experiment below for accuracy comparisons.

### 3. Run the full experiment

```console
python main.py derive --species dog --pruning-rate 0.1 --out-dir outputs/dog_full
```

The full pipeline learns masks for 10,000 updates, exports an **anchor** (the
model after mask learning), prunes it, and compares the original, anchor and
pruned models using validation AP and measured latency.

`--pruning-rate 0.1` targets a 10% reduction in matrix operations. The report
shows the achieved reduction and measured speedup. Choose a new output folder
for each fresh experiment; see [reusing work](#reuse-work) for completed runs.

## Find your results

For the full-run example above:

| File | What it contains |
| --- | --- |
| `outputs/dog_full/ofp/edge.pth` | Final pruned model |
| `outputs/dog_full/comparison/comparison.md` | Accuracy, size and speed comparison |
| `outputs/dog_full/anchor/anchor.pth` | Model after mask learning |
| `outputs/dog_full/anchor/best_masks.pth` | Learned masks; used to export an anchor |
| `outputs/dog_full/finetune/best.pth` | Fine-tuned model, when `--retrain` is enabled |

Comparison results are also saved as CSV and JSON. Smoke runs save
`last_masks.pth` and, if fine-tuning is enabled, `last.pth` instead of the best
checkpoints. Smoke outputs are marked in the reports.

Evaluate or profile the pruned model separately:

```console
python main.py evaluate --checkpoint outputs/dog_full/ofp/edge.pth
python main.py profile --checkpoint outputs/dog_full/ofp/edge.pth --out outputs/dog_full/profile.json
```

Evaluation uses the species and split recorded in the checkpoint. Profiling
prints a short summary; `--out` saves the complete JSON report.

## Choose species and data

List available species, or prepare one model for cats and dogs together:

```console
python main.py prepare --list
python main.py prepare --species cat dog
python main.py derive --species cat dog --pruning-rate 0.1 --out-dir outputs/cat_dog
```

Names are case-insensitive and order-independent. Quote names containing spaces,
such as `--species cat "brown bear"`. Use `--species all` for the complete dataset.
Later model commands inherit the species from their input checkpoint when the
option is omitted; an explicit selection must match that checkpoint.

Preparation keeps the official train/validation split and original IDs. For
cats and dogs, it writes annotations under
`data/ap-10k/annotations/species/cat__dog/` and a config at
`outputs/nuwa_prepare/cat__dog/split1/config.py`. Model commands apply the dataset
paths automatically. To choose the config source and destination, use
`python main.py prepare --species cat dog --config path/to/source.py --out-config outputs/cat_dog_config.py`.

Split 1 is the default. To use another official split, select it in both commands:

```console
python main.py prepare --species dog --split 2
python main.py derive --species dog --split-id 2 --pruning-rate 0.1 --out-dir outputs/dog_split2
```

### Default locations

| Input or output | Default location |
| --- | --- |
| AP-10K dataset | `data/ap-10k/` |
| Pretrained checkpoint | `checkpoints/split_base/ap10k.pth` |
| Model config | [ViTPose-B AP-10K config](vitpose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_base_ap10k_256x192.py) |
| Generated files | `outputs/` |

Use `--data-root`, `--checkpoint`, `--config` and `--out-dir` to override these
locations. Pass the same data root to preparation and model commands. Relative
paths are resolved from the project root; absolute paths are also accepted.
When evaluating older ViTPose-S checkpoints, pass the matching
[Small config](vitpose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_small_ap10k_256x192.py)
with `--config`.

## Common options

| Option | Use it to |
| --- | --- |
| `--device cpu` | Force CPU execution |
| `--device cuda:0` | Use the first GPU visible to the process |
| `--batch-size 2` | Increase the mask-training or fine-tuning batch size if memory permits |
| `--threads 2` | Set the number of PyTorch CPU threads |
| `--retrain --finetune-steps 1000` | Fine-tune model weights after pruning |
| `--lr-scheduler cosine` | Enable warmup and cosine decay for mask learning and fine-tuning |
| `--warmup-steps 100 --min-lr-ratio 0.1` | Warm up over 100 updates, then decay to 10% of each peak LR |
| `--skip-comparison` | Stop after generating models |
| `--verbose` | Show per-layer details, verification checks and library output |

Model commands default to `--device auto`: CUDA when available, otherwise CPU.
An explicit CUDA selection reports an error if that device is unavailable.
Training batch size defaults to 1. Evaluation and calibration use one crop at a
time. On a shared GPU or MIG allocation, device numbers refer to the devices
visible inside your session.

Default peak learning rates are the original values: `--mask-lr 1e-3`,
`--beta-lr 0.1`, and `--learning-rate 1e-4` for fine-tuning. The default pruning target is 10%
(`--pruning-rate 0.1`). Learning rates
are constant by default (`--lr-scheduler none`). To use warmup and cosine
decay for a Base experiment with fine-tuning, run:

```console
python main.py derive --species cat dog --device cuda:0 --pruning-rate 0.1 --lr-scheduler cosine --retrain --out-dir outputs/cat_dog_base_alpha01_original_lr_finetuned
```

The specified learning rates are the peaks. Each training stage warms up from
peak / warmup-updates to its peak, then decays to the selected fraction on its
last update. Warmup is capped to one less than the stage's update count for
short or smoke runs. Without warmup, cosine starts at the peak. Each parameter
group keeps its own peak and floor. `anchor` and `train` accept the same schedule
options. Actual update rates are printed and recorded in `history.jsonl`;
effective schedule settings are saved in run configs and training checkpoints.
Use a fresh output folder when changing the schedule. Its effect on AP must be
measured in a new experiment.

## Reuse work

**Reuse an anchor** to skip mask learning in a new experiment:

```console
python main.py derive --anchor outputs/dog_full/anchor/anchor.pth --pruning-rate 0.6 --out-dir outputs/dog_60
```

**Export saved masks** as a standalone anchor:

```console
python main.py anchor --masks outputs/dog_full/anchor/best_masks.pth --out-dir outputs/dog_anchor
```

Repeating the same `anchor` command verifies and reuses a matching completed
anchor without rewriting it. Empty output folders and folders containing only
a configuration from a failed initialization can also be retried.

**Resume a pipeline** with the original command and `--resume`:

```console
python main.py derive --species dog --pruning-rate 0.4 --out-dir outputs/dog_full --resume
```

Resume reuses verified completed stages with matching inputs, settings, code and device.
It does not restore a partially completed training optimizer. If mask learning
was interrupted, select its saved masks with `--masks` in a new run. For
interrupted fine-tuning, use the saved checkpoint as input to a new `train` run.
Comparison and ablation commands also support `--resume`.

## Other commands

| Command | Purpose |
| --- | --- |
| `anchor` | Learn masks and export an anchor, or export existing masks |
| `prune` | Prune an existing anchor |
| `train` | Fine-tune an existing model |
| `compare` | Compare existing original, anchor and pruned models |
| `ablate` | Run controlled pruning experiments |
| `check-model` | Check checkpoint loading, heatmaps and gradients |
| `self-test` | Run project checks without data or pretrained weights |

For example, compare existing models or run the ablation study:

```console
python main.py compare --anchor outputs/dog_full/anchor/anchor.pth --edge nuwa=outputs/dog_full/ofp/edge.pth --out-dir outputs/comparison
python main.py ablate --anchor outputs/dog_full/anchor/anchor.pth --rates 0.4 0.6 --out-dir outputs/ablation
```

Use `python main.py --help` to list commands and `python main.py derive --help`
for command options. `python -m nuwa ...` is an equivalent entry point. See
[SETUP.md#troubleshooting](SETUP.md#troubleshooting) if a command fails.

<details>
<summary>Measurement and implementation notes</summary>

- Console AP and comparison tables use points from 0 to 100. JSON metrics and
  saved validation AP values use fractions from 0 to 1. The Python helper
  `engine.eval.evaluate` returns AP points; `evaluate_ap` and `evaluate_metrics`
  use fractions.
- MACs count matrix operations for the complete pose model. Matrix GFLOPs use
  two FLOPs per MAC. The legacy `get_gflops` helper returns GMACs.
- Timing measures heatmap forward only. It excludes image loading,
  preprocessing, decoding and flip testing. Compare latency on the same device
  and hardware; fewer matrix operations do not guarantee lower latency.
- Model computation uses FP32. CUDA selection disables TF32 for heatmap
  equivalence checks. NumPy SVD and least-squares calculations run on CPU, with
  calibration features stored on disk. Checkpoints can load on CPU or GPU.
- Mask learning (SKP) uses the original top-k/gradient rule, per-block betas and
  AdamW. Pruning (OFP) uses attention factorization, eight-wide MLP rounding and
  pseudoinverse reconstruction. Mask selection uses validation AP, then fewer
  neurons on ties; calibration reads training images only.
- Smoke runs check execution and checkpoint conversion. Use full runs to assess
  accuracy and training quality.

</details>
