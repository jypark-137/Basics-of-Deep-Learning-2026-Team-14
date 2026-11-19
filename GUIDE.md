# Using this project

This project prunes a ViTPose model for one or more animal species. The
experiments below create **one cat-and-dog model and one separate horse model**.

## Get ready

Follow [SETUP.md](SETUP.md) to install the dependencies and arrange the data
and pretrained checkpoint. Activate your environment, then open a terminal
in the project root—the folder containing `main.py`.

The default locations are:

```text
checkpoints/split_base/ap10k.pth  pretrained ViTPose-B model
data/ap-10k/                     dataset and annotations
outputs/                        generated models, logs and results
```

Check that the model loads:

```console
python main.py check-model
```

GPU is used automatically when available; otherwise the code uses CPU.
Add `--device cpu` to a model command to force CPU execution.

## Choose your species

List the available species, then prepare your selection:

```console
python main.py prepare --list
python main.py prepare --species cat dog
python main.py prepare --species horse
```

Use `--species dog` for one species, or `--species all` for every species.
Quote names containing spaces, for example `--species cat "brown bear"`.
Preparation creates filtered annotations and reuses the original images.
It also saves a config with the dataset paths already filled in under
`outputs/nuwa_prepare/<species>/split1/config.py`. Continue with the commands
below; no Python code needs to be copied or edited.

## Try a quick run

```console
python main.py derive --species cat dog --pruning-rate 0.1 --mask-lr 1e-4 --beta-lr 1e-2 --learning-rate 1e-5 --lr-scheduler cosine --smoke-test --out-dir outputs/cat_dog_base_alpha01_lr10_smoke
```

This checks the pipeline with ten mask updates and two calibration crops.
It skips accuracy evaluation. Use it to confirm everything runs before
starting a full experiment.

## Run the full pipeline

Use these explicit learning-rate overrides for the 10x-reduced experiments.
The stored defaults use the original learning rates shown in the table.

| Learning rate | Original | These experiments (10x lower) |
| --- | ---: | ---: |
| Mask | `1e-3` | `1e-4` |
| Beta | `0.1` | `0.01` |
| Fine-tuning, when enabled | `1e-4` | `1e-5` |

Cats and dogs together:

```console
python main.py derive --species cat dog --device cuda:0 --pruning-rate 0.1 --mask-lr 1e-4 --beta-lr 1e-2 --learning-rate 1e-5 --lr-scheduler cosine --out-dir outputs/cat_dog_base_alpha01_lr10
```

Horses:

```console
python main.py derive --species horse --device cuda:0 --pruning-rate 0.1 --mask-lr 1e-4 --beta-lr 1e-2 --learning-rate 1e-5 --lr-scheduler cosine --out-dir outputs/horse_base_alpha01_lr10
```

Each run learns masks, prunes the Base model, and compares the original and
generated models. Both use split 1, seed 42, batch size 1, 10,000 mask updates,
128 calibration crops, and validation every 1,000 updates. Alpha `0.1` targets
a 10% reduction in matrix computation (MACs); the report shows the achieved
reduction and measured performance.

Cosine scheduling uses 100 warmup updates and decays each rate to 10% of its
peak by the end of its training stage. Final rates are `1e-5` for masks,
`1e-3` for beta controls, and `1e-6` for optional fine-tuning.

Use a **new output folder for each fresh run**. To fine-tune the weights after
pruning, add `--retrain --finetune-steps 1000` to the command.

## Find and evaluate your model

Results are saved separately in `outputs/cat_dog_base_alpha01_lr10/` and
`outputs/horse_base_alpha01_lr10/`. Both folders use this layout:

| File | What it contains |
| --- | --- |
| `anchor/anchor.pth` | Model produced by mask learning |
| `ofp/edge.pth` | Final pruned model |
| `comparison/comparison.md` | Accuracy, model size, computation and timing comparison |
| `finetune/best.pth` | Best fine-tuned model, when `--retrain` is enabled |

Evaluate the pruned model on the selected species' validation data:

```console
python main.py evaluate --checkpoint outputs/cat_dog_base_alpha01_lr10/ofp/edge.pth
```

```console
python main.py evaluate --checkpoint outputs/horse_base_alpha01_lr10/ofp/edge.pth
```

The species list is read from the checkpoint, so you do not need to repeat
`--species`. Evaluation prints AP and saves predictions and metrics under
`outputs/nuwa_eval/`.

## Get more help

```console
python main.py --help
python main.py derive --help
```

See [README.md](README.md) for individual stages, checkpoint reuse, profiling
and ablation commands. See [WORKFLOW.md](WORKFLOW.md) for the research plan.
Add `--verbose` to any command when you need detailed diagnostic output.
