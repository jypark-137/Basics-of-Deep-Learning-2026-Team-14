# Using this project

This project prunes a ViTPose model for one or more animal species. The
example below creates **one model for cats and dogs together**.

## Get ready

Follow [SETUP.md](SETUP.md) to install the dependencies and arrange the data
and pretrained checkpoint. Activate your environment, then open a terminal
in the project root—the folder containing `main.py`.

The default locations are:

```text
checkpoints/split/ap10k.pth       pretrained model
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
```

Use `--species dog` for one species, or `--species all` for every species.
Quote names containing spaces, for example `--species cat "brown bear"`.
Preparation creates filtered annotations and reuses the original images.

## Try a quick run

```console
python main.py derive --species cat dog --pruning-rate 0.4 --smoke-test --out-dir outputs/cat_dog_smoke
```

This checks the pipeline with ten mask updates and two calibration crops.
It skips accuracy evaluation. Use it to confirm everything runs before
starting a full experiment.

## Run the full pipeline

```console
python main.py derive --species cat dog --pruning-rate 0.4 --out-dir outputs/cat_dog
```

This learns masks, prunes the model, and compares the original and generated
models. Full runs use 10,000 mask updates by default. `0.4` targets a 40%
reduction in matrix computation (MACs); the report
shows the achieved reduction and measured performance.

Use a **new output folder for each fresh run**. To fine-tune the weights after
pruning, add `--retrain --finetune-steps 1000` to the command.

## Find and evaluate your model

After the full run, look inside `outputs/cat_dog/`:

| File | What it contains |
| --- | --- |
| `anchor/anchor.pth` | Model produced by mask learning |
| `ofp/edge.pth` | Final pruned model |
| `comparison/comparison.md` | Accuracy, model size, computation and timing comparison |
| `finetune/best.pth` | Best fine-tuned model, when `--retrain` is enabled |

Evaluate the pruned model on the selected species' validation data:

```console
python main.py evaluate --checkpoint outputs/cat_dog/ofp/edge.pth
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
