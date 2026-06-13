# Training

Training is launched through [`train.py`](../train.py), which loads a Hydra
config by name from [`training/config/`](../training/config/) and delegates to
the trainer. Always run from the repository root.

## Prerequisites

- The package installed in editable mode: `pip install -e ".[train]"`.
- A CUDA-capable GPU (multi-GPU is supported through `torchrun`).
- Dataset paths exported as environment variables (see
  [dataset.md](dataset.md)).
- VGGT base / layout-initialised checkpoints available, with the weights
  directory exported (see [checkpoints.md](checkpoints.md) for the file list and
  download status):

  ```bash
  export ROOMENV_WEIGHTS_DIR=/path/to/weights
  ```

## Single-GPU

```bash
python train.py --config room_envelopes/e1b_uf12_layout_depth_only_original_regression
```

## Multi-GPU

```bash
torchrun --nproc_per_node=4 train.py \
    --config room_envelopes/e2a_uf12_layout_depth_mask_oca_original_regression
```

## Selecting an experiment

`--config` takes a config name without the `.yaml` suffix. The mapping from
readable experiment names (for example, *Layout head, last 12 blocks unfrozen*)
to config names is in [configs.md](configs.md). A convenience wrapper is provided
in [`scripts/train_example.sh`](../scripts/train_example.sh).

## What training controls

The configs set, among other things:

- which prediction heads are enabled (layout depth, mask, normal, camera),
- how many backbone blocks are unfrozen (`unfreeze_n_blocks`) and their learning
  rate,
- the loss weights for each supervision signal,
- the optimiser, learning-rate schedule, and automatic mixed precision,
- the checkpoint to initialise from (`checkpoint.resume_checkpoint_path`).

## Logging

Weights-and-Biases logging is **disabled by default** in every config
(`logging.use_wandb: false`). To enable it, set `use_wandb: true` and provide
your own `wandb_entity` and `wandb_project` in the config, then log in with
`wandb login`. TensorBoard logs are written under the run's log directory.

## Outputs

Checkpoints and logs are written to `training/logs/<experiment>/` (git-ignored).
The best checkpoint is saved as `best.pt`. Use that checkpoint path with the
evaluation scripts (see [evaluation.md](evaluation.md)).

## Tests

A small `pytest` suite lives under [`evaluations/tests/`](../evaluations/tests/):

```bash
pytest evaluations/tests
```
