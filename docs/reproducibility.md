# Reproducibility notes

This is a single-seed research prototype. The results that accompany it should be
read as trends rather than statistically significant effects. This page records
the settings needed to reproduce the experimental setup.

## Environment

- Python >= 3.10.
- Pinned core dependencies: `torch==2.3.1`, `torchvision==0.18.1` (see
  [`pyproject.toml`](../pyproject.toml)). Adjust the Torch build to match your
  CUDA toolkit.
- Install with `pip install -e ".[train,dev]"`.

## Random seeds

The configs set `seed_value: 42` for training. The evaluation splits are pinned
with a separate seed encoded in the manifest filenames
(`..._seed4550.json`). Keeping these seeds fixed reproduces the same train and
evaluation splits.

## Eval manifests

The seed-pinned 1-to-5-view evaluation splits are stored as JSON manifests under
`training/cache/room_envelopes/`. These files are **not** shipped with the
repository because they are regenerable and dataset-derived. Regenerate them
after preparing the dataset:

```bash
python evaluations/src/manifests/build_room_envelopes_eval_manifest.py
```

Set `ROOMENV_EVAL_CACHE_DIR` to control where they are written and read, or pass
`--manifest-dir` to the N-view orchestrator.

## Checkpoints

The initialization weights and one trained checkpoint per experiment are being
uploaded to external hosting; see [checkpoints.md](checkpoints.md) for the file
list and status. To reproduce a result from scratch instead of downloading:

1. Prepare the dataset and export the dataset and weights environment variables
   (see [dataset.md](dataset.md)).
2. Train the chosen configuration (see [training.md](training.md)). The best
   checkpoint is saved as `training/logs/<experiment>/best.pt`.
3. Evaluate that checkpoint (see [evaluation.md](evaluation.md)).

The warm-start configurations expect a checkpoint from the corresponding
camera-supervised run; set `ROOMENV_E3A_BEST_CKPT` / `ROOMENV_E3B_BEST_CKPT`
accordingly.

## Determinism caveats

The configs use `cudnn_benchmark: true` and TF32, which favour throughput over
bitwise determinism. Exact numerical reproduction across different GPUs or driver
versions is not guaranteed; the intended reproducibility is at the level of the
reported trends.
