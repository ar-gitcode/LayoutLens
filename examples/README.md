# Examples

This directory collects short usage notes. It does not contain datasets or
checkpoints, both of which must be obtained separately (see
[../docs/dataset.md](../docs/dataset.md)).

## Minimal workflow

1. **Install** the package:

   ```bash
   pip install -e ".[train,dev]"
   ```

2. **Point at the data and weights** (see [../docs/dataset.md](../docs/dataset.md)):

   ```bash
   export ROOMENV_DATA_DIR=/path/to/datasets/room_envelopes
   export ROOMENV_WEIGHTS_DIR=/path/to/weights
   ```

3. **Train** a configuration (see [../docs/training.md](../docs/training.md)):

   ```bash
   bash scripts/train_example.sh
   ```

4. **Evaluate** the resulting checkpoint (see
   [../docs/evaluation.md](../docs/evaluation.md)):

   ```bash
   bash scripts/evaluate_example.sh /path/to/checkpoint.pt
   ```

## Running on your own images

The training and evaluation pipelines expect the Room Envelopes data layout,
including layout-depth supervision and camera metadata, so they do not run
directly on arbitrary loose images. To experiment with custom inputs you would
need to adapt the dataset loader in
[../training/data/datasets/room_envelopes.py](../training/data/datasets/room_envelopes.py)
to your own data format. This is left as an exercise rather than provided as a
turnkey script.
