# Evaluation

The evaluation suite writes JSON metrics only (no Markdown or CSV summary
tables). It is organised into 2D prediction metrics, 3D reconstruction metrics,
and an N-view orchestrator. Run all scripts from the repository root.

## 2D prediction metrics

Evaluates the per-pixel layout-depth, layout-mask, and surface-normal
predictions for one experiment.

```bash
python evaluations/src/2d/eval_2d.py run \
    --experiment e1c_uf12_layout_depth_mask_original_regression \
    --checkpoint /path/to/checkpoint.pt \
    --split test \
    --output-dir ./eval_out/2d
```

Pass `--help` for the full option list (split selection, view count, evaluation
space, scene limits, and optional W&B logging). The `--experiment` value is the
experiment identifier; see [configs.md](configs.md) for the mapping.

## 3D room-envelope reconstruction

Unprojects the predicted layout depth into a point cloud and compares it with
the ground-truth room shell.

```bash
python evaluations/src/3d/eval_room_envelope_reconstruction.py \
    --config room_envelopes/e1b_uf12_layout_depth_only_original_regression \
    --checkpoint /path/to/checkpoint.pt \
    --output_dir ./eval_out/reconstruction \
    --camera_mode gt
```

### Camera modes

- `--camera_mode gt` (default): unproject the predicted layout depth using
  ground-truth camera extrinsics and intrinsics. Alignment uses a median depth
  scale (scale-aligned track).
- `--camera_mode pred`: decode the model's own camera predictions to extrinsics
  and intrinsics and unproject through those. Alignment uses a similarity
  transform on camera centres (sim3-aligned track), and per-frame pose metrics
  are reported.

## N-view orchestrator

Runs reconstruction across 1 to 5 input views using the seed-pinned manifests.

```bash
python evaluations/eval_all_nview_manifests.py \
    --config room_envelopes/e1b_uf12_layout_depth_only_original_regression \
    --checkpoint /path/to/checkpoint.pt \
    --output-dir ./eval_out/nview
```

The N-view manifests are regenerated locally (see
[reproducibility.md](reproducibility.md)); set `ROOMENV_EVAL_CACHE_DIR` or
`--manifest-dir` to point at them.

## Classical fitting baselines

The classical plane, Manhattan, and cuboid fitting baselines are applied to the
unadapted model's points and can be tuned per scene against the ground truth.
The relevant post-processing lives in
[`training/geometry/`](../training/geometry/), and a per-scene ground-truth-tuned
baseline driver is provided at
[`evaluations/oracle_e0_traditional_postprocess.py`](../evaluations/oracle_e0_traditional_postprocess.py).
These baselines are comparison points for the learned head, not part of the
trained model.

## Evaluation settings

Reconstruction is reported under scale-aligned and scene-normalised settings.
The prediction itself is not scene-normalised; the normalisation is part of the
evaluation protocol so that predictions and ground truth are compared on a
common scale. The metric and alignment options exposed by the N-view
orchestrator (`--eval-space`, `--alignment`, `--scale-alignment`) select among
these settings. Pass `--help` to any script for the complete list.

## Outputs

Each run writes a JSON metric summary (for example, `metrics_summary.json`) under
the chosen output directory, and optionally point-cloud PLY files for visual
inspection.
