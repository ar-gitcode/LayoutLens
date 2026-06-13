# Checkpoints

**The checkpoints are being uploaded.** The files are large, and a suitable
hosting platform is being arranged. A download link will be added here once it
is available. Datasets are not redistributed and must be obtained separately
(see [dataset.md](dataset.md)).

This page lists the checkpoint files that accompany LayoutLens and explains where
each one is used. There are two groups: initialization weights (needed to train,
and to run the zero-shot baseline) and trained model checkpoints (one per
experiment).

## License

These checkpoints are derived from VGGT model weights and are therefore subject
to the [VGGT License](../LICENSE), the same license that covers this repository.
They were fine-tuned from the original VGGT-1B weights, which are licensed for
non-commercial use, so **these checkpoints are for non-commercial use only**.
Commercial use would require retraining from the separately licensed
VGGT-1B-Commercial weights; consult the upstream VGGT terms before redistributing
or using the checkpoints commercially.

## Initialization weights

Place these in your weights directory and point `ROOMENV_WEIGHTS_DIR` at it.

| File | Role |
| --- | --- |
| `model.pt` | VGGT pretrained base weights. The Vanilla VGGT baseline evaluates these directly, and they are the base for all adaptation. |
| `model_layout_init_from_depth.pt` | VGGT with the layout-depth head initialised from the pretrained depth head. This is the starting point (`resume_checkpoint_path`) for every trained experiment below. |

```bash
export ROOMENV_WEIGHTS_DIR=/path/to/weights   # contains model.pt, model_layout_init_from_depth.pt
```

## Trained model checkpoints

One checkpoint per experiment (the best validation checkpoint). The readable name
matches the experiment names used in the thesis; the config column is the value
passed to `--config`. Suggested filenames mirror the config stem.

### Core reconstruction

| Readable experiment name | Config | Checkpoint file |
| --- | --- | --- |
| Frozen layout head | `room_envelopes/e1_layout_depth_only_frozen` | `e1_layout_depth_only_frozen.pt` |
| Layout head, fully frozen (main loss) | `room_envelopes/e1b_layout_depth_only_frozen_original_regression` | `e1b_layout_depth_only_frozen_original_regression.pt` |
| Layout head, no blocks unfrozen | `room_envelopes/e1b_uf0_layout_depth_only_original_regression` | `e1b_uf0_layout_depth_only_original_regression.pt` |
| Layout head, last 2 blocks unfrozen | `room_envelopes/e1b_uf2_layout_depth_only_original_regression` | `e1b_uf2_layout_depth_only_original_regression.pt` |
| Layout head, last 4 blocks unfrozen | `room_envelopes/e1b_uf4_layout_depth_only_original_regression` | `e1b_uf4_layout_depth_only_original_regression.pt` |
| Layout head, last 8 blocks unfrozen | `room_envelopes/e1b_uf8_layout_depth_only_original_regression` | `e1b_uf8_layout_depth_only_original_regression.pt` |
| Layout head, last 12 blocks unfrozen (main configuration) | `room_envelopes/e1b_uf12_layout_depth_only_original_regression` | `e1b_uf12_layout_depth_only_original_regression.pt` |

### Auxiliary supervision

| Readable experiment name | Config | Checkpoint file |
| --- | --- | --- |
| Layout head + mask supervision | `room_envelopes/e1c_uf12_layout_depth_mask_original_regression` | `e1c_uf12_layout_depth_mask_original_regression.pt` |
| Layout head + normal supervision | `room_envelopes/e1d_uf12_layout_depth_normals_original_regression` | `e1d_uf12_layout_depth_normals_original_regression.pt` |
| Layout head + mask + normal supervision | `room_envelopes/e1e_uf12_layout_depth_mask_normals_original_regression` | `e1e_uf12_layout_depth_mask_normals_original_regression.pt` |

### Cross-view attention

| Readable experiment name | Config | Checkpoint file |
| --- | --- | --- |
| Mask-gated cross-view attention | `room_envelopes/e2a_uf12_layout_depth_mask_oca_original_regression` | `e2a_uf12_layout_depth_mask_oca_original_regression.pt` |
| Cross-view attention with epipolar bias | `room_envelopes/e2b_uf12_layout_depth_mask_oca_epipolar_original_regression` | `e2b_uf12_layout_depth_mask_oca_epipolar_original_regression.pt` |
| Cross-view attention + normal supervision | `room_envelopes/e2c_uf12_layout_depth_mask_normals_oca_original_regression` | `e2c_uf12_layout_depth_mask_normals_oca_original_regression.pt` |
| Cross-view attention (epipolar) + normal supervision | `room_envelopes/e2d_uf12_layout_depth_mask_normals_oca_epipolar_original_regression` | `e2d_uf12_layout_depth_mask_normals_oca_epipolar_original_regression.pt` |

### Camera supervision

| Readable experiment name | Config | Checkpoint file |
| --- | --- | --- |
| Camera-supervised model | `room_envelopes/e3a_uf12_layout_depth_mask_camera_original_regression` | `e3a_uf12_layout_depth_mask_camera_original_regression.pt` |
| Camera-supervised model with planar consistency | `room_envelopes/e3b_uf12_layout_depth_mask_camera_planar_consistency_original_regression` | `e3b_uf12_layout_depth_mask_camera_planar_consistency_original_regression.pt` |
| Camera-supervised model, warm-started cross-view attention | `room_envelopes/e3a_oca_warmstart_uf12_layout_depth_mask_camera_original_regression` | `e3a_oca_warmstart_uf12_layout_depth_mask_camera_original_regression.pt` |
| Camera-supervised model + planar consistency, warm-started cross-view attention | `room_envelopes/e3b_oca_warmstart_uf12_layout_depth_mask_camera_planar_consistency_original_regression` | `e3b_oca_warmstart_uf12_layout_depth_mask_camera_planar_consistency_original_regression.pt` |

## Using a checkpoint

For evaluation, pass the file directly:

```bash
python evaluations/src/3d/eval_room_envelope_reconstruction.py \
    --config room_envelopes/e1b_uf12_layout_depth_only_original_regression \
    --checkpoint /path/to/weights/e1b_uf12_layout_depth_only_original_regression.pt \
    --output_dir ./eval_out/reconstruction --camera_mode gt
```

The two warm-started cross-view attention configs initialise from the
camera-supervised checkpoints. Point the environment variables at them before
training those configs:

```bash
export ROOMENV_E3A_BEST_CKPT=/path/to/weights/e3a_uf12_layout_depth_mask_camera_original_regression.pt
export ROOMENV_E3B_BEST_CKPT=/path/to/weights/e3b_uf12_layout_depth_mask_camera_planar_consistency_original_regression.pt
```

## Suggested directory layout

```text
weights/
├── model.pt
├── model_layout_init_from_depth.pt
├── e1_layout_depth_only_frozen.pt
├── e1b_uf12_layout_depth_only_original_regression.pt
├── ...
└── e3b_oca_warmstart_uf12_layout_depth_mask_camera_planar_consistency_original_regression.pt
```
