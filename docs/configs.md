# Configuration files

Training and evaluation are driven by [Hydra](https://hydra.cc/) YAML configs
under [`training/config/`](../training/config/). A config is selected by name
(without the `.yaml` suffix), for example:

```bash
python train.py --config room_envelopes/e1b_uf12_layout_depth_only_original_regression
```

The configuration files keep the short identifiers used during development. This
page maps each one to the readable experiment name used in the thesis, so the
identifiers do not need to be memorised.

## How to read a config name

The identifiers are built from a few repeated tokens:

| Token | Meaning |
| --- | --- |
| `layout_depth` | The layout-depth head is trained. |
| `mask` | Layout-mask supervision is added. |
| `normals` | Surface-normal supervision is added. |
| `camera` | Camera supervision is added (real extrinsics enforced). |
| `planar_consistency` | A planar-consistency prior is added to the layout depth. |
| `oca` | The mask-gated cross-view attention module is added on the layout-depth path. |
| `epipolar` | The cross-view attention uses an epipolar attention bias. |
| `uf<N>` | The last `N` backbone blocks are unfrozen (`uf0` = fully frozen). |
| `frozen` | The backbone is fully frozen. |
| `original_regression` | The main layout-depth regression loss formulation. |
| `warmstart` | Initialised from a previous run's best checkpoint. |
| `smoke` | A short sanity-check run, not a full experiment. |

## Main experiments (Room Envelopes dataset)

Files live in [`training/config/room_envelopes/`](../training/config/room_envelopes/).

| Readable experiment name | Config name to pass to `--config` |
| --- | --- |
| Vanilla VGGT (zero-shot baseline, evaluation only) | `room_envelopes/e0_vanilla_eval_only` |
| Frozen layout head | `room_envelopes/e1_layout_depth_only_frozen` |
| Layout head, fully frozen (main loss) | `room_envelopes/e1b_layout_depth_only_frozen_original_regression` |
| Layout head, no blocks unfrozen (unfreeze sweep, N=0) | `room_envelopes/e1b_uf0_layout_depth_only_original_regression` |
| Layout head, last 2 blocks unfrozen | `room_envelopes/e1b_uf2_layout_depth_only_original_regression` |
| Layout head, last 4 blocks unfrozen | `room_envelopes/e1b_uf4_layout_depth_only_original_regression` |
| Layout head, last 8 blocks unfrozen | `room_envelopes/e1b_uf8_layout_depth_only_original_regression` |
| Layout head, last 12 blocks unfrozen (main configuration) | `room_envelopes/e1b_uf12_layout_depth_only_original_regression` |
| Layout head + mask supervision | `room_envelopes/e1c_uf12_layout_depth_mask_original_regression` |
| Layout head + normal supervision | `room_envelopes/e1d_uf12_layout_depth_normals_original_regression` |
| Layout head + mask + normal supervision | `room_envelopes/e1e_uf12_layout_depth_mask_normals_original_regression` |
| Mask-gated cross-view attention | `room_envelopes/e2a_uf12_layout_depth_mask_oca_original_regression` |
| Cross-view attention with epipolar bias | `room_envelopes/e2b_uf12_layout_depth_mask_oca_epipolar_original_regression` |
| Cross-view attention + normal supervision | `room_envelopes/e2c_uf12_layout_depth_mask_normals_oca_original_regression` |
| Cross-view attention (epipolar) + normal supervision | `room_envelopes/e2d_uf12_layout_depth_mask_normals_oca_epipolar_original_regression` |
| Camera-supervised model | `room_envelopes/e3a_uf12_layout_depth_mask_camera_original_regression` |
| Camera-supervised model with planar consistency | `room_envelopes/e3b_uf12_layout_depth_mask_camera_planar_consistency_original_regression` |

The "no blocks unfrozen" sweep entry (`e1b_uf0_...`) and the "fully frozen"
config both freeze the backbone; the former reaches that state through the
unfreeze-sweep mechanism (`unfreeze_n_blocks: 0`) and is the natural N=0 point of
the unfreeze ladder, while the latter is the standalone frozen configuration.

## Warm-started cross-view attention

These configs add a cross-view attention block to the camera-supervised models
and initialise it from a previous run rather than training from scratch.

| Description | Config name |
| --- | --- |
| Camera-supervised model, warm-started cross-view attention | `room_envelopes/e3a_oca_warmstart_uf12_layout_depth_mask_camera_original_regression` |
| Camera-supervised model + planar consistency, warm-started cross-view attention | `room_envelopes/e3b_oca_warmstart_uf12_layout_depth_mask_camera_planar_consistency_original_regression` |

The warm-start configs initialise from the best checkpoint of the corresponding
camera-supervised run. Set `ROOMENV_E3A_BEST_CKPT` / `ROOMENV_E3B_BEST_CKPT` to
point at your own checkpoint, or train the base camera-supervised model first.

## Base and shared configs

| File | Role |
| --- | --- |
| `default.yaml` | Base defaults shared by all runs. |
| `default_dataset.yaml` | Base dataset defaults. |
| `room_envelopes/0_default_dataset_room_envelopes.yaml` | Room Envelopes dataset defaults, included by every Room Envelopes config via the Hydra `defaults` list. |

## Smoke configs

Configs ending in `_smoke` (and `room_envelopes/e0_multiview_real_extrinsics_smoke`)
are short sanity checks of the data loader and training loop. They are not full
experiments and are not evaluated.
