# Dataset preparation

LayoutLens does not redistribute any dataset. This page explains how to obtain
and lay out the data so the code can find it.

## Room Envelopes

The experiments use the **Room Envelopes** dataset, a synthetic dataset for
indoor layout reconstruction from images
([Bahrami and Campbell, 2025](https://arxiv.org/abs/2511.03970)). Obtain it from
its original source under its own license.

Each sample provides:

- an RGB view of an indoor scene,
- a **layout-depth** map giving the depth of the first structural surface along
  each ray, including structure occluded by furniture,
- an optional **visibility / layout mask**,
- an optional **surface-normal** map,
- per-frame camera intrinsics, and recovered extrinsics for multi-view samples.

### Pointing the code at the data

All external locations are resolved through environment variables. Set the ones
that apply before training or evaluating:

```bash
export ROOMENV_DATA_DIR=/path/to/datasets/room_envelopes
# Extracted WebDataset shards (defaults to $ROOMENV_DATA_DIR/data_wds_extracted)
export ROOMENV_DATA_WDS_DIR=$ROOMENV_DATA_DIR/data_wds_extracted
# Camera extrinsics manifest (defaults to $ROOMENV_DATA_DIR/extrinsics_manifest.npz)
export ROOMENV_EXTRINSICS_MANIFEST=$ROOMENV_DATA_DIR/extrinsics_manifest.npz
```

The same variables are read by the Hydra configs (via OmegaConf
`${oc.env:...}` interpolation) and by the Python entry points (via
[`room_envelopes/paths.py`](../room_envelopes/paths.py)), so setting them once
relocates everything.

### Depth decoding convention

Depth and layout-depth PNGs are MoGe log-encoded `uint16` images, with the
`near` and `far` range stored in PNG `tEXt` chunks. The decode is:

```text
t     = (raw - 1) / 65533
depth = near**(1 - t) * far**t        # metric z-depth, in metres
```

with sentinel raw values `0` (unknown / invalid) and `65535` (sky / beyond far).
The only correct decoder is [`room_envelopes/io.py`](../room_envelopes/io.py);
do not approximate it with `raw / 1000`. The `meters_per_asset_unit` value from
the extrinsics manifest applies only to camera-pose translations, never to depth.

## What is not included

- Raw or processed dataset files.
- Camera extrinsics manifests (`*.npz`).
- Sequence caches (`*.pkl`) and seed-pinned eval manifests (`*.json`); these are
  regenerated locally, see [reproducibility.md](reproducibility.md).
