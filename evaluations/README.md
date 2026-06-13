# Evaluation suite

Evaluation code for LayoutLens. All scripts write JSON metrics only and are run
from the repository root. For the full guide, including metric definitions and
camera/alignment options, see [../docs/evaluation.md](../docs/evaluation.md).

## Layout

```text
evaluations/
├── eval_all_nview_manifests.py        # N-view (1-5) reconstruction orchestrator
├── oracle_e0_traditional_postprocess.py  # per-scene ground-truth-tuned classical baseline
├── src/
│   ├── common/                        # shared I/O, alignment, scaling, manifests
│   ├── 2d/                            # layout depth / mask / normal metrics
│   ├── 3d/                            # reconstruction + Chamfer metrics, pose
│   └── manifests/                     # build the seed-pinned eval splits
└── tests/                             # pytest suite
```

## Common entry points

```bash
# 2D prediction metrics for one experiment
python evaluations/src/2d/eval_2d.py run --experiment <name> --checkpoint <ckpt.pt>

# 3D room-envelope reconstruction
python evaluations/src/3d/eval_room_envelope_reconstruction.py \
    --config <config> --checkpoint <ckpt.pt> --output_dir ./eval_out/recon --camera_mode gt

# N-view orchestrator
python evaluations/eval_all_nview_manifests.py \
    --config <config> --checkpoint <ckpt.pt> --output-dir ./eval_out/nview
```

Pass `--help` to any script for its full option list.
