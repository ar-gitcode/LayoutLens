# Eval-manifest cache

This directory holds seed-pinned evaluation manifests (1-to-5-view JSON splits)
used to make evaluation reproducible.

The manifest files are **not** committed to the repository. They are regenerable
and dataset-derived, so they are produced locally after the dataset is prepared:

```bash
python evaluations/src/manifests/build_room_envelopes_eval_manifest.py
```

Set `ROOMENV_EVAL_CACHE_DIR` to control where they are written and read. See
[../../../docs/reproducibility.md](../../../docs/reproducibility.md) for details.
