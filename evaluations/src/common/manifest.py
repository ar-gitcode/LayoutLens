"""Shared eval-manifest helpers (single source of truth).

Loading, iteration, split-checking, dataset-child lookup, and seq-index
resolution for the seed-pinned eval manifests. Used by the 2D runners and the
N-view orchestrator alike.
"""
from __future__ import annotations

import json
from typing import Optional


def _load_eval_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _find_room_envelopes_child(inner_ds):
    """Locate the RoomEnvelopesDataset child inside a ComposedDataset wrapper."""
    try:
        children = inner_ds.base_dataset.datasets
    except AttributeError as e:
        raise RuntimeError(
            "manifest mode requires a ComposedDataset wrapping a base dataset"
        ) from e
    for c in children:
        if type(c).__name__ == "RoomEnvelopesDataset":
            return c
    if len(children) == 1:
        return children[0]
    raise RuntimeError(
        "manifest mode could not find a RoomEnvelopesDataset child "
        f"(found: {[type(c).__name__ for c in children]})"
    )


def _manifest_iter_items(manifest: dict, requested_num_views: Optional[int] = None
                         ) -> tuple[list[dict], str, Optional[int]]:
    """Return ``(items, mode, manifest_num_views)``.

    - mode='per_view': manifest covers a single num_views (meta.num_views).
    - mode='mixed': flat samples list with per-item num_views (meta.view_counts).
    """
    if "samples" not in manifest:
        raise ValueError("manifest has no 'samples' key (expected the new schema)")
    meta = manifest.get("meta") or {}
    items = list(manifest["samples"])

    if "num_views" in meta:
        manifest_nv = int(meta["num_views"])
        if requested_num_views is not None and int(requested_num_views) != manifest_nv:
            raise ValueError(
                f"--num-views={requested_num_views} disagrees with "
                f"manifest meta.num_views={manifest_nv}"
            )
        return items, "per_view", manifest_nv

    # Mixed manifest, num_views comes from each sample.
    if "view_counts" in meta:
        return items, "mixed", None

    # Fallback: infer from samples.
    nvs = {int(it["num_views"]) for it in items if "num_views" in it}
    if len(nvs) == 1:
        return items, "per_view", next(iter(nvs))
    return items, "mixed", None


def _check_manifest_split(manifest: dict, requested_split: str,
                          allow_split_mismatch: bool) -> None:
    meta_split = (manifest.get("meta") or {}).get("split")
    if meta_split is None:
        return
    if meta_split != requested_split and not allow_split_mismatch:
        raise ValueError(
            f"manifest split={meta_split!r} != requested split={requested_split!r}. "
            f"Pass --allow-split-mismatch to override."
        )


def _resolve_seq_index(item: dict, scene_cam_lookup: dict[str, int]) -> int:
    """Prefer item['scene_cam'] → seq_index; fall back to item['seq_index']."""
    sc = item.get("scene_cam")
    if sc is not None and sc in scene_cam_lookup:
        return scene_cam_lookup[sc]
    si = item.get("seq_index")
    if si is None:
        raise ValueError(
            f"manifest item has neither resolvable scene_cam={sc!r} nor "
            f"seq_index; item keys={list(item.keys())}"
        )
    return int(si)
