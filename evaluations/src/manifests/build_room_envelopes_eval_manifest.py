#!/usr/bin/env python3
"""Build deterministic Room Envelopes evaluation manifests, config-free.

Scans the Room Envelopes extracted dataset directly (no Hydra, no
RoomEnvelopesDataset instantiation) and writes one JSON per (split,
num_views) pair plus an optional validation-only mixed-overlap manifest.

Per-view manifest (one file per num_views):

    eval_manifest_<split>_<n>view_seed<base_seed>.json

Strategy ``seeded_non_overlapping_frame_grouping``: for each sequence,
local frame ids ``0..n_frames-1`` are (optionally) shuffled with a stable
hashlib.md5-derived seed and partitioned into non-overlapping groups of
size ``num_views``. Leftover frames are saved separately and are NOT
evaluated.

Mixed validation manifest (optional, val only):

    eval_manifest_val_mixed_1to5_overlap_seed<base_seed>.json

Strategy ``seeded_overlapping_mixed_view_grouping``: a flat list of
samples mixing 1- through 5-view groups. Overlap across samples is
permitted; within a single sample ids are unique. For validation
robustness / model selection only.

Example::

    python evaluations/src/manifests/build_room_envelopes_eval_manifest.py \
      --data-dir "$ROOMENV_DATA_WDS_DIR" \
      --splits val test \
      --output-dir training/cache/room_envelopes \
      --view-counts 1 2 3 4 5 \
      --base-seed 4550 \
      --shuffle true \
      --min-frames 1 \
      --build-mixed-val \
      --mixed-samples-per-sequence-per-view 2
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np

# Make the in-repo packages importable when this script is run directly
# (flat sys.path bootstrap; see common/_paths.py). No chdir, this builder
# resolves paths against the user's cwd.
_d = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_d) != "src":
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "common"))
import _paths  # noqa: E402: adds repo root, training, all eval subdirs to sys.path
from room_envelopes.paths import DATA_WDS_DIR  # noqa: E402

DEFAULT_DATA_DIR = DATA_WDS_DIR
DEFAULT_BASE_SEED = 4550
DEFAULT_VIEW_COUNTS = [1, 2, 3, 4, 5]
DEFAULT_OUTPUT_DIR = "training/cache/room_envelopes"
DEFAULT_SPLITS = ["val", "test"]
DEFAULT_MIN_FRAMES = 1


# ---------------------------------------------------------------------------
# Stable seeding (hashlib.md5, reproducible across runs/machines)
# ---------------------------------------------------------------------------

def _stable_seed(*parts: Any) -> int:
    joined = "|".join(str(p) for p in parts)
    digest = hashlib.md5(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


# ---------------------------------------------------------------------------
# Direct dataset scan, mirrors RoomEnvelopesDataset._build_sequences()
# ---------------------------------------------------------------------------

def scan_sequences(data_dir: str, split: str, min_frames: int) -> list[dict]:
    """Return ``[{scene_cam, scene, cam, frames}]`` sorted by scene_cam.

    Mirrors the layout/sort order in
    ``training/data/datasets/room_envelopes.py::_build_sequences`` so the
    resulting ``seq_index`` matches a dataset built with the same
    ``min_frames`` and no extra filtering.
    """
    shard_pattern = os.path.join(data_dir, f"{split}-*")
    shard_dirs = sorted(glob.glob(shard_pattern))
    if not shard_dirs:
        raise RuntimeError(
            f"no shards found for split={split!r} under {data_dir!r} "
            f"(pattern: {shard_pattern})"
        )

    groups: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for shard in shard_dirs:
        for jp in sorted(glob.glob(os.path.join(shard, "*.json"))):
            base = os.path.basename(jp)[: -len(".json")]
            head, _, frame_str = base.rpartition("-")
            try:
                frame_idx = int(frame_str)
            except ValueError:
                continue
            groups[head].append((frame_idx, shard, base))

    sequences: list[dict] = []
    for scene_cam, frames in sorted(groups.items()):
        frames_sorted = sorted(frames, key=lambda x: x[0])
        if len(frames_sorted) < min_frames:
            continue
        scene, cam = scene_cam.rsplit("-", 1)
        sequences.append({
            "scene_cam": scene_cam,
            "scene": scene,
            "cam": cam,
            "frames": frames_sorted,
        })
    return sequences


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

def _frame_records(seq: dict) -> list[dict]:
    """One JSON-friendly dict per frame; local_id = position in the list."""
    out = []
    for local_id, (frame_idx, shard_dir, base) in enumerate(seq["frames"]):
        out.append({
            "local_id": int(local_id),
            "frame_idx": int(frame_idx),
            "base": str(base),
            "shard_dir": str(shard_dir),
        })
    return out


def _sequence_header(seq: dict, seq_index: int, num_frames: int) -> dict:
    return {
        "seq_index": int(seq_index),
        "seq_name": f"roomenv_{seq['scene']}_{seq['cam']}",
        "scene_cam": str(seq["scene_cam"]),
        "scene": str(seq["scene"]),
        "cam": str(seq["cam"]),
        "num_frames": int(num_frames),
    }


def build_per_view_manifest(
    sequences: list[dict],
    split: str,
    num_views: int,
    base_seed: int,
    shuffle: bool,
    data_dir: str,
    min_frames: int,
) -> dict:
    samples: list[dict] = []
    leftovers: list[dict] = []

    for seq_index, seq in enumerate(sequences):
        frames_dbg = _frame_records(seq)
        n_frames = len(frames_dbg)
        header = _sequence_header(seq, seq_index, n_frames)

        ids_pool = np.arange(n_frames, dtype=np.int64)
        if shuffle:
            seed = _stable_seed(
                "seeded_non_overlapping_frame_grouping",
                base_seed, split, header["scene_cam"], num_views,
            )
            rng = np.random.default_rng(seed)
            rng.shuffle(ids_pool)

        n_full = n_frames // num_views
        for g in range(n_full):
            grp = ids_pool[g * num_views:(g + 1) * num_views]
            grp_sorted = sorted(int(x) for x in grp)
            item = dict(header)
            item.update({
                "num_views": int(num_views),
                "group_id": int(g),
                "ids": grp_sorted,
                "frames": [frames_dbg[i] for i in grp_sorted],
            })
            samples.append(item)

        leftover_ids = sorted(int(x) for x in ids_pool[n_full * num_views:])
        if leftover_ids:
            leftovers.append({
                **header,
                "num_views": int(num_views),
                "ids": leftover_ids,
                "frames": [frames_dbg[i] for i in leftover_ids],
            })

    return {
        "meta": {
            "strategy": "seeded_non_overlapping_frame_grouping",
            "split": split,
            "num_views": int(num_views),
            "base_seed": int(base_seed),
            "shuffle": bool(shuffle),
            "data_dir": str(data_dir),
            "min_frames": int(min_frames),
            "note": "Each eval sample uses explicit ids. Leftovers are dropped.",
        },
        "samples": samples,
        "leftovers": leftovers,
    }


def build_mixed_val_manifest(
    sequences: list[dict],
    split: str,
    view_counts: list[int],
    base_seed: int,
    shuffle: bool,
    samples_per_sequence_per_view: int,
    data_dir: str,
    min_frames: int,
) -> dict:
    if split != "val":
        raise ValueError(f"mixed manifest is validation-only; got split={split!r}")

    samples: list[dict] = []
    next_sample_id = 0

    for seq_index, seq in enumerate(sequences):
        frames_dbg = _frame_records(seq)
        n_frames = len(frames_dbg)
        header = _sequence_header(seq, seq_index, n_frames)

        for num_views in view_counts:
            if n_frames < num_views:
                continue
            for s in range(samples_per_sequence_per_view):
                seed = _stable_seed(
                    "seeded_overlapping_mixed_view_grouping",
                    base_seed, split, header["scene_cam"], num_views, s,
                )
                rng = np.random.default_rng(seed)
                picked = rng.choice(n_frames, size=num_views, replace=False)
                picked_sorted = sorted(int(x) for x in picked)
                item = dict(header)
                item.update({
                    "num_views": int(num_views),
                    "sample_id": int(next_sample_id),
                    "ids": picked_sorted,
                    "frames": [frames_dbg[i] for i in picked_sorted],
                })
                samples.append(item)
                next_sample_id += 1

    return {
        "meta": {
            "strategy": "seeded_overlapping_mixed_view_grouping",
            "split": split,
            "view_counts": [int(v) for v in view_counts],
            "base_seed": int(base_seed),
            "shuffle": bool(shuffle),
            "overlap_allowed_across_samples": True,
            "duplicate_ids_within_sample": False,
            "samples_per_sequence_per_view": int(samples_per_sequence_per_view),
            "data_dir": str(data_dir),
            "min_frames": int(min_frames),
            "note": (
                "Validation-only mixed-view manifest. Samples may overlap "
                "across groups, but each sample has unique ids."
            ),
        },
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_per_view_manifest(manifest: dict, sequences: list[dict]) -> None:
    """Raise ``ValueError`` on any inconsistency."""
    num_views = int(manifest["meta"]["num_views"])
    seq_n_frames = {i: len(seq["frames"]) for i, seq in enumerate(sequences)}

    groups_by_seq: dict[int, list[list[int]]] = {}
    for item in manifest["samples"]:
        if int(item["num_views"]) != num_views:
            raise ValueError(
                f"sample claims num_views={item['num_views']} but manifest "
                f"meta.num_views={num_views}"
            )
        if len(item["ids"]) != num_views:
            raise ValueError(
                f"seq_index={item['seq_index']}: group has "
                f"{len(item['ids'])} ids, expected {num_views}"
            )
        if len(set(item["ids"])) != num_views:
            raise ValueError(
                f"seq_index={item['seq_index']}: duplicate ids in group: "
                f"{item['ids']}"
            )
        si = int(item["seq_index"])
        nf = seq_n_frames.get(si)
        if nf is None:
            raise ValueError(f"seq_index {si} out of dataset range")
        for idx in item["ids"]:
            if idx < 0 or idx >= nf:
                raise ValueError(
                    f"seq_index={si}: id {idx} out of range [0, {nf})"
                )
        groups_by_seq.setdefault(si, []).append(list(item["ids"]))

    leftovers_by_seq: dict[int, list[int]] = {}
    for item in manifest["leftovers"]:
        leftovers_by_seq[int(item["seq_index"])] = list(item["ids"])

    # Partition check: groups + leftover exactly cover [0, n_frames).
    all_seq_indices = set(seq_n_frames.keys())
    touched_seq_indices = set(groups_by_seq.keys()) | set(leftovers_by_seq.keys())
    for si in touched_seq_indices:
        nf = seq_n_frames[si]
        used: set[int] = set()
        for g in groups_by_seq.get(si, []):
            for x in g:
                if x in used:
                    raise ValueError(
                        f"seq_index={si}: id {x} appears in multiple groups"
                    )
                used.add(int(x))
        leftover = set(int(x) for x in leftovers_by_seq.get(si, []))
        inter = used & leftover
        if inter:
            raise ValueError(
                f"seq_index={si}: ids overlap between groups and leftovers: "
                f"{sorted(inter)}"
            )
        covered = used | leftover
        expected = set(range(nf))
        if covered != expected:
            missing = expected - covered
            extra = covered - expected
            raise ValueError(
                f"seq_index={si}: groups+leftovers != [0,{nf}). "
                f"missing={sorted(missing)[:10]} extra={sorted(extra)[:10]}"
            )

    # Sequences with n_frames < num_views won't be touched at all, they
    # contribute a leftover only if they have any frames; with num_views==1
    # they always produce one full group. Validate consistency:
    for si in all_seq_indices - touched_seq_indices:
        if seq_n_frames[si] >= num_views:
            raise ValueError(
                f"seq_index={si} has {seq_n_frames[si]} frames but is "
                f"missing from both groups and leftovers"
            )


def verify_mixed_manifest(manifest: dict, sequences: list[dict]) -> None:
    seq_n_frames = {i: len(seq["frames"]) for i, seq in enumerate(sequences)}
    counts_by_view: dict[int, int] = {}
    for item in manifest["samples"]:
        num_views = int(item["num_views"])
        ids = item["ids"]
        if len(ids) != num_views:
            raise ValueError(
                f"sample_id={item.get('sample_id')}: len(ids)={len(ids)} "
                f"but num_views={num_views}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"sample_id={item.get('sample_id')}: duplicate ids {ids}"
            )
        si = int(item["seq_index"])
        nf = seq_n_frames.get(si)
        if nf is None:
            raise ValueError(f"seq_index {si} out of dataset range")
        for x in ids:
            if x < 0 or x >= nf:
                raise ValueError(
                    f"sample_id={item.get('sample_id')}: id {x} out of "
                    f"range [0, {nf})"
                )
        counts_by_view[num_views] = counts_by_view.get(num_views, 0) + 1
    print(f"  mixed verify: counts by num_views = {counts_by_view}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def _save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _print_per_view_summary(manifest: dict, sequences: list[dict], out_path: str) -> None:
    meta = manifest["meta"]
    n_seq = len(sequences)
    total_frames = sum(len(s["frames"]) for s in sequences)
    print("=" * 70)
    print(f"  split             : {meta['split']}")
    print(f"  num_views         : {meta['num_views']}")
    print(f"  strategy          : {meta['strategy']}")
    print(f"  sequences         : {n_seq}")
    print(f"  total frames      : {total_frames}")
    print(f"  base_seed         : {meta['base_seed']}  shuffle={meta['shuffle']}")
    print(f"  eval samples      : {len(manifest['samples'])}")
    print(f"  with leftovers    : {len(manifest['leftovers'])} sequences")
    for it in manifest["samples"][:3]:
        print(f"    sample seq_index={it['seq_index']} "
              f"scene_cam={it['scene_cam']} ids={it['ids']}")
    print(f"  saved             : {out_path}")
    print("=" * 70)


def _print_mixed_summary(manifest: dict, out_path: str) -> None:
    meta = manifest["meta"]
    total = len(manifest["samples"])
    counts: dict[int, int] = {}
    for it in manifest["samples"]:
        counts[int(it["num_views"])] = counts.get(int(it["num_views"]), 0) + 1
    print("=" * 70)
    print(f"  mixed manifest    : {meta['split']}")
    print(f"  strategy          : {meta['strategy']}")
    print(f"  total samples     : {total}")
    print(f"  counts by views   : {counts}")
    for it in manifest["samples"][:3]:
        print(f"    sample sample_id={it['sample_id']} "
              f"seq_index={it['seq_index']} nv={it['num_views']} ids={it['ids']}")
    print(f"  saved             : {out_path}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", dest="data_dir", default=DEFAULT_DATA_DIR,
                    help=f"Root of the extracted Room Envelopes dataset (default: {DEFAULT_DATA_DIR}).")
    sp = ap.add_mutually_exclusive_group()
    sp.add_argument("--split", choices=("val", "test"), default=None)
    sp.add_argument("--splits", nargs="+", choices=("val", "test"), default=None)
    ap.add_argument("--output-dir", dest="output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--view-counts", dest="view_counts", nargs="+", type=int,
                    default=DEFAULT_VIEW_COUNTS)
    ap.add_argument("--base-seed", dest="base_seed", type=int, default=DEFAULT_BASE_SEED)
    ap.add_argument("--shuffle", type=_parse_bool, default=True)
    ap.add_argument("--min-frames", dest="min_frames", type=int, default=DEFAULT_MIN_FRAMES)

    ap.add_argument("--build-mixed-val", dest="build_mixed_val", action="store_true",
                    help="Also emit the validation-only mixed-overlapping manifest.")
    ap.add_argument("--mixed-samples-per-sequence-per-view", dest="mixed_samples",
                    type=int, default=2)
    ap.add_argument("--mixed-output", dest="mixed_output", default=None,
                    help="Optional explicit path for the mixed-val manifest.")

    args = ap.parse_args()

    splits = args.splits if args.splits else (
        [args.split] if args.split else list(DEFAULT_SPLITS)
    )

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    print(f"[manifest] data_dir={args.data_dir} splits={splits} "
          f"base_seed={args.base_seed} shuffle={args.shuffle} "
          f"view_counts={args.view_counts} min_frames={args.min_frames}")

    for split in splits:
        print(f"\n[manifest] scanning split={split} ...")
        sequences = scan_sequences(args.data_dir, split, args.min_frames)
        print(f"[manifest] found {len(sequences)} sequences, "
              f"{sum(len(s['frames']) for s in sequences)} frames")

        for num_views in args.view_counts:
            manifest = build_per_view_manifest(
                sequences,
                split=split,
                num_views=num_views,
                base_seed=args.base_seed,
                shuffle=args.shuffle,
                data_dir=args.data_dir,
                min_frames=args.min_frames,
            )
            verify_per_view_manifest(manifest, sequences)
            out_path = os.path.join(
                output_dir,
                f"eval_manifest_{split}_{num_views}view_seed{args.base_seed}.json",
            )
            _save_json(out_path, manifest)
            _print_per_view_summary(manifest, sequences, out_path)

        if args.build_mixed_val and split == "val":
            print(f"\n[manifest] building mixed-overlapping val manifest ...")
            mixed = build_mixed_val_manifest(
                sequences,
                split=split,
                view_counts=args.view_counts,
                base_seed=args.base_seed,
                shuffle=args.shuffle,
                samples_per_sequence_per_view=args.mixed_samples,
                data_dir=args.data_dir,
                min_frames=args.min_frames,
            )
            verify_mixed_manifest(mixed, sequences)
            mixed_path = args.mixed_output or os.path.join(
                output_dir,
                f"eval_manifest_val_mixed_1to5_overlap_seed{args.base_seed}.json",
            )
            _save_json(mixed_path, mixed)
            _print_mixed_summary(mixed, mixed_path)

    if args.build_mixed_val and "val" not in splits:
        print("[manifest] --build-mixed-val requested but 'val' not in --splits; skipped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
