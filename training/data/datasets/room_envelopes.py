"""Room Envelopes dataset loader for VGGT layout-depth training.

Routes all I/O through :mod:`room_envelopes.io` and :mod:`room_envelopes.extrinsics`
so the decode logic (MoGe log-depth, Z-flipped normals, manifest poses) lives in
one place and can be unit-tested.

Yielded keys (one per sampled view, lists of per-frame tensors/arrays):
    seq_name, ids, frame_num, images, depths, layout_depths,
    layout_masks, layout_depth_masks, layout_normals, layout_normal_masks,
    point_masks, extrinsics, intrinsics, cam_points, world_points

Mask semantics (do NOT conflate):
- ``layout_depth_masks`` = ``layout_depth > 1e-6`` (validity of layout depth).
- ``layout_masks``       = visibility, by default the pre-stored ``.seen_mask``
  thresholded at 127, or derived from ``|vis_depth - layout_depth|`` when
  ``mask_source="derived"``.
- ``layout_normal_masks`` = ``(layout_depth > 1e-6) & (||normal|| > 1e-6)``.

Depth notes:
- ``.depth`` and ``.layout_depth`` are decoded by ``room_envelopes.io.decode_moge_depth``
  (MoGe log-encoded uint16 with PNG ``near``/``far`` tEXt chunks → metric metres).
- NaN/sky sentinels are collapsed to 0.0 so the ``> 1e-6`` validity check still works.
- ``meters_per_asset_unit`` from the extrinsics manifest is NEVER applied to depth.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pickle
import random
from collections import defaultdict

import numpy as np

from data.base_dataset import BaseDataset

# room_envelopes/ lives next to training/ at the repo root; the train.py entry
# point puts the repo root on sys.path so this import resolves.
from room_envelopes.io import (
    decode_moge_depth,
    derive_seen_mask,
    load_intrinsics_pixel,
    load_normal_png,
    load_rgb,
    load_seen_mask,
)
from room_envelopes.extrinsics import ExtrinsicsManifest, MissingExtrinsicsError
from room_envelopes.paths import DATA_WDS_DIR

logger = logging.getLogger(__name__)


_DEFAULT_CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "cache", "room_envelopes")
)

SPLITS = ("train", "val", "test")


class RoomEnvelopesDataset(BaseDataset):
    """Room Envelopes loader. One sequence = one ``(scene, camera)`` pair."""

    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_DIR: str = DATA_WDS_DIR,
        min_frames: int = 1,
        len_train: int = 50000,
        len_val: int = 5000,
        len_test: int = 4000,
        load_mask: bool = True,
        load_normal: bool = False,
        mask_source: str = "seen_mask",
        mask_occlusion_thresh: float = 1e-6,
        normal_source: str = "layout_normal",
        use_sequence_cache: bool = True,
        refresh_sequence_cache: bool = False,
        sequence_cache_dir: str | None = None,
        sequence_cache_version: str = "re_moge_v1",
        use_real_extrinsics: bool = False,
        extrinsics_manifest: str | None = None,
        require_extrinsics: bool = False,
        fallback_identity_extrinsics: bool = True,
        exclude_invalid_extrinsics: bool = True,
        manifest_path: str | None = None,
    ):
        """
        Args:
            common_conf:            Shared config (img_size, patch_size, augs, ...).
            split:                  ``"train"``, ``"val"``, or ``"test"``.
            DATA_DIR:               Root of the extracted Room Envelopes dataset.
            min_frames:             Skip ``(scene, cam)`` sequences with fewer frames.
            load_mask / load_normal: Whether to populate ``layout_masks``/``layout_normals``.
            mask_source:            ``"seen_mask"`` (default) or ``"derived"``.
            mask_occlusion_thresh:  Threshold for ``"derived"`` masks.
            normal_source:          ``"layout_normal"`` or ``"layout_normal_rerendered"``.
            use_sequence_cache:     Cache the per-split file scan to disk.
            sequence_cache_version: Bump to invalidate stale caches.

            use_real_extrinsics:    Swap identity placeholder for OpenCV w2c 3×4
                                    poses recovered from Hypersim.
            extrinsics_manifest:    NPZ path; required when ``use_real_extrinsics``.
            require_extrinsics:     Raise on any frame lacking a valid pose.
            fallback_identity_extrinsics:
                                    Only consulted when ``use_real_extrinsics``
                                    is True AND ``require_extrinsics`` is False.
            exclude_invalid_extrinsics:
                                    Drop manifest-invalid samples up front.
        """
        super().__init__(common_conf=common_conf)

        if mask_source not in ("seen_mask", "derived"):
            raise ValueError(f"mask_source must be 'seen_mask' or 'derived', got {mask_source!r}")
        if normal_source not in ("layout_normal", "layout_normal_rerendered"):
            raise ValueError(
                f"normal_source must be 'layout_normal' or 'layout_normal_rerendered', "
                f"got {normal_source!r}"
            )
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = getattr(common_conf, "get_nearby", False)
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = getattr(common_conf, "allow_duplicate_img", True)

        self.DATA_DIR = DATA_DIR
        self.min_frames = min_frames
        self.load_mask = load_mask
        self.load_normal = load_normal
        self.mask_source = mask_source
        self.mask_occlusion_thresh = mask_occlusion_thresh
        self.normal_source = normal_source

        self.use_sequence_cache = use_sequence_cache
        self.refresh_sequence_cache = refresh_sequence_cache
        self.sequence_cache_version = sequence_cache_version
        self._sequence_cache_dir = sequence_cache_dir or _DEFAULT_CACHE_DIR

        self.use_real_extrinsics = use_real_extrinsics
        self.extrinsics_manifest_path = extrinsics_manifest
        self.require_extrinsics = require_extrinsics
        self.fallback_identity_extrinsics = fallback_identity_extrinsics
        self.exclude_invalid_extrinsics = exclude_invalid_extrinsics
        self._manifest: ExtrinsicsManifest | None = None
        self._fallback_warned = False

        if self.use_real_extrinsics:
            if not self.extrinsics_manifest_path:
                raise ValueError(
                    "extrinsics_manifest is required when use_real_extrinsics=True; "
                    "pass the NPZ path built by build_extrinsics_manifest.py."
                )
            if self.require_extrinsics and self.fallback_identity_extrinsics:
                logger.warning(
                    "[RoomEnvelopes] require_extrinsics=True overrides "
                    "fallback_identity_extrinsics=True; missing poses will raise."
                )
            self._manifest = ExtrinsicsManifest(self.extrinsics_manifest_path)

        self._split = split

        if split == "train":
            self.len_train = len_train
        elif split == "val":
            self.len_train = len_val
        else:
            self.len_train = len_test

        self.manifest_path = manifest_path
        self.manifest_samples: list | None = None

        if manifest_path is not None:
            # Manifest-driven mode: deterministic sample list. Each entry has its own
            # frame group and num_views; ignore the file-scan / inside_random path.
            self.manifest_samples = self._load_manifest(manifest_path)
            self.manifest_samples = self._filter_manifest_samples_by_extrinsics(
                self.manifest_samples
            )
            self.sequence_list_len = len(self.manifest_samples)
            self.sequences = []
            logger.info(
                f"[RoomEnvelopes] split={split}: manifest mode, "
                f"{self.sequence_list_len} samples from {manifest_path}, "
                f"use_real_extrinsics={self.use_real_extrinsics}"
            )
        else:
            sequences, cache_status = self._load_or_build_sequences()
            if self.use_real_extrinsics:
                sequences = self._apply_extrinsics_filter(sequences)
            self.sequences = sequences
            self.sequence_list_len = len(self.sequences)

            logger.info(
                f"[RoomEnvelopes] split={split}: {self.sequence_list_len} sequences "
                f"({sum(len(s['frames']) for s in self.sequences)} frames), "
                f"cache={cache_status}, use_real_extrinsics={self.use_real_extrinsics}"
            )

        if self.sequence_list_len == 0:
            raise RuntimeError(
                f"No Room Envelopes sequences found for split={split} in {DATA_DIR}"
            )

    def _load_manifest(self, path: str) -> list:
        with open(path, "r") as f:
            m = json.load(f)
        out: list[dict] = []
        for s in m["samples"]:
            frames = [
                (int(fr["frame_idx"]), str(fr["shard_dir"]), str(fr["base"]))
                for fr in s["frames"]
            ]
            out.append({
                "seq_name":  s["seq_name"],
                "scene_cam": s["scene_cam"],
                "scene":     s.get("scene", ""),
                "cam":       s.get("cam", ""),
                "num_views": int(s["num_views"]),
                "frames":    frames,
            })
        # Sort by (num_views, seq_name) so the batch sampler can group same-num_views
        # samples into contiguous batches without crossing num_views boundaries.
        out.sort(key=lambda s: (s["num_views"], s["seq_name"], s["frames"][0][2]))
        return out

    # ------------------------------------------------------------------ #
    # Sequence cache
    # ------------------------------------------------------------------ #

    def _make_cache_key(self) -> dict:
        return {
            "DATA_DIR": self.DATA_DIR,
            "split": self._split,
            "min_frames": self.min_frames,
            "debug": self.debug,
            "version": self.sequence_cache_version,
        }

    def _cache_path(self) -> str:
        tag = "debug" if self.debug else "full"
        return os.path.join(
            self._sequence_cache_dir,
            f"{self.sequence_cache_version}_{self._split}_min{self.min_frames}_{tag}.pkl",
        )

    def _load_sequence_cache(self, path: str):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning(f"[RoomEnvelopes] could not read cache {path}: {e}")
            return None
        if data.get("cache_key") != self._make_cache_key():
            logger.warning(f"[RoomEnvelopes] cache key mismatch at {path}; rebuilding")
            return None
        return data.get("sequences")

    def _save_sequence_cache(self, path: str, sequences: list) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "wb") as f:
                pickle.dump(
                    {"cache_key": self._make_cache_key(), "sequences": sequences},
                    f, protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info(f"[RoomEnvelopes] cache saved ({len(sequences)} sequences) → {path}")
        except Exception as e:
            logger.warning(f"[RoomEnvelopes] could not write cache: {e}")

    def _load_or_build_sequences(self):
        if self.use_sequence_cache:
            p = self._cache_path()
            if not self.refresh_sequence_cache and os.path.isfile(p):
                cached = self._load_sequence_cache(p)
                if cached is not None:
                    return cached, "hit"
            built = self._build_sequences()
            self._save_sequence_cache(p, built)
            return built, "miss"
        return self._build_sequences(), "disabled"

    def _build_sequences(self) -> list:
        shard_pattern = os.path.join(self.DATA_DIR, f"{self._split}-*")
        groups: dict[str, list[tuple[int, str, str]]] = defaultdict(list)

        shard_dirs = sorted(glob.glob(shard_pattern))
        if self.debug:
            shard_dirs = shard_dirs[:1]

        for shard in shard_dirs:
            for jp in sorted(glob.glob(os.path.join(shard, "*.json"))):
                base = os.path.basename(jp)[: -len(".json")]
                head, _, frame_str = base.rpartition("-")
                try:
                    frame_idx = int(frame_str)
                except ValueError:
                    continue
                groups[head].append((frame_idx, shard, base))

        sequences = []
        for scene_cam, frames in sorted(groups.items()):
            frames_sorted = sorted(frames, key=lambda x: x[0])
            if len(frames_sorted) < self.min_frames:
                continue
            scene, cam = scene_cam.rsplit("-", 1)
            sequences.append({
                "scene_cam": scene_cam,
                "scene":     scene,
                "cam":       cam,
                "frames":    [(f, d, b) for (f, d, b) in frames_sorted],
            })
        return sequences

    # ------------------------------------------------------------------ #
    # Real extrinsics
    # ------------------------------------------------------------------ #

    def _filter_manifest_samples_by_extrinsics(self, manifest_samples: list) -> list:
        # Manifest-mode counterpart to _apply_extrinsics_filter. Only fires when
        # the caller has declared that identity fallback is unacceptable
        # (require_extrinsics=True), epipolar OCA configs. Non-epipolar configs
        # leave the manifest untouched and rely on identity fallback at load time.
        if not (self.use_real_extrinsics and self.require_extrinsics):
            return manifest_samples
        if self._manifest is None:
            return manifest_samples
        valid = self._manifest.valid_bases()
        kept: list = []
        n_drop = 0
        for s in manifest_samples:
            bases = [fr[2] for fr in s["frames"]]
            if all(b in valid for b in bases):
                kept.append(s)
            else:
                n_drop += 1
        logger.info(
            f"[RoomEnvelopes] require_extrinsics=True: dropped {n_drop} "
            f"manifest samples with invalid/missing extrinsics "
            f"(kept {len(kept)} / {len(manifest_samples)})"
        )
        return kept

    def _apply_extrinsics_filter(self, sequences: list) -> list:
        """Drop manifest-invalid/missing samples up front if requested."""
        if not self.exclude_invalid_extrinsics or self._manifest is None:
            return sequences
        valid = self._manifest.valid_bases()
        n_dropped_invalid = 0
        n_dropped_missing = 0
        n_dropped_sequences = 0
        out: list[dict] = []
        for s in sequences:
            keep = []
            for tup in s["frames"]:
                _frame_idx, _shard, base = tup
                if base not in self._manifest:
                    n_dropped_missing += 1
                    continue
                if base not in valid:
                    n_dropped_invalid += 1
                    continue
                keep.append(tup)
            if len(keep) < self.min_frames:
                n_dropped_sequences += 1
                continue
            new_seq = dict(s)
            new_seq["frames"] = keep
            out.append(new_seq)
        logger.info(
            f"[RoomEnvelopes] exclude_invalid_extrinsics: dropped "
            f"{n_dropped_invalid} invalid, {n_dropped_missing} missing, "
            f"{n_dropped_sequences} sequences below min_frames"
        )
        return out

    def _resolve_extrinsics(self, base: str, identity: np.ndarray) -> np.ndarray:
        try:
            return self._manifest.w2c34(base)
        except MissingExtrinsicsError as e:
            if self.require_extrinsics or not self.fallback_identity_extrinsics:
                raise RuntimeError(
                    f"[RoomEnvelopes] real extrinsics unavailable for {base!r}: {e}"
                ) from e
            if not self._fallback_warned:
                logger.warning(
                    f"[RoomEnvelopes] falling back to identity extrinsics (first: {base!r}): {e}. "
                    f"Set require_extrinsics=True to fail loudly instead."
                )
                self._fallback_warned = True
            return identity.copy()

    # ------------------------------------------------------------------ #
    # Frame loading
    # ------------------------------------------------------------------ #

    def _load_frame(self, shard_dir: str, base: str):
        p = os.path.join(shard_dir, base)
        image = load_rgb(p + ".image")
        vis_depth, _vis_near, _vis_far = decode_moge_depth(p + ".depth")
        lay_depth, _lay_near, _lay_far = decode_moge_depth(p + ".layout_depth")

        H, W = image.shape[:2]
        intri = load_intrinsics_pixel(p + ".json", H=H, W=W)

        if self.load_mask:
            if self.mask_source == "seen_mask":
                seg_mask = load_seen_mask(p + ".seen_mask")
            else:  # "derived"
                seg_mask = derive_seen_mask(
                    vis_depth, lay_depth, self.mask_occlusion_thresh,
                )
        else:
            seg_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        normal = None
        if self.load_normal:
            normal_path = p + "." + self.normal_source
            normal = load_normal_png(normal_path)
            if normal is None:
                normal = np.zeros((*image.shape[:2], 3), dtype=np.float32)

        return {
            "image":     image,
            "vis_depth": vis_depth,
            "lay_depth": lay_depth,
            "intri":     intri,
            "seg_mask":  seg_mask,
            "normal":    normal,
        }

    # ------------------------------------------------------------------ #
    # BaseDataset interface
    # ------------------------------------------------------------------ #

    def get_data(
        self,
        seq_index: int | None = None,
        img_per_seq: int | None = None,
        seq_name: str | None = None,
        ids: list | None = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.manifest_samples is not None:
            # Manifest mode: each manifest sample is one fixed (scene_cam, num_views, frames)
            # unit. Ignore the sampler's random seq_index/img_per_seq and use what the
            # manifest dictates.
            sample = self.manifest_samples[seq_index]
            frames = sample["frames"]
            n_frames = len(frames)
            ids = np.arange(n_frames, dtype=np.int64)
            seq = {"scene": sample["scene"], "cam": sample["cam"], "frames": frames}
        else:
            if self.inside_random:
                seq_index = random.randint(0, self.sequence_list_len - 1)

            seq = self.sequences[seq_index]
            frames = seq["frames"]
            n_frames = len(frames)

            if ids is not None:
                ids = np.asarray(ids, dtype=np.int64)

                if img_per_seq is not None and len(ids) != img_per_seq:
                    raise ValueError(
                        f"Expected {img_per_seq} ids, got {len(ids)}: {ids}"
                    )

                if np.any(ids < 0) or np.any(ids >= n_frames):
                    raise ValueError(
                        f"ids out of range for sequence with {n_frames} frames: {ids}"
                    )
            else:
                replace = self.allow_duplicate_img or (img_per_seq > n_frames)
                ids = np.random.choice(n_frames, img_per_seq, replace=replace)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, layout_depths = [], [], []
        layout_masks, layout_depth_masks = [], []
        layout_normals, layout_normal_masks = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points, point_masks = [], [], []

        extri_identity = np.concatenate(
            [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1
        )

        for idx in ids:
            frame_idx, shard_dir, base = frames[int(idx)]
            d = self._load_frame(shard_dir, base)

            image     = d["image"]
            vis_depth = d["vis_depth"]
            lay_depth = d["lay_depth"]
            intri     = d["intri"]
            seg_mask  = d["seg_mask"]
            raw_normal = d["normal"]

            original_size = np.array(image.shape[:2])
            if self.use_real_extrinsics:
                extri = self._resolve_extrinsics(base, extri_identity)
            else:
                extri = extri_identity.copy()

            if self.load_normal:
                (
                    image,
                    vis_depth,
                    lay_depth,
                    proc_mask,
                    proc_normal,
                    extri,
                    intri,
                    world_coords,
                    cam_coords,
                    point_mask,
                    layout_depth_mask,
                    layout_normal_mask,
                    _,
                ) = self.process_one_image_w_layout_depth_seg_and_normals(
                    image, vis_depth, lay_depth, seg_mask, raw_normal,
                    extri, intri, original_size, target_image_shape,
                )
                layout_normals.append(proc_normal.astype(np.float32))
                layout_normal_masks.append(layout_normal_mask)
            else:
                (
                    image,
                    vis_depth,
                    lay_depth,
                    proc_mask,
                    extri,
                    intri,
                    world_coords,
                    cam_coords,
                    point_mask,
                    layout_depth_mask,
                    _,
                ) = self.process_one_image_w_layout_depth_and_seg(
                    image, vis_depth, lay_depth, seg_mask,
                    extri, intri, original_size, target_image_shape,
                )

            images.append(image)
            depths.append(vis_depth)
            layout_depths.append(lay_depth)
            layout_masks.append(proc_mask.astype(np.float32))
            layout_depth_masks.append(layout_depth_mask)
            extrinsics.append(extri)
            intrinsics.append(intri)
            cam_points.append(cam_coords)
            world_points.append(world_coords)
            point_masks.append(point_mask)

        return {
            "seq_name":            f"roomenv_{seq['scene']}_{seq['cam']}",
            "ids":                 np.array(ids),
            "frame_num":           len(extrinsics),
            "images":              images,
            "depths":              depths,
            "layout_depths":       layout_depths,
            "layout_masks":        layout_masks,
            "layout_depth_masks":  layout_depth_masks,
            "layout_normals":      layout_normals if self.load_normal else None,
            "layout_normal_masks": layout_normal_masks if self.load_normal else None,
            "point_masks":         point_masks,
            "extrinsics":          extrinsics,
            "intrinsics":          intrinsics,
            "cam_points":          cam_points,
            "world_points":        world_points,
        }
