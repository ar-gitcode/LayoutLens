"""Shared argparse flag definitions for the eval runners.

Each helper registers the canonical hyphenated flag **and** the legacy
underscore spelling as an alias (same ``dest``), so existing command lines using
either convention keep working. Namespace attribute names are the underscore
form (e.g. ``args.camera_mode``).
"""
from __future__ import annotations

import argparse


def add_camera_mode(p: argparse.ArgumentParser, allow_both: bool = False) -> None:
    choices = ("gt", "pred", "both") if allow_both else ("gt", "pred")
    help_txt = ("Camera source for 3D reconstruction: 'gt' (default) or 'pred'.")
    if allow_both:
        help_txt += (" 'both' runs the gt and pred reconstructions in a single "
                     "pass; predicted-camera 3D metrics are emitted with a "
                     "'predcam_' prefix alongside the gt ones.")
    p.add_argument("--camera-mode", "--camera_mode", dest="camera_mode",
                   default="gt", choices=choices, help=help_txt)


def add_extrinsics_convention(p: argparse.ArgumentParser) -> None:
    p.add_argument("--extrinsics-convention", "--extrinsics_convention",
                   dest="extrinsics_convention", default="w2c", choices=("w2c", "c2w"),
                   help="Dataset stores camera-from-world (w2c). Override only if "
                        "your inputs differ.")


def add_max_points_per_scene(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-points-per-scene", "--max_points_per_scene",
                   dest="max_points_per_scene", type=int, default=50_000)


def add_use_depth_as_layout(p: argparse.ArgumentParser) -> None:
    p.add_argument("--use-depth-as-layout", "--use_depth_as_layout",
                   dest="use_depth_as_layout", action="store_true",
                   help="E0: if no layout_depth output, use 'depth' output as "
                        "layout-depth proxy")


def add_device(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default=None)


def add_split(p: argparse.ArgumentParser, default: str = "val") -> None:
    p.add_argument("--split", default=default, choices=("train", "val", "test"))
