# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from wcmatch import fnmatch
from functools import wraps
from typing import List

import torch.nn as nn

# ------------------------------------------------------------
# Glob‑matching flags (behave like the Unix shell) 
# ------------------------------------------------------------
GLOB_FLAGS = (
    fnmatch.CASE       # case‑sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out‑of‑the‑box
)


def freeze_modules(model: nn.Module, patterns: List[str], recursive: bool = True) -> nn.Module:
    """Freeze (stop training) parts of *model* whose *name* matches *patterns*.

    Parameters
    ----------
    model : nn.Module
        The complete model you are working with.
    patterns : list[str]
        Glob patterns to match sub‑module names.  Example: ``["encoder.*", "cls_head"]``
    recursive : bool, default = True
        • ``True``  → also freeze every child of a matched module.
        • ``False`` → freeze only the matched module itself.

    Returns
    -------
    nn.Module
        The same model object, now with some parts frozen.

    Example
    -------
    >>> freeze_modules(model, ["encoder.*", "decoder.layer1"], recursive=True)
    """
    matched: set[str] = set()

    for name, mod in model.named_modules():
        # does *name* match ANY user pattern?
        if any(fnmatch.fnmatch(name, p, flags=GLOB_FLAGS) for p in patterns):
            matched.add(name)
            _freeze(mod, recursive)

    _check_every_pattern_used(matched, patterns)
    return model


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _freeze(mod: nn.Module, recursive: bool) -> None:
    """Put *mod* in eval mode and lock its parameters."""

    if recursive:
        mod.eval()            # affects the whole subtree
    else:
        mod.training = False  # only this exact module

    original_train = mod.train

    @wraps(original_train)
    def locked_train(mode: bool = True):
        if recursive:
            return original_train(False)  # ignore user's *mode*
        out = original_train(mode)        # children follow user's choice
        out.training = False              # but this module stays frozen
        return out

    mod.train = locked_train  # type: ignore[attr-defined]

    param_iter = (
        mod.parameters()              # default recurse=True
        if recursive
        else mod.parameters(recurse=False)
    )
    for p in param_iter:
        p.requires_grad = False


def _check_every_pattern_used(matched_names: set[str], patterns: List[str]):
    unused = [p for p in patterns if not any(fnmatch.fnmatch(n, p, flags=GLOB_FLAGS)
                                             for n in matched_names)]
    if unused:
        raise ValueError(f"These patterns matched nothing: {unused}")


# ------------------------------------------------------------
# Room-envelope extensions
# ------------------------------------------------------------

def unfreeze_last_n_blocks(model: nn.Module, parent_attr: str, n: int) -> None:
    """Unfreeze the last *n* transformer blocks within ``model.<parent_attr>``.

    Checks for ``.blocks`` first (generic); falls back to ``.frame_blocks`` and
    ``.global_blocks`` so that the VGGT ``Aggregator`` (which uses two parallel
    block lists instead of a single ``.blocks``) is handled correctly.

    Called after :func:`freeze_modules` to selectively re-enable the tail of the backbone.
    Sets ``requires_grad = True`` and puts the unfrozen blocks back into training mode.
    Logs the specific block indices that were unfrozen.
    """
    parent = getattr(model, parent_attr, None)
    if parent is None:
        logging.warning(f"[freeze] model has no attribute '{parent_attr}'; skipping unfreeze.")
        return

    # Prefer .blocks; fall back to Aggregator-style split lists
    if hasattr(parent, "blocks"):
        block_attrs = ["blocks"]
    else:
        block_attrs = [a for a in ("frame_blocks", "global_blocks") if hasattr(parent, a)]

    if not block_attrs:
        logging.warning(
            f"[freeze] model.{parent_attr} has no '.blocks', '.frame_blocks', or "
            f"'.global_blocks'; skipping unfreeze."
        )
        return

    for attr in block_attrs:
        blocks = getattr(parent, attr)
        total = len(blocks)
        n_actual = min(n, total)
        unfrozen_indices = list(range(total - n_actual, total))
        for block in list(blocks)[-n_actual:]:
            for p in block.parameters():
                p.requires_grad = True
            # Restore training mode, call the original .train() on the class,
            # bypassing the locked wrapper installed by freeze_modules().
            nn.Module.train(block, True)
        logging.info(
            f"[freeze] Unfroze last {n_actual}/{total} blocks of "
            f"model.{parent_attr}.{attr} (indices: {unfrozen_indices})"
        )


def print_trainable_param_summary(model: nn.Module, log_file: str = None) -> None:
    """Print a detailed table of trainable vs. frozen parameter counts.

    Lists overall totals, top-level child breakdown, and, for any top-level
    child that is *partially* trainable (e.g. the VGGT aggregator with some
    blocks unfrozen), drills one level deeper so the exact subgroups that
    received ``requires_grad = True`` are visible.

    Args:
        model:    the model to inspect.
        log_file: optional path. When provided, the same table is appended to
                  the file (and routed through the standard logger so it shows
                  up in the run's log_dir/log.txt as well).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    sep = "=" * 78
    lines = [
        sep,
        "Trainable parameter summary",
        sep,
        f"  TOTAL parameters:     {total:>14,}",
        f"  TRAINABLE parameters: {trainable:>14,}  ({100 * trainable / max(total, 1):.3f}%)",
        f"  FROZEN parameters:    {frozen:>14,}",
        "-" * 78,
        "  Top-level children:",
    ]
    partial = []  # (name, child) pairs that are only partially trainable
    for name, child in model.named_children():
        t = sum(p.numel() for p in child.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in child.parameters())
        if tot == 0:
            continue
        if t == 0:
            status = "FROZEN   "
        elif t == tot:
            status = "TRAINABLE"
        else:
            status = "PARTIAL  "
            partial.append((name, child))
        lines.append(f"    [{status}] {name:<28s} {t:>12,} / {tot:>12,}")

    # Drill one level into partially-trainable children (e.g. aggregator).
    for parent_name, parent_mod in partial:
        lines.append("-" * 78)
        lines.append(f"  Partial breakdown for '{parent_name}':")
        for sub_name, sub_mod in parent_mod.named_children():
            sub_total = sum(p.numel() for p in sub_mod.parameters())
            sub_train = sum(p.numel() for p in sub_mod.parameters() if p.requires_grad)
            if sub_total == 0:
                continue
            # For ModuleList of blocks, list per-block status as a compact pattern.
            if isinstance(sub_mod, nn.ModuleList):
                trainable_idx = [
                    i for i, blk in enumerate(sub_mod)
                    if any(p.requires_grad for p in blk.parameters())
                ]
                lines.append(
                    f"    {parent_name}.{sub_name:<22s} {sub_train:>12,} / {sub_total:>12,}  "
                    f"  trainable_block_indices={trainable_idx}"
                )
            else:
                status = "frozen" if sub_train == 0 else ("trainable" if sub_train == sub_total else "partial")
                lines.append(
                    f"    {parent_name}.{sub_name:<22s} {sub_train:>12,} / {sub_total:>12,}  [{status}]"
                )

    # List trainable parameter name globs (top-level prefixes).
    trainable_names = sorted({
        n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad
    })
    lines.append("-" * 78)
    lines.append(f"  Trainable top-level module groups: {trainable_names}")
    lines.append(sep)

    block = "\n".join(lines)
    print("\n" + block + "\n")
    logging.info("Trainable parameter summary:\n%s", block)

    if log_file:
        try:
            with open(log_file, "a") as f:
                f.write(block + "\n")
        except OSError as exc:
            logging.warning("Could not write trainable-param summary to %s: %s",
                            log_file, exc)
