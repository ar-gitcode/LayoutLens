"""Lightweight per-stage timing helper for eval scripts.

Single shared :class:`StageTimer` instance per run, passed through to the
per-scene helpers. When ``enabled=False`` every method is a fast no-op, so
callers can leave ``timer.time(...)`` context managers in place
unconditionally with no overhead on non-profile runs.
"""

from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class StageTimer:
    """Accumulate per-stage wall-clock samples and produce a JSON-safe report."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._samples: dict[str, list[float]] = {}

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._samples.setdefault(name, []).append(dt)

    def record(self, name: str, dt: float) -> None:
        """Manually record a single sample (seconds)."""
        if not self.enabled:
            return
        self._samples.setdefault(name, []).append(float(dt))

    def report(self) -> dict[str, dict[str, float]]:
        """Return per-stage stats: count, mean_s, median_s, p95_s, total_s."""
        if not self.enabled or not self._samples:
            return {}
        out: dict[str, dict[str, float]] = {}
        for name, xs_unsorted in self._samples.items():
            n = len(xs_unsorted)
            if n == 0:
                continue
            xs = sorted(xs_unsorted)
            total = float(sum(xs))
            p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
            out[name] = {
                "count": int(n),
                "mean_s": total / n,
                "median_s": float(statistics.median(xs)),
                "p95_s": float(xs[p95_idx]),
                "total_s": total,
            }
        return out

    def print_table(self, prefix: str = "") -> None:
        if not self.enabled:
            return
        rep = self.report()
        if not rep:
            return
        ordered = sorted(rep.items(), key=lambda kv: kv[1]["total_s"], reverse=True)
        name_w = max(20, max(len(k) for k in rep))
        header = (
            f"{prefix}{'stage':<{name_w}}  {'n':>6}  {'mean_s':>10}  "
            f"{'median_s':>10}  {'p95_s':>10}  {'total_s':>10}"
        )
        print(header)
        print(prefix + "-" * (len(header) - len(prefix)))
        for name, st in ordered:
            print(
                f"{prefix}{name:<{name_w}}  {int(st['count']):>6}  "
                f"{st['mean_s']:>10.4f}  {st['median_s']:>10.4f}  "
                f"{st['p95_s']:>10.4f}  {st['total_s']:>10.3f}"
            )

    def dump_json(self, path: str | Path) -> None:
        if not self.enabled:
            return
        rep = self.report()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(rep, f, indent=2, sort_keys=True)
