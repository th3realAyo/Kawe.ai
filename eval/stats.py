"""Small statistics helpers: bootstrap intervals and paired tests."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def mean_ci(values: Sequence[float], n_boot: int = 10_000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean with a percentile-bootstrap 95% interval."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def clustered_rate_ci(groups: Dict[str, List[bool]], n_boot: int = 10_000, seed: int = 0) -> Tuple[float, float, float]:
    """Pooled rate over all items, with a 95% interval from resampling whole groups (papers)."""
    keys = [k for k, v in groups.items() if v]
    if not keys:
        return math.nan, math.nan, math.nan
    hits = np.array([sum(groups[k]) for k in keys], dtype=float)
    sizes = np.array([len(groups[k]) for k in keys], dtype=float)
    rate = hits.sum() / sizes.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot = hits[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return float(rate), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired comparison a - b: mean difference with bootstrap CI, wins, and the Wilcoxon signed-rank p-value."""
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    mean, lo, hi = mean_ci(diff)
    try:
        p = float(wilcoxon(a, b).pvalue) if np.any(diff != 0) else 1.0
    except ValueError:
        p = math.nan
    return {"mean_diff": mean, "ci_low": lo, "ci_high": hi, "wins": int((diff > 0).sum()), "n": int(len(diff)), "p": p}


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"
