"""Paired statistics for per-prompt binary outcomes (e.g. harmful yes/no).

All comparisons are paired: the same prompts are run through bf16 and a
quantized model, so the test is McNemar's exact test on the discordant pairs.
"""

from __future__ import annotations

from math import comb


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = only-A positive, c = only-B positive."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * _binom_cdf(min(b, c), n))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI for a rate k/n, by bisection on the binomial CDF."""
    if n == 0:
        return (0.0, 1.0)

    def solve(f, target, increasing):
        """Find p in [0, 1] with f(p) = target for a monotone f."""
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # lower: P(X >= k | p) = alpha/2 (increasing in p); upper: P(X <= k | p) = alpha/2 (decreasing)
    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p), alpha / 2, True)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p), alpha / 2, False)
    return (lower, upper)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[key] = running
    return out


def paired(a: dict, b: dict) -> dict:
    """Compare two {prompt_id: bool} outcome maps on their shared prompts."""
    ids = sorted(set(a) & set(b))
    n = len(ids)
    ka, kb = sum(a[i] for i in ids), sum(b[i] for i in ids)
    only_a = sum(a[i] and not b[i] for i in ids)
    only_b = sum(b[i] and not a[i] for i in ids)
    return {
        "n": n,
        "rate_a": ka / n, "ci_a": clopper_pearson(ka, n),
        "rate_b": kb / n, "ci_b": clopper_pearson(kb, n),
        "diff": (kb - ka) / n,
        "only_a": only_a, "only_b": only_b,
        "p_mcnemar": mcnemar_exact(only_a, only_b),
    }
