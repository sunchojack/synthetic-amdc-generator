"""Sampling primitives fitted to published moments and quantiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ANCHOR_Q = (0.25, 0.50, 0.75, 0.90)
_ANCHOR_KEYS = ("m_p25", "m_p50", "m_p75", "m_p90")


@dataclass
class HeavyTail:
    """Quantile-interpolated body below p90, truncated Pareto tail above.

    The tail is truncated, not open-ended. An untruncated Pareto with the index the
    published mean demands sits near alpha = 1, where the sample mean of even 10^8
    draws stays far below the theoretical mean, so the generated file would miss the
    target it was fitted to. Truncating at the value expected to be exceeded once in
    a file of `n_rows` gives a stable mean and a plausible largest observation, and
    makes the largest generated value shrink with SCALE the way a smaller file's
    maximum genuinely would.
    """

    q: np.ndarray          # anchor probabilities, ascending
    x: np.ndarray          # anchor values, ascending and strictly positive
    alpha: float           # Pareto index of the tail above the last anchor
    x_max: float           # truncation point of the tail
    zero_share: float      # probability mass placed exactly at zero
    target_mean: float

    # ---------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, numeric: dict[str, float], zero_share: float = 0.0,
            n_rows: int | None = None, floor: float = 1e-3) -> "HeavyTail":
        mean = float(numeric.get("mean") or 0.0)
        pairs = [(p, float(numeric[k])) for p, k in zip(_ANCHOR_Q, _ANCHOR_KEYS)
                 if k in numeric and numeric[k] is not None]

        # rebase the published unconditional quantiles onto the non-zero part
        z = min(max(zero_share, 0.0), 0.999)
        cond: list[tuple[float, float]] = []
        for p, v in pairs:
            if v <= 0 or p <= z:
                continue
            cond.append(((p - z) / (1.0 - z), v))

        if not cond:
            # everything the publication tells us is zero at these quantiles;
            # fall back to a single anchor implied by the mean
            m_cond = mean / (1.0 - z) if z < 1 else 0.0
            cond = [(0.5, max(m_cond, floor))]

        # enforce strict monotonicity
        qs, xs = [], []
        for p, v in sorted(cond):
            if xs and v <= xs[-1]:
                v = xs[-1] * 1.000001
            qs.append(p)
            xs.append(max(v, floor))

        q = np.asarray(qs, dtype=float)
        x = np.asarray(xs, dtype=float)
        target_cond_mean = mean / (1.0 - z) if z < 1 else 0.0
        n_tail = max((n_rows or 10 ** 8) * (1.0 - z) * (1.0 - q[-1]), 10.0)
        alpha, x_max = _solve_alpha(q, x, target_cond_mean, n_tail)
        return cls(q=q, x=x, alpha=alpha, x_max=x_max, zero_share=z, target_mean=mean)

    # --------------------------------------------------------------- sampling
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros(n, dtype=float)
        if self.zero_share >= 1.0:
            return out
        nz = rng.random(n) >= self.zero_share
        u = rng.random(int(nz.sum()))
        out[nz] = self._icdf(u)
        return out

    def sample_positive(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draws from the non-zero part only, ignoring the zero mass.

        Use when the zero rows are decided by a structural rule (a code that declares
        no unit, a variable only collected in one direction) rather than by the fit.
        """
        return self._icdf(rng.random(n))

    def _icdf(self, u: np.ndarray) -> np.ndarray:
        q, x = self.q, self.x
        out = np.empty_like(u)

        # body: log-linear interpolation between anchors
        lo_q, hi_q = q[0], q[-1]
        body = u < hi_q
        ub = u[body]
        # below the first anchor, extend with the slope of the first segment
        logx = np.log(x)
        if len(q) >= 2:
            slope0 = (logx[1] - logx[0]) / (q[1] - q[0])
        else:
            slope0 = np.log(10.0) / 0.25
        below = ub < lo_q
        res = np.empty_like(ub)
        res[below] = np.exp(logx[0] + slope0 * (ub[below] - lo_q))
        inner = ~below
        if len(q) >= 2:
            res[inner] = np.exp(np.interp(ub[inner], q, logx))
        else:
            res[inner] = x[0]
        out[body] = res

        # tail: Pareto truncated at x_max
        tail = ~body
        ut = (u[tail] - hi_q) / (1.0 - hi_q)
        ratio = self.x_max / x[-1]
        cut = 1.0 - ratio ** (-self.alpha)
        out[tail] = x[-1] * (1.0 - ut * cut) ** (-1.0 / self.alpha)
        return out


def _body_mean(q: np.ndarray, x: np.ndarray, grid: int = 4096) -> float:
    """E[X * 1{U < q_last}] under log-linear quantile interpolation."""
    hi = q[-1]
    u = (np.arange(grid) + 0.5) / grid * hi
    logx = np.log(x)
    slope0 = (logx[1] - logx[0]) / (q[1] - q[0]) if len(q) >= 2 else np.log(10.0) / 0.25
    vals = np.where(u < q[0],
                    np.exp(logx[0] + slope0 * (u - q[0])),
                    np.exp(np.interp(u, q, logx)))
    return float(vals.mean() * hi)


def _trunc_pareto_mean(xm: float, x_max: float, a: float) -> float:
    """Mean of a Pareto(xm, a) conditioned on X <= x_max."""
    r = x_max / xm
    denom = 1.0 - r ** (-a)
    if denom <= 0:
        return xm
    if abs(a - 1.0) < 1e-9:
        return xm * np.log(r) / denom
    return xm * a / (a - 1.0) * (1.0 - r ** (1.0 - a)) / denom


def _solve_alpha(q: np.ndarray, x: np.ndarray, target_mean: float, n_tail: float,
                 lo: float = 0.35, hi: float = 60.0) -> tuple[float, float]:
    """Tail index and truncation point reproducing the published mean.

    The truncation point is the value expected to be exceeded once in a file of
    `n_tail` tail observations, so the tail's largest value is the largest value the
    real file would plausibly contain rather than an artefact of an unbounded draw.
    """
    hi_q = q[-1]
    xm = x[-1]
    body = _body_mean(q, x)
    mass = 1.0 - hi_q

    def x_max_of(a: float) -> float:
        return xm * n_tail ** (1.0 / a)

    def total(a: float) -> float:
        return body + mass * _trunc_pareto_mean(xm, x_max_of(a), a)

    if target_mean <= 0:
        return hi, x_max_of(hi)
    if total(hi) > target_mean:
        return hi, x_max_of(hi)      # quantiles alone already exceed the mean
    if total(lo) < target_mean:
        return lo, x_max_of(lo)      # tail heavier than this family can express
    a, b = lo, hi
    for _ in range(90):
        m = 0.5 * (a + b)
        if total(m) > target_mean:
            a = m
        else:
            b = m
    a = 0.5 * (a + b)
    return a, x_max_of(a)


# --------------------------------------------------------------- categoricals

def zipf_weights(n: int, s: float = 1.15) -> np.ndarray:
    """Rank-frequency weights for code universes AMDC gives no level counts for."""
    r = np.arange(1, n + 1, dtype=float)
    w = r ** (-s)
    return w / w.sum()


def weights_from_levels(levels: dict[str, int], keys: list[str]) -> np.ndarray:
    """Normalised weights for `keys` taken from published level counts."""
    raw = np.array([float(levels.get(k, 0)) for k in keys])
    if raw.sum() <= 0:
        raw = np.ones(len(keys))
    return raw / raw.sum()


def allocate(total: int, weights: np.ndarray) -> np.ndarray:
    """Split `total` into integer parts proportional to weights, exactly summing."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    base = np.floor(w * total).astype(np.int64)
    short = total - int(base.sum())
    if short > 0:
        order = np.argsort(-(w * total - base))
        base[order[:short]] += 1
    return base


def weights_matching_numeric_quantiles(values: np.ndarray, numeric: dict[str, float]) -> np.ndarray:
    """Distribution over a code universe reproducing published numeric quantiles.

    AMDC treats codes such as OENACE, NUTS-3 and the municipality number as integers
    and publishes their mean and quantiles. Those quantiles carry the real
    composition of the population: the 25th percentile of the OENACE code says how
    much weight sits in the low-numbered sections. Rather than inventing sector
    shares, this reads the composition back out of the published quantiles by
    interpolating the implied CDF onto the code axis.
    """
    v = np.sort(np.asarray(values, dtype=float))
    anchors: list[tuple[float, float]] = []
    for p, key in ((0.0, "m_min"), *zip(_ANCHOR_Q, _ANCHOR_KEYS), (1.0, "m_max")):
        if key in numeric and numeric[key] is not None:
            anchors.append((p, float(numeric[key])))
    if len(anchors) < 2:
        return np.full(len(v), 1.0 / len(v))
    qs = np.array([a[0] for a in anchors])
    xs = np.array([a[1] for a in anchors])
    order = np.argsort(xs)
    xs, qs = xs[order], qs[order]
    # strictly increasing value axis so np.interp is well defined
    for i in range(1, len(xs)):
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-6

    edges = np.empty(len(v) + 1)
    edges[1:-1] = 0.5 * (v[:-1] + v[1:])
    edges[0] = v[0] - (edges[1] - v[0] if len(v) > 1 else 1.0)
    edges[-1] = v[-1] + (v[-1] - edges[-2] if len(v) > 1 else 1.0)
    cdf = np.interp(edges, xs, qs)
    w = np.diff(cdf)
    w = np.maximum(w, 1e-9)
    w = w / w.sum()
    # return in the caller's original order
    back = np.empty(len(v))
    back[np.argsort(np.asarray(values, dtype=float), kind="stable")] = w
    return back


def rebalance_group_mass(weights: np.ndarray, in_group: np.ndarray,
                         target_mass: float) -> np.ndarray:
    """Rescale so the flagged codes carry exactly `target_mass` of the probability.

    Used where a published missing rate is a property of the classification rather
    than of reporting: the share of rows on codes that declare a special unit, or a
    second quantity unit, has to equal the published non-missing rate.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    m = np.asarray(in_group, dtype=bool)
    inside, outside = w[m].sum(), w[~m].sum()
    if inside <= 0 or outside <= 0:
        return w
    t = min(max(target_mass, 1e-9), 1 - 1e-9)
    out = w.copy()
    out[m] = w[m] * (t / inside)
    out[~m] = w[~m] * ((1 - t) / outside)
    return out / out.sum()


def tilt_to_mean(values: np.ndarray, weights: np.ndarray, target_mean: float) -> np.ndarray:
    """Exponentially tilt weights until the weighted mean hits the target.

    Maximum-entropy reweighting: the smallest change to `weights`, in relative
    entropy, that satisfies the one published moment. Used where the published
    quantiles are smoothed onto values no real code takes (AMDC reports a median
    NUTS-3 code of 148.5, which is not a NUTS-3 region) so the mean is the only
    moment that can be matched exactly.
    """
    v = np.asarray(values, dtype=float)
    w0 = np.asarray(weights, dtype=float)
    w0 = w0 / w0.sum()
    if not np.isfinite(target_mean) or v.min() >= target_mean or v.max() <= target_mean:
        return w0
    c = v - target_mean
    scale = max(np.abs(c).max(), 1e-9)
    c = c / scale

    def mean_at(lam: float) -> float:
        z = np.exp(np.clip(lam * c, -500, 500)) * w0
        return float((c * z).sum() / z.sum())

    lo, hi = -50.0, 50.0
    if mean_at(lo) > 0 or mean_at(hi) < 0:
        return w0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < 0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    w = np.exp(np.clip(lam * c, -500, 500)) * w0
    return w / w.sum()


def ensure_all_levels(drawn: np.ndarray, levels, rng: np.random.Generator) -> np.ndarray:
    """Force every level to appear at least once, where the row count allows it.

    AMDC publishes a distinct-value count per year. Rare levels would otherwise go
    missing to sampling noise and the generated file would understate that count.
    """
    levels = list(levels)
    if len(levels) > len(drawn):
        return drawn
    missing = [lv for lv in levels if lv not in set(drawn.tolist())]
    if not missing:
        return drawn
    slots = rng.choice(len(drawn), size=len(missing), replace=False)
    drawn = drawn.copy()
    drawn[slots] = np.asarray(missing, dtype=drawn.dtype)
    return drawn


def ensure_distinct_count(drawn: np.ndarray, universe: np.ndarray, target: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Raise the distinct-value count of a draw to a published target.

    AMDC publishes how many distinct CN8 and OEPRODCOM codes appear each year, around
    8,900 and 3,200. At a reduced SCALE a concentrated draw cannot reach that on its
    own, and a pipeline tested on 4,000 codes has not been tested on the real breadth
    of the code space. Unseen codes are therefore injected into a small share of rows.
    The cost is a slightly flatter frequency tail, which the calibration report states.
    """
    if target <= 0:
        return drawn
    seen = set(np.unique(drawn).tolist())
    if len(seen) >= target:
        return drawn
    unseen = np.array([c for c in universe.tolist() if c not in seen])
    if unseen.size == 0:
        return drawn
    need = min(target - len(seen), unseen.size, drawn.shape[0])
    add = rng.permutation(unseen)[:need]
    slots = rng.choice(drawn.shape[0], size=need, replace=False)
    out = drawn.copy()
    out[slots] = add.astype(drawn.dtype)
    return out


def rank_couple(driver: np.ndarray, target: np.ndarray, rho: float,
                rng: np.random.Generator) -> np.ndarray:
    """Reorder `target` so it correlates with `driver` without changing its values.

    Marginal calibration and joint realism pull against each other: AMDC publishes a
    separate summary per variable and nothing about how they move together, so drawing
    each variable from its own fitted marginal gives exactly the published quantiles
    and a correlation of zero. In the real trade file weight and value move together,
    which is what makes a unit value mean anything.

    Permuting rather than redrawing resolves it. The multiset of `target` values is
    untouched, so every published quantile and the mean survive exactly, while the
    pairing with `driver` reaches roughly the requested rank correlation.
    """
    n = driver.shape[0]
    if n < 2:
        return target
    rho = float(np.clip(rho, -1.0, 1.0))
    dr = np.argsort(np.argsort(driver)) / max(n - 1, 1)
    blended = rho * dr + np.sqrt(max(1.0 - rho * rho, 0.0)) * rng.random(n)
    out = np.empty_like(target)
    out[np.argsort(blended)] = np.sort(target)
    return out


def firm_baskets(rng: np.random.Generator, counts: np.ndarray, codes: np.ndarray,
                 weights: np.ndarray, coef: float = 2.0, expo: float = 0.45,
                 concentration: float = 0.9) -> np.ndarray:
    """Draw each unit's rows from its own persistent basket of codes.

    A real trader ships the same handful of commodity codes month after month; drawing
    every row from the economy-wide code distribution instead gives a firm 150 distinct
    codes across 276 shipments and hands almost every firm the same top code. Firm-level
    concentration is what any product-portfolio or supplier-dependency measure reads, so
    the basket has to exist.

    Basket size grows sublinearly with the unit's row count, after Heaps' law, so large
    traders carry broad ranges and small ones carry a few lines.
    """
    total = int(counts.sum())
    out = np.empty(total, dtype=codes.dtype)
    at = 0
    for count in counts:
        count = int(count)
        if count <= 0:
            continue
        size = int(np.clip(round(coef * count ** expo), 1, min(len(codes), count * 4)))
        basket = rng.choice(len(codes), size=size, replace=False, p=weights)
        bw = zipf_weights(size, s=concentration)
        out[at:at + count] = codes[basket[rng.choice(size, size=count, p=bw)]]
        at += count
    return out


def lognormal_with_mean(mean: float, cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Lognormal draws with the requested arithmetic mean and coefficient of variation."""
    sigma2 = np.log(1.0 + cv * cv)
    mu = np.log(max(mean, 1e-9)) - 0.5 * sigma2
    return rng.lognormal(mu, np.sqrt(sigma2), n)
