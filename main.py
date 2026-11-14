#!/usr/bin/env python3
"""Quantile forecaster for make.fun polymarket-case/v1 competitions.

Reads every <id>.in case file in the input directory and writes one
<id>.out forecast per case: for each market of the case and each of the
horizon minutes after its checkpoint, the five quantiles of that
minute's Yes price.

Model, per market, in logit space (so forecasts respect the (0, 1)
bounds and a one-tick move near 0.03 is not treated like one near 0.5):

  * Diffusion scale: a precision-shrunk RMS of one-minute logit returns
    over several lookback windows, blended with fixed weights and
    shrunk toward a prior when a window is short. Step-wise Polymarket
    prices sit still for minutes and then jump; the RMS over each
    window prices both the quiet minutes and the jumps.
  * Median path: the checkpoint logit, plus a heavily shrunk recent
    drift (capped over the whole horizon), plus a gentle pull toward
    the last day's mean logit price with a 15-hour half-life.
  * Spread at horizon h: sigma * sqrt(h + 0.09), the small constant
    widening the first minutes where a single tick already moves the
    price, with normal quantiles stretched by a smooth factor that
    grows with |z| so the 5% and 95% tails are fatter than Gaussian
    while the interquartile range stays close to it.
  * Shape: those parametric deviations are blended 20/80 with the
    empirical quantiles of the market's own multi-horizon logit
    returns. Polymarket prices sit still for long stretches, so the
    true move distribution is far more peaked than a Gaussian of the
    same variance; the empirical part carries that peak and the
    parametric part hedges jumps the short history has not shown.

Everything is a deterministic function of the case file: no clock, no
randomness, no network, standard library only. Two runs of the same
case give byte-identical output.
"""

import json
import math
import os
from statistics import NormalDist

# Prices are clamped to this band before the logit, so a market sitting
# on 0 or 1 still has finite coordinates.
P_LO = 0.0005
P_HI = 0.9995
# Written quantiles are clamped to this band instead of the bare [0, 1]:
# the actual price can sit on a boundary, never beyond it, and a hair of
# margin costs almost nothing under pinball loss.
OUT_LO = 0.001
OUT_HI = 0.999

# Volatility: prior scale (logit units per square-root minute), the
# number of pseudo-returns the prior is worth, and the lookback windows
# (minutes, weight) blended into sigma. The blend leans on the last
# hour and a half, where the current regime lives, but keeps a quarter
# of the weight on the 9-hour, 36-hour and full-history windows so a
# quiet recent stretch cannot shrink the spread to nothing.
PRIOR_SIGMA = 0.016
PRIOR_WEIGHT = 42.0
WINDOWS = ((90, 0.42), (540, 0.31), (2160, 0.17), (None, 0.10))
SIGMA_MIN = 0.011
SIGMA_MAX = 0.55

# Drift: the mean logit return of the last DRIFT_WINDOW minutes,
# shrunk twice (a flat factor, and n / (n + 240) for short samples) and
# capped so the whole horizon never drifts more than DRIFT_CAP_TOTAL.
DRIFT_WINDOW = 480
DRIFT_SHRINK = 0.30
DRIFT_CAP_TOTAL = 0.30

# Mean reversion of the median toward the mean logit of the last
# REV_WINDOW minutes: REV_WEIGHT of the gap, closed with a half-life of
# REV_HALFLIFE minutes.
REV_WINDOW = 1440
REV_WEIGHT = 0.55
REV_HALFLIFE = 900.0

# Spread shape: s(h) = sigma * sqrt(h + JUMP_PAD**2), and the normal
# quantile z is stretched to z * (1 + TAIL_A * z^2 / (TAIL_B + z^2)),
# which is 1.02x at the quartiles and 1.09x at the 5% / 95% levels.
JUMP_PAD = 0.30
TAIL_A = 0.15
TAIL_B = 2.0

# Empirical shape: the deviation of each quantile from the median is
# also read off the market's own history, as the empirical quantiles of
# its a-minute logit returns at anchor horizons a, interpolated in
# sqrt(h). Polymarket prices sit still for long stretches, so the true
# move distribution is far more peaked than any variance-matched
# Gaussian; the empirical deviations carry that peak, and the
# parametric ones (weight 1 - EMP_WEIGHT) hedge the jump risk a short
# history has not shown yet. Samples come from the last SAMPLE_MINUTES
# minutes; an anchor needs at least MIN_ANCHOR_SAMPLES returns.
ANCHORS = (1, 2, 4, 8, 15, 30, 60, 120, 240)
SAMPLE_MINUTES = 2880
MIN_ANCHOR_SAMPLES = 40
EMP_WEIGHT = 0.80

_NORM = NormalDist()


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _logit(p):
    p = _clamp(p, P_LO, P_HI)
    return math.log(p / (1.0 - p))


def _logistic(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _fat_z(tau):
    """The tau quantile with fat tails: a stretched standard normal.
    tau is clamped off the bare 0 and 1, where the inverse CDF is
    infinite, so any level a case could name stays finite."""
    z = _NORM.inv_cdf(_clamp(tau, 1e-6, 1.0 - 1e-6))
    return z * (1.0 + TAIL_A * z * z / (TAIL_B + z * z))


def _window_sigma(rets, width):
    """Shrunk RMS of the last `width` logit returns (all of them when
    width is None): the observed sum of squares pooled with the prior's
    pseudo-returns, so a short window speaks softly."""
    seg = rets if width is None or width >= len(rets) else rets[-width:]
    n = len(seg)
    if n == 0:
        return PRIOR_SIGMA
    ss = 0.0
    for r in seg:
        ss += r * r
    return math.sqrt((ss + PRIOR_WEIGHT * PRIOR_SIGMA * PRIOR_SIGMA) / (n + PRIOR_WEIGHT))


def _quantile(sorted_vals, tau):
    """The tau quantile of sorted values, linear on the (n - 1) index."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = lo + 1 if lo + 1 < n else n - 1
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _empirical_deviations(logits, taus):
    """Per anchor horizon, each quantile's deviation from the empirical
    median of that horizon's logit returns, in the market's own recent
    history. Deviations are forced to widen monotonically with the
    horizon: below the median they never rise as h grows, above it
    they never fall. Returns (anchors, devs) or None when the history
    is too short for even the smallest anchor."""
    n = len(logits)
    if n - 1 < MIN_ANCHOR_SAMPLES:
        return None
    start = max(0, n - SAMPLE_MINUTES)
    anchors = []
    devs = []
    med_idx = min(range(len(taus)), key=lambda i: abs(taus[i] - 0.5))
    for a in ANCHORS:
        if n - a < MIN_ANCHOR_SAMPLES:
            continue
        samples = sorted(logits[i + a] - logits[i] for i in range(start, n - a))
        if len(samples) < MIN_ANCHOR_SAMPLES:
            continue
        med = _quantile(samples, taus[med_idx])
        dev = [_quantile(samples, t) - med for t in taus]
        if devs:
            prev = devs[-1]
            dev = [
                min(d, p) if taus[k] < 0.5 else max(d, p) if taus[k] > 0.5 else 0.0
                for k, (d, p) in enumerate(zip(dev, prev))
            ]
        anchors.append(a)
        devs.append(dev)
    if not anchors:
        return None
    return anchors, devs


def _interp_dev(anchors, devs, h, k):
    """The k-th quantile's empirical deviation at horizon h, linear in
    sqrt(h) between anchors and sqrt-scaled outside them."""
    if h <= anchors[0]:
        return devs[0][k] * math.sqrt(h / anchors[0])
    if h >= anchors[-1]:
        return devs[-1][k] * math.sqrt(h / anchors[-1])
    for i in range(len(anchors) - 1):
        a0, a1 = anchors[i], anchors[i + 1]
        if a0 <= h <= a1:
            t = (math.sqrt(h) - math.sqrt(a0)) / (math.sqrt(a1) - math.sqrt(a0))
            return devs[i][k] * (1.0 - t) + devs[i + 1][k] * t
    return devs[-1][k]


def _forecast_market(prices, horizon, taus):
    """The horizon x len(taus) quantile rows for one market."""
    if not prices:
        prices = [0.5]
    logits = [_logit(p) for p in prices]
    last = logits[-1]
    rets = [b - a for a, b in zip(logits, logits[1:])]

    if rets:
        sigma = 0.0
        for width, weight in WINDOWS:
            sigma += weight * _window_sigma(rets, width)
    else:
        sigma = PRIOR_SIGMA
    sigma = _clamp(sigma, SIGMA_MIN, SIGMA_MAX)

    if rets:
        seg = rets[-DRIFT_WINDOW:]
        n = len(seg)
        drift = DRIFT_SHRINK * (sum(seg) / n) * (n / (n + 240.0))
        cap = DRIFT_CAP_TOTAL / horizon
        drift = _clamp(drift, -cap, cap)
    else:
        drift = 0.0

    tail = logits[-REV_WINDOW:]
    long_mean = sum(tail) / len(tail)
    gap = long_mean - last
    decay = math.log(2.0) / REV_HALFLIFE

    zs = [_fat_z(t) for t in taus]
    emp = _empirical_deviations(logits, taus)
    rows = []
    for h in range(1, horizon + 1):
        med = last + drift * h + REV_WEIGHT * (1.0 - math.exp(-decay * h)) * gap
        s = sigma * math.sqrt(h + JUMP_PAD * JUMP_PAD)
        row = []
        for k, z in enumerate(zs):
            dev = z * s
            if emp is not None:
                anchors, devs = emp
                dev = EMP_WEIGHT * _interp_dev(anchors, devs, h, k) + (1.0 - EMP_WEIGHT) * dev
            row.append(round(_clamp(_logistic(med + dev), OUT_LO, OUT_HI), 6))
        # The blend can cross neighbouring quantiles on odd samples;
        # the format requires each row non-decreasing.
        for k in range(1, len(row)):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append(row)
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case.get("horizon_minutes", 240))
        taus = [float(t) for t in case.get("quantiles", (0.05, 0.25, 0.5, 0.75, 0.95))]
        forecasts = [
            {"id": m["id"], "q": _forecast_market(m["prices"], horizon, taus)}
            for m in case["markets"]
        ]
        out_name = name[: -len(".in")] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")))


if __name__ == "__main__":
    main()
