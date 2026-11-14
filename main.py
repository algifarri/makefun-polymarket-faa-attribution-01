#!/usr/bin/env python3
"""Quantile forecaster for make.fun polymarket-case/v1 competitions.

For every <id>.in case in the input directory, writes one <id>.out:
per market and per minute of the 240 minutes after the checkpoint, the
five quantiles of that minute's Yes price.

Approach, market by market, entirely in logit coordinates (so a move
near 0.03 is not treated like a move near 0.5 and forecasts never leave
(0, 1)):

  * Scale. One-minute logit returns give a diffusion scale: the RMS
    return over several lookbacks, blended with fixed weights and
    pooled with a prior so short windows cannot collapse the spread.
    Polymarket prices sit still for minutes and then jump; the RMS
    prices both the quiet minutes and the jumps.
  * Centre. The median path starts at the checkpoint logit, adds a
    heavily shrunk recent drift (capped over the whole horizon) and a
    gentle pull toward the last day's mean logit, closed with a
    15-hour half-life.
  * Shape. The spread at horizon h grows like sigma * sqrt(h + 0.09)
    and the normal quantiles are stretched so the 5% / 95% tails are
    fatter than Gaussian while the quartiles stay close to it. That
    parametric spread is blended 20/80 with the market's own
    empirical multi-horizon return quantiles: because prices sit still
    for long stretches, the true move distribution is far more peaked
    than a variance-matched Gaussian, and the empirical part carries
    that peak; the parametric part hedges jumps the short history has
    not shown yet.

Deterministic: a pure function of the case file. No clock, no random
numbers, no network, standard library only, so two runs of the same
case produce byte-identical output.
"""

import json
import math
import os
from statistics import NormalDist

# Prices are pulled into this band before the logit, so a market
# sitting on 0 or 1 still has finite coordinates.
LOGIT_LO = 0.0005
LOGIT_HI = 0.9995
# Forecasts are written inside this band instead of the bare [0, 1]:
# the true price can sit on a boundary, never beyond it, and a hair of
# margin costs almost nothing under pinball loss.
OUT_LO = 0.001
OUT_HI = 0.999

# Diffusion scale: the prior scale (logit units per root-minute), how
# many pseudo-returns the prior is worth, and the lookbacks (minutes,
# weight) blended into sigma. The blend leans on the last hour and a
# half, where the current regime lives, but keeps a quarter of the
# weight on the 9-hour, 36-hour and full-history windows so a quiet
# recent stretch cannot shrink the spread to nothing.
VOL_PRIOR = 0.016
VOL_PRIOR_N = 42.0
VOL_LOOKBACKS = ((90, 0.42), (540, 0.31), (2160, 0.17), (None, 0.10))
VOL_MIN = 0.011
VOL_MAX = 0.55

# Centre drift: the mean logit return of the last DRIFT_SPAN minutes,
# shrunk twice (a flat gain, and n / (n + 240) for short samples) and
# capped so the whole horizon never drifts more than DRIFT_TOTAL_CAP.
DRIFT_SPAN = 480
DRIFT_GAIN = 0.30
DRIFT_TOTAL_CAP = 0.30

# Mean reversion of the median toward the mean logit of the last
# PULL_SPAN minutes: PULL_GAIN of the gap, closed with a half-life of
# PULL_HALF_LIFE minutes.
PULL_SPAN = 1440
PULL_GAIN = 0.55
PULL_HALF_LIFE = 900.0

# Spread shape: s(h) = sigma * sqrt(h + SPREAD_PAD ** 2), and the
# normal quantile z is stretched to z * (1 + TAIL_STRENGTH * z^2 /
# (TAIL_SCALE + z^2)): 1.02x at the quartiles, 1.09x at 5% / 95%.
SPREAD_PAD = 0.30
TAIL_STRENGTH = 0.15
TAIL_SCALE = 2.0

# Empirical shape: the deviation of each quantile from the median is
# also read off the market's own history, as the empirical quantiles of
# its a-minute logit returns at anchor horizons a, interpolated in
# sqrt(h). Samples come from the last HISTORY_SPAN minutes; an anchor
# needs at least MIN_RETURNS returns.
HISTORY_ANCHORS = (1, 2, 4, 8, 15, 30, 60, 120, 240)
HISTORY_SPAN = 2880
MIN_RETURNS = 40
HISTORY_WEIGHT = 0.80

_GAUSS = NormalDist()


def _bound(value, low, high):
    return low if value < low else high if value > high else value


def _to_logit(price):
    # A non-finite price (NaN/inf from a malformed case) is treated as
    # the neutral 0.5 instead of poisoning every downstream value:
    # _bound passes NaN through unchanged, and log(NaN) is NaN.
    if not math.isfinite(price):
        price = 0.5
    price = _bound(price, LOGIT_LO, LOGIT_HI)
    return math.log(price / (1.0 - price))


def _from_logit(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _stretched_z(level):
    """The level quantile of a fat-tailed standard normal: the plain
    normal quantile stretched by a smooth factor that grows with |z|.
    The level is kept off the bare 0 and 1, where the inverse CDF is
    infinite, so any level a case could name stays finite."""
    z = _GAUSS.inv_cdf(_bound(level, 1e-6, 1.0 - 1e-6))
    return z * (1.0 + TAIL_STRENGTH * z * z / (TAIL_SCALE + z * z))


def _rms_with_prior(returns, span):
    """Shrunk RMS of the last `span` logit returns (all of them when
    span is None): the observed sum of squares pooled with the prior's
    pseudo-returns, so a short window speaks softly."""
    window = returns if span is None or span >= len(returns) else returns[-span:]
    count = len(window)
    if count == 0:
        return VOL_PRIOR
    squares = 0.0
    for r in window:
        squares += r * r
    return math.sqrt(
        (squares + VOL_PRIOR_N * VOL_PRIOR * VOL_PRIOR) / (count + VOL_PRIOR_N)
    )


def _sample_quantile(ordered, level):
    """The level quantile of sorted values, linear on the (n - 1) index."""
    n = len(ordered)
    if n == 1:
        return ordered[0]
    position = level * (n - 1)
    low = int(math.floor(position))
    high = low + 1 if low + 1 < n else n - 1
    frac = position - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


class MarketForecaster:
    """One market's forecast: fitted once from its price history, then
    asked for each horizon minute in turn."""

    def __init__(self, prices, horizon, levels):
        if not prices:
            prices = [0.5]
        self.levels = levels
        self.horizon = horizon
        logits = [_to_logit(p) for p in prices]
        self.logits = logits
        self.checkpoint = logits[-1]
        self.returns = [b - a for a, b in zip(logits, logits[1:])]
        self.sigma = self._diffusion_scale()
        self.drift = self._drift_rate()
        self.anchor = self._mean_anchor()
        self.zs = [_stretched_z(t) for t in levels]
        self.empirical = self._empirical_shape()

    def _diffusion_scale(self):
        if not self.returns:
            return VOL_PRIOR
        blended = 0.0
        for span, weight in VOL_LOOKBACKS:
            blended += weight * _rms_with_prior(self.returns, span)
        return _bound(blended, VOL_MIN, VOL_MAX)

    def _drift_rate(self):
        """Shrunk per-minute drift, capped so the total drift over the
        horizon stays inside DRIFT_TOTAL_CAP logit units."""
        if not self.returns:
            return 0.0
        recent = self.returns[-DRIFT_SPAN:]
        n = len(recent)
        rate = DRIFT_GAIN * (sum(recent) / n) * (n / (n + 240.0))
        return _bound(rate, -DRIFT_TOTAL_CAP / self.horizon,
                      DRIFT_TOTAL_CAP / self.horizon)

    def _mean_anchor(self):
        """(long-run mean logit, gap from the checkpoint, decay rate):
        the median is pulled toward the mean of the last PULL_SPAN
        minutes, closing PULL_GAIN of the gap with the given half-life."""
        tail = self.logits[-PULL_SPAN:]
        mean = sum(tail) / len(tail)
        return mean, mean - self.checkpoint, math.log(2.0) / PULL_HALF_LIFE

    def _empirical_shape(self):
        """Per anchor horizon, each quantile's deviation from the
        empirical median of that horizon's logit returns, in the
        market's own recent history. Deviations are forced to widen
        monotonically with the horizon: below the median they never
        rise as h grows, above it they never fall. Returns
        (anchors, deviations) or None when the history is too short
        for even the smallest anchor."""
        logits = self.logits
        n = len(logits)
        if n - 1 < MIN_RETURNS:
            return None
        start = max(0, n - HISTORY_SPAN)
        median_pos = min(range(len(self.levels)),
                         key=lambda i: abs(self.levels[i] - 0.5))
        anchors = []
        deviations = []
        for a in HISTORY_ANCHORS:
            if n - a < MIN_RETURNS:
                continue
            samples = sorted(
                logits[i + a] - logits[i] for i in range(start, n - a)
            )
            if len(samples) < MIN_RETURNS:
                continue
            centre = _sample_quantile(samples, self.levels[median_pos])
            dev = [_sample_quantile(samples, t) - centre for t in self.levels]
            if deviations:
                previous = deviations[-1]
                dev = [
                    min(d, p) if self.levels[k] < 0.5
                    else max(d, p) if self.levels[k] > 0.5
                    else 0.0
                    for k, (d, p) in enumerate(zip(dev, previous))
                ]
            anchors.append(a)
            deviations.append(dev)
        if not anchors:
            return None
        return anchors, deviations

    def _empirical_at(self, h, k):
        """The k-th quantile's empirical deviation at horizon h, linear
        in sqrt(h) between anchors and sqrt-scaled outside them."""
        anchors, deviations = self.empirical
        if h <= anchors[0]:
            return deviations[0][k] * math.sqrt(h / anchors[0])
        if h >= anchors[-1]:
            return deviations[-1][k] * math.sqrt(h / anchors[-1])
        for i in range(len(anchors) - 1):
            a0, a1 = anchors[i], anchors[i + 1]
            if a0 <= h <= a1:
                t = (math.sqrt(h) - math.sqrt(a0)) / (
                    math.sqrt(a1) - math.sqrt(a0)
                )
                return deviations[i][k] * (1.0 - t) + deviations[i + 1][k] * t
        return deviations[-1][k]

    def median_at(self, h):
        """The median logit h minutes after the checkpoint."""
        mean, gap, decay = self.anchor
        return (
            self.checkpoint
            + self.drift * h
            + PULL_GAIN * (1.0 - math.exp(-decay * h)) * gap
        )

    def rows(self):
        """The horizon x len(levels) quantile rows for this market."""
        out = []
        for h in range(1, self.horizon + 1):
            centre = self.median_at(h)
            width = self.sigma * math.sqrt(h + SPREAD_PAD * SPREAD_PAD)
            row = []
            for k, z in enumerate(self.zs):
                dev = z * width
                if self.empirical is not None:
                    dev = (
                        HISTORY_WEIGHT * self._empirical_at(h, k)
                        + (1.0 - HISTORY_WEIGHT) * dev
                    )
                row.append(round(_bound(_from_logit(centre + dev),
                                       OUT_LO, OUT_HI), 6))
            # The blend can cross neighbouring quantiles on odd samples;
            # the format requires each row non-decreasing.
            for k in range(1, len(row)):
                if row[k] < row[k - 1]:
                    row[k] = row[k - 1]
            out.append(row)
        return out


def forecast_case(case):
    horizon = int(case.get("horizon_minutes", 240))
    levels = [float(t) for t in case.get(
        "quantiles", (0.05, 0.25, 0.5, 0.75, 0.95))]
    return [
        {"id": m["id"],
         "q": MarketForecaster(m["prices"], horizon, levels).rows()}
        for m in case["markets"]
    ]


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        forecasts = forecast_case(case)
        out_name = name[: -len(".in")] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")))


if __name__ == "__main__":
    main()
