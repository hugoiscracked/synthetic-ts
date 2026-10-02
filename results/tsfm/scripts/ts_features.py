"""Statistical descriptors for real and synthetic time series.

These are the four families of statistic referenced in the revision:
seasonality strength, change-point rate, autocorrelation decay, and tail
weight.  They serve two purposes:

  1. characterising what the generator's difficulty knob ``d`` actually does
     to the produced signal (answering "how is difficulty established
     objectively?"), and
  2. profiling real datasets and synthetic bundles on a common axis so that
     bundle choice can be conditioned on a measurable target profile rather
     than on intuition.

Everything is implemented on numpy/scipy alone so the same code runs on the
cluster without extra dependencies.  Each function takes a 1-D array and
returns a scalar; NaN is returned when a series is too short or degenerate.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats

__all__ = [
    "spectral_entropy",
    "seasonality_strength",
    "trend_strength",
    "hurst_exponent",
    "changepoint_rate",
    "acf_decay",
    "hill_tail_index",
    "excess_kurtosis",
    "FEATURE_FUNCS",
    "describe_series",
    "describe_matrix",
]

_EPS = 1e-12
DEFAULT_PERIODS = (24, 48, 96, 168, 336)


def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _standardise(x: np.ndarray) -> np.ndarray:
    s = x.std()
    if s < _EPS:
        return np.zeros_like(x)
    return (x - x.mean()) / s


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------

def spectral_entropy(x: np.ndarray) -> float:
    """Shannon entropy of the normalised power spectral density, in [0, 1].

    0 means all power sits at one frequency (a pure sinusoid); 1 means a flat
    spectrum (white noise).  This is the cleanest single summary of how
    "predictable" the periodic structure of a series is.
    """
    x = _clean(x)
    if x.size < 16 or x.std() < _EPS:
        return np.nan
    nperseg = min(256, x.size)
    _, psd = sp_signal.welch(_standardise(x), nperseg=nperseg)
    psd = psd[psd > 0]
    if psd.size < 2:
        return np.nan
    p = psd / psd.sum()
    return float(-(p * np.log(p)).sum() / np.log(p.size))


# ---------------------------------------------------------------------------
# Seasonality and trend (classical additive decomposition)
# ---------------------------------------------------------------------------

def _decompose(x: np.ndarray, period: int):
    """Classical additive decomposition into trend, seasonal, remainder."""
    if x.size < 2 * period:
        return None
    # Centred moving average of one full period as the trend estimate.
    kernel = np.ones(period) / period
    trend = np.convolve(x, kernel, mode="same")
    # Edges of a 'same' convolution are biased; drop half a period each side.
    half = period // 2
    sl = slice(half, x.size - half)
    if sl.stop - sl.start < period:
        return None
    detrended = x[sl] - trend[sl]
    # Seasonal profile = mean of the detrended series by phase.
    phase = np.arange(sl.start, sl.stop) % period
    profile = np.array([detrended[phase == p].mean() if np.any(phase == p) else 0.0
                        for p in range(period)])
    profile -= profile.mean()
    seasonal = profile[phase]
    remainder = detrended - seasonal
    return trend[sl], seasonal, remainder


def seasonality_strength(x: np.ndarray, periods=DEFAULT_PERIODS) -> float:
    """Strength of the strongest seasonal component, in [0, 1].

    Follows the Hyndman feature ``1 - Var(remainder) / Var(seasonal +
    remainder)`` but uses a classical decomposition rather than STL so that no
    extra dependency is needed.  The maximum over candidate periods is taken,
    since the dominant period is not known a priori.
    """
    x = _clean(x)
    if x.size < 32 or x.std() < _EPS:
        return np.nan
    best = 0.0
    for period in periods:
        parts = _decompose(x, int(period))
        if parts is None:
            continue
        _, seasonal, remainder = parts
        denom = np.var(seasonal + remainder)
        if denom < _EPS:
            continue
        best = max(best, 1.0 - np.var(remainder) / denom)
    return float(np.clip(best, 0.0, 1.0))


def trend_strength(x: np.ndarray, period: int = 96) -> float:
    """Strength of the trend component, in [0, 1]."""
    x = _clean(x)
    if x.size < 32 or x.std() < _EPS:
        return np.nan
    parts = _decompose(x, int(period))
    if parts is None:
        return np.nan
    trend, _, remainder = parts
    detrended_var = np.var(trend - trend.mean() + remainder)
    if detrended_var < _EPS:
        return np.nan
    return float(np.clip(1.0 - np.var(remainder) / detrended_var, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Long-range dependence
# ---------------------------------------------------------------------------

def hurst_exponent(x: np.ndarray) -> float:
    """Hurst exponent via detrended fluctuation analysis.

    H ~ 0.5 indicates a random walk / no memory, H > 0.5 persistent long
    memory, H < 0.5 mean reversion.  DFA is used rather than R/S because it
    tolerates non-stationary trends, which several bundles deliberately have.
    """
    x = _clean(x)
    if x.size < 64 or x.std() < _EPS:
        return np.nan
    y = np.cumsum(_standardise(x))
    n = y.size
    scales = np.unique(np.floor(np.logspace(np.log10(8), np.log10(n // 4), 12)).astype(int))
    scales = scales[scales >= 8]
    if scales.size < 4:
        return np.nan

    fluct = []
    for s in scales:
        n_seg = n // s
        if n_seg < 1:
            continue
        segs = y[: n_seg * s].reshape(n_seg, s)
        t = np.arange(s)
        # Remove a linear trend from each segment, then take the RMS residual.
        coeffs = np.polyfit(t, segs.T, 1)
        fitted = np.outer(coeffs[0], t) + coeffs[1][:, None]
        resid = segs - fitted
        fluct.append(np.sqrt((resid ** 2).mean()))

    fluct = np.asarray(fluct)
    ok = fluct > _EPS
    if ok.sum() < 4:
        return np.nan
    slope = np.polyfit(np.log(scales[: fluct.size][ok]), np.log(fluct[ok]), 1)[0]
    return float(slope)


def acf_decay(x: np.ndarray, max_lag: int = 100) -> float:
    """Lag at which the autocorrelation first falls below 1/e.

    Larger values mean slower-decaying autocorrelation, i.e. longer memory.
    Returns ``max_lag`` if the ACF never drops below the threshold.
    """
    x = _clean(x)
    if x.size < 32 or x.std() < _EPS:
        return np.nan
    z = _standardise(x)
    max_lag = int(min(max_lag, x.size // 4))
    if max_lag < 2:
        return np.nan
    full = np.correlate(z, z, mode="full")[z.size - 1:]
    acf = full[: max_lag + 1] / (full[0] + _EPS)
    thresh = 1.0 / np.e
    below = np.where(acf < thresh)[0]
    if below.size == 0:
        return float(max_lag)
    k = int(below[0])
    if k == 0:
        return 0.0
    # Linear interpolation between the bracketing lags.
    a0, a1 = acf[k - 1], acf[k]
    if abs(a0 - a1) < _EPS:
        return float(k)
    return float(k - 1 + (a0 - thresh) / (a0 - a1))


# ---------------------------------------------------------------------------
# Change points
# ---------------------------------------------------------------------------

def _binary_segment(z: np.ndarray, start: int, stop: int, penalty: float, out: list,
                    min_size: int = 20, depth: int = 0, max_depth: int = 8) -> None:
    """Recursive binary segmentation on shifts in the mean."""
    n = stop - start
    if n < 2 * min_size or depth >= max_depth:
        return
    seg = z[start:stop]
    csum = np.cumsum(seg)
    total = csum[-1]
    k = np.arange(min_size, n - min_size)
    if k.size == 0:
        return
    left = csum[k - 1]
    # Standard CUSUM statistic for a single mean shift.
    stat = np.abs(left / k - (total - left) / (n - k)) * np.sqrt(k * (n - k) / n)
    j = int(np.argmax(stat))
    if stat[j] > penalty:
        cut = start + int(k[j])
        out.append(cut)
        _binary_segment(z, start, cut, penalty, out, min_size, depth + 1, max_depth)
        _binary_segment(z, cut, stop, penalty, out, min_size, depth + 1, max_depth)


def changepoint_rate(x: np.ndarray, penalty: float = 3.0, per: int = 1000) -> float:
    """Detected mean-shift change points per ``per`` observations.

    Binary segmentation with a CUSUM statistic and a fixed threshold on the
    standardised series.  The absolute count depends on the penalty, so this is
    meaningful as a *relative* measure across series of equal length -- which
    is how it is used throughout.
    """
    x = _clean(x)
    if x.size < 64 or x.std() < _EPS:
        return np.nan
    z = _standardise(x)
    cuts: list[int] = []
    _binary_segment(z, 0, z.size, penalty, cuts)
    return float(len(cuts) * per / z.size)


# ---------------------------------------------------------------------------
# Tail weight
# ---------------------------------------------------------------------------

def excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis of the first differences (0 for a Gaussian)."""
    x = _clean(x)
    if x.size < 32:
        return np.nan
    d = np.diff(x)
    if d.size < 8 or d.std() < _EPS:
        return np.nan
    return float(sp_stats.kurtosis(d, fisher=True, bias=False))


def hill_tail_index(x: np.ndarray, tail_frac: float = 0.05) -> float:
    """Hill estimator of the tail index of |first differences|.

    Smaller values mean heavier tails.  Computed on differences so that the
    statistic reflects shock magnitude rather than level, which makes it
    comparable between stationary and drifting series.
    """
    x = _clean(x)
    if x.size < 64:
        return np.nan
    d = np.abs(np.diff(x))
    d = d[d > _EPS]
    if d.size < 32:
        return np.nan
    k = max(8, int(tail_frac * d.size))
    k = min(k, d.size - 1)
    top = np.sort(d)[-k:]
    thresh = top[0]
    if thresh <= _EPS:
        return np.nan
    logs = np.log(top[1:] / thresh)
    if logs.size == 0 or logs.mean() <= _EPS:
        return np.nan
    return float(1.0 / logs.mean())


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------

FEATURE_FUNCS = {
    "seasonality_strength": seasonality_strength,
    "trend_strength": trend_strength,
    "spectral_entropy": spectral_entropy,
    "changepoint_rate": changepoint_rate,
    "acf_decay": acf_decay,
    "hurst": hurst_exponent,
    "excess_kurtosis": excess_kurtosis,
    "tail_index": hill_tail_index,
}

# The four axes used for bundle/target matching. Deliberately a subset of the
# above: these are the ones that discriminate between bundles.
MATCH_FEATURES = ("seasonality_strength", "changepoint_rate", "acf_decay", "tail_index")


def describe_series(x: np.ndarray) -> dict:
    """All features for one univariate series."""
    return {name: fn(x) for name, fn in FEATURE_FUNCS.items()}


def describe_matrix(X: np.ndarray, max_channels: int | None = None) -> dict:
    """Channel-averaged features for a (T, D) matrix.

    Features are computed per channel and then averaged, so that a wide dataset
    is not dominated by whichever channel happens to be first.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n_ch = X.shape[1] if max_channels is None else min(X.shape[1], max_channels)
    rows = [describe_series(X[:, j]) for j in range(n_ch)]
    out = {}
    for name in FEATURE_FUNCS:
        vals = np.array([r[name] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[name] = float(vals.mean()) if vals.size else np.nan
    return out
