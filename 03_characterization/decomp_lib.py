"""Baseline / excursion decomposition of a glucose series.

On a regular 5-minute grid the slow baseline is a centred rolling median (default 3 h =
36 samples, min_periods=18); the excursion component is the residual glucose - baseline.
"""
import pandas as pd

WINDOW_5MIN = 36   # 3 hours at 5-min cadence


def to_grid(series_with_dtindex, freq="5min"):
    """Resample an irregular datetime-indexed glucose series to a regular grid."""
    return series_with_dtindex.resample(freq).mean()


def baseline_excursion(g_grid, window=WINDOW_5MIN):
    """Return (baseline, excursion) for a 5-min-gridded glucose Series.
    baseline = centered rolling median; excursion = g - baseline."""
    baseline = g_grid.rolling(window=window, center=True,
                              min_periods=max(3, window // 2)).median()
    excursion = g_grid - baseline
    return baseline, excursion
