"""Drift detector: catches slow, steady change (like a leak) that never looks odd in one step.

Idea: a trailing mean over exactly one day removes the daily wave. Then ask how much that
daily mean moved compared with one day ago. A flat series moves ~0; a leak keeps moving.
It never reads ground truth and knows no business words.

Assumes evenly spaced points (window and lookback are counted in points).
"""
import numpy as np
import pandas as pd

from src.anomaly_agent.detectors.runs import alerts_frame, runs_to_alerts


def detect_drift(metrics, window=288, lookback=288, z_threshold=6.0, min_rel_change=0.01,
                 train_days=8, min_points=3):
    """
    window: points in the trailing mean (288 = 1 day of 5-minute points).
    lookback: how far back to compare the trailing mean.
    z_threshold: change must beat this many spreads of the change seen in training.
    min_rel_change: ...and at least this fraction of the series' normal level.
    """
    rows = []
    for (entity, metric), g in metrics.groupby(["entity", "metric"]):
        g = g.sort_values("timestamp")
        ts = pd.DatetimeIndex(g["timestamp"])
        s = pd.Series(g["value"].to_numpy(float))

        level = s.rolling(window, min_periods=window).mean()
        change = level - level.shift(lookback)

        train = np.asarray(ts < ts[0] + pd.Timedelta(days=train_days))
        spread = change[train].std()
        spread = 0.0 if np.isnan(spread) else spread
        limit = max(z_threshold * spread, min_rel_change * abs(s[train].median()))

        score = (change.abs() - limit).to_numpy()  # NaN at the start: counts as normal
        rows += runs_to_alerts("drift", entity, metric, g["timestamp"].to_numpy(),
                               g["value"].to_numpy(float), score, min_points)
    return alerts_frame(rows)
