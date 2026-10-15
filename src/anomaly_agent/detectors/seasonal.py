"""Seasonal baseline detector.

Learns what is normal for every hour of the week (168 slots) from a training period,
using the median and a robust spread (MAD). Then a point is odd when it is far from the
normal value FOR THAT HOUR. So a quiet-hour burst is visible and a busy-hour peak is not.

Assumption: the training period is clean (a "learning period"). Median and MAD are used
because they shrug off a few bad points that slip into training.
It never reads ground truth and knows no business words.
"""
import numpy as np
import pandas as pd

from src.anomaly_agent.detectors.runs import alerts_frame, runs_to_alerts


def detect_seasonal(metrics, train_days=8, z_threshold=4.0, min_points=3,
                    event_days=(), suppress_upward_on_events=False, spread_floor=0.02,
                    pool_spread=True):
    """
    train_days: the first N days of each series are used to learn normal.
    z_threshold: how many robust spreads away counts as odd.
    event_days: planned event dates (e.g. a sale), from the profile calendar.
    suppress_upward_on_events: on event days, ignore points ABOVE normal (volume surges are
        expected). Points below normal still alert.
    spread_floor: spread is never less than this fraction of the normal value.
    pool_spread: a slot's spread is never below the median spread over all slots (less noisy).
    """
    events = {pd.Timestamp(d).date() for d in event_days}
    rows = []
    for (entity, metric), g in metrics.groupby(["entity", "metric"]):
        g = g.sort_values("timestamp")
        ts = pd.DatetimeIndex(g["timestamp"])
        values = g["value"].to_numpy(float)
        slot = pd.Series((ts.dayofweek * 24 + ts.hour).to_numpy())

        # learn normal per hour-of-week, from the training period only (split by time)
        train = np.asarray(ts < ts[0] + pd.Timedelta(days=train_days))
        learn = pd.DataFrame({"slot": slot[train].to_numpy(), "v": values[train]})
        median = learn.groupby("slot")["v"].median()
        mad = learn.groupby("slot")["v"].apply(lambda v: (v - v.median()).abs().median()) * 1.4826

        if pool_spread:
            # a slot has few training points, so its spread is noisy; never go below the typical slot
            mad = mad.clip(lower=mad.median())

        expected = slot.map(median).to_numpy()
        spread = slot.map(mad).to_numpy()
        spread = np.maximum(spread, spread_floor * np.abs(expected)) + 1e-9  # NaN stays NaN

        z = (values - expected) / spread
        score = np.abs(z) - z_threshold

        if suppress_upward_on_events and events:
            on_event = pd.Series(ts.date).isin(events).to_numpy()
            score = np.where(on_event & (z > 0), -1.0, score)

        rows += runs_to_alerts("seasonal", entity, metric, g["timestamp"].to_numpy(), values,
                               score, min_points)
    return alerts_frame(rows)
