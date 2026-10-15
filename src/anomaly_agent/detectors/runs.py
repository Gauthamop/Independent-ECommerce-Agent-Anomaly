"""Shared helper: turn per-point scores into alerts. Every detector uses this.

A detector gives each point a `score`. score > 0 means "outside normal".
Alerts are runs of at least `min_points` such points in a row (NaN counts as normal).
"""
import numpy as np
import pandas as pd

ALERT_COLUMNS = ["detector", "entity", "metric", "start", "end", "detected_at", "peak_value"]


def runs_to_alerts(detector, entity, metric, times, values, score, min_points):
    """detected_at is when you would first KNOW: the point where the run reaches min_points."""
    flag = score > 0
    # +1 where a run starts, -1 just after it ends
    edges = np.diff(np.concatenate(([0], flag.astype(int), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]  # exclusive

    rows = []
    for s, e in zip(starts, ends):
        if e - s < min_points:
            continue
        peak = values[s + int(np.argmax(score[s:e]))]  # the most extreme point of the run
        rows.append({
            "detector": detector,
            "entity": entity,
            "metric": metric,
            "start": times[s],
            "end": times[e - 1],
            "detected_at": times[s + min_points - 1],
            "peak_value": float(peak),
        })
    return rows


def alerts_frame(rows):
    out = pd.DataFrame(rows, columns=ALERT_COLUMNS)
    return out.sort_values("start").reset_index(drop=True)
