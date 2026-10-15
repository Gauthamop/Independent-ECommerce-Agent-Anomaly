"""Static threshold detector: the baseline every later detector must beat.

Input is the canonical metrics table (timestamp, entity, metric, value) and a dict of
fixed limits per metric. It never reads ground truth and knows no business words.
"""
import numpy as np

from src.anomaly_agent.detectors.runs import ALERT_COLUMNS, alerts_frame, runs_to_alerts  # noqa: F401


def detect_threshold(metrics, limits, min_points=3):
    """Alert when a series stays outside its limits for at least `min_points` points in a row.

    limits: {metric: {"upper": x, "lower": y}} (either side optional).
    """
    rows = []
    for (entity, metric), g in metrics.groupby(["entity", "metric"]):
        limit = limits.get(metric)
        if not limit:
            continue
        g = g.sort_values("timestamp")
        values = g["value"].to_numpy(float)
        times = g["timestamp"].to_numpy()

        # score = how far outside the limits (positive = outside)
        score = np.full(len(values), -np.inf)
        if "upper" in limit:
            score = np.maximum(score, values - limit["upper"])
        if "lower" in limit:
            score = np.maximum(score, limit["lower"] - values)

        rows += runs_to_alerts("threshold", entity, metric, times, values, score, min_points)
    return alerts_frame(rows)
