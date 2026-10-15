"""Static threshold detector: the baseline every later detector must beat.

Input is the canonical metrics table (timestamp, entity, metric, value) and a dict of
fixed limits per metric. It never reads ground truth and knows no business words.
"""
import numpy as np
import pandas as pd

ALERT_COLUMNS = ["detector", "entity", "metric", "start", "end", "detected_at", "peak_value"]


def detect_threshold(metrics, limits, min_points=3):
    """Alert when a series stays outside its limits for at least `min_points` points in a row.

    limits: {metric: {"upper": x, "lower": y}} (either side optional).
    detected_at is when you would first KNOW: the point where the run reaches min_points.
    """
    alerts = []
    for (entity, metric), g in metrics.groupby(["entity", "metric"]):
        limit = limits.get(metric)
        if not limit:
            continue
        g = g.sort_values("timestamp")
        values = g["value"].to_numpy()
        times = g["timestamp"].to_numpy()

        outside = np.zeros(len(values), dtype=bool)
        if "upper" in limit:
            outside |= values > limit["upper"]
        if "lower" in limit:
            outside |= values < limit["lower"]

        # find runs of consecutive True: +1 where a run starts, -1 just after it ends
        edges = np.diff(np.concatenate(([0], outside.astype(int), [0])))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]  # exclusive

        for s, e in zip(starts, ends):
            if e - s < min_points:
                continue
            run = values[s:e]
            peak = run.max() if "upper" in limit else run.min()
            alerts.append({
                "detector": "threshold",
                "entity": entity,
                "metric": metric,
                "start": times[s],
                "end": times[e - 1],
                "detected_at": times[s + min_points - 1],
                "peak_value": float(peak),
            })

    out = pd.DataFrame(alerts, columns=ALERT_COLUMNS)
    return out.sort_values("start").reset_index(drop=True)
