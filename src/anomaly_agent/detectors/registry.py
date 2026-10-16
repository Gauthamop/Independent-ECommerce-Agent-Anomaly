"""Detector registry: pick detectors by METRIC TYPE (rate / count / gauge), as the profile says.

The profile decides, for example, that gauges get a seasonal baseline AND a drift detector.
This file contains no business words: only types, detector names and params from the profile.
"""
import pandas as pd

from src.anomaly_agent.detectors.drift import detect_drift
from src.anomaly_agent.detectors.runs import ALERT_COLUMNS, alerts_frame
from src.anomaly_agent.detectors.seasonal import detect_seasonal
from src.anomaly_agent.detectors.threshold import detect_threshold

DETECTORS = {
    "threshold": detect_threshold,
    "seasonal": detect_seasonal,
    "drift": detect_drift,
}


def run_detectors(metrics, metric_info, profile):
    """metrics: canonical table. metric_info: (metric, type). profile: parsed YAML dict."""
    types = dict(zip(metric_info["metric"], metric_info["type"]))
    params = profile.get("detector_params", {})
    event_days = [e["date"] for e in profile.get("events", [])]

    overrides = profile.get("overrides", {})  # per-metric settings, e.g. a slower sampling rate
    frames = []
    for metric_type, names in profile["detectors"].items():
        of_type = metrics[metrics["metric"].map(types) == metric_type]
        for metric, subset in of_type.groupby("metric"):
            for name in names:
                kwargs = {**params.get(name, {}), **overrides.get(metric, {}).get(name, {})}
                if name == "seasonal":
                    kwargs["event_days"] = event_days
                    # volume (count) metrics are expected to surge on planned event days
                    kwargs["suppress_upward_on_events"] = metric_type == "count"
                if name == "threshold":
                    kwargs["limits"] = profile["thresholds"]
                frames.append(DETECTORS[name](subset, **kwargs))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return alerts_frame([])
    return pd.concat(frames, ignore_index=True)[ALERT_COLUMNS].sort_values("start").reset_index(drop=True)
