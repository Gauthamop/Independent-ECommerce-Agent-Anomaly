import numpy as np
import pandas as pd

from src.anomaly_agent.detectors.drift import detect_drift
from src.anomaly_agent.detectors.registry import run_detectors
from src.anomaly_agent.detectors.seasonal import detect_seasonal

START = pd.Timestamp("2025-01-01")  # a Wednesday


def make_series(days=14, entity="x", metric="m", level=100.0, wave=40.0, seed=0, ramp_from=None,
                ramp_per_day=0.0):
    """A daily wave plus small noise, optionally with a slow ramp from day `ramp_from`."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(START, periods=days * 288, freq="5min")
    hour = ts.hour + ts.minute / 60
    values = level + wave * np.sin((hour - 10) / 24 * 2 * np.pi) + rng.normal(0, 1, len(ts))
    if ramp_from is not None:
        days_in = (ts - (START + pd.Timedelta(days=ramp_from))) / pd.Timedelta(days=1)
        values = values + ramp_per_day * np.clip(np.asarray(days_in), 0, None)
    return pd.DataFrame({"timestamp": ts, "entity": entity, "metric": metric, "value": values})


def bump(df, start, hours, amount):
    out = df.copy()
    t0 = pd.Timestamp(start)
    mask = (out.timestamp >= t0) & (out.timestamp < t0 + pd.Timedelta(hours=hours))
    out.loc[mask, "value"] += amount
    return out


def test_clean_series_has_no_seasonal_alerts():
    assert detect_seasonal(make_series()).empty


def test_same_jump_is_alert_at_quiet_hour():
    # +60 at 04:00 (the quiet low of the wave) is far from normal FOR THAT HOUR
    data = bump(make_series(), "2025-01-10 04:00", 1, 60)
    alerts = detect_seasonal(data)
    assert len(alerts) == 1
    assert alerts.loc[0, "start"] == pd.Timestamp("2025-01-10 04:00")


def test_event_day_hides_upward_surge_but_not_a_drop():
    event = ["2025-01-10"]
    surge = bump(make_series(), "2025-01-10 00:00", 24, 60)
    assert not detect_seasonal(surge).empty                       # no calendar: alert
    assert detect_seasonal(surge, event_days=event, suppress_upward_on_events=True).empty

    drop = bump(make_series(), "2025-01-10 00:00", 24, -60)
    assert not detect_seasonal(drop, event_days=event, suppress_upward_on_events=True).empty


def test_drift_catches_slow_ramp_and_ignores_flat():
    ramp = make_series(days=14, level=50, wave=3, ramp_from=10, ramp_per_day=4.5)
    alerts = detect_drift(ramp)
    assert len(alerts) >= 1
    # found within 2 days of the ramp starting
    assert alerts.loc[0, "detected_at"] < START + pd.Timedelta(days=12)

    assert detect_drift(make_series(days=14, level=50, wave=3)).empty


def test_registry_picks_detectors_by_metric_type():
    count = make_series(entity="c", metric="vol", level=50, wave=3, ramp_from=10, ramp_per_day=4.5)
    gauge = make_series(entity="g", metric="mem", level=50, wave=3, ramp_from=10, ramp_per_day=4.5)
    metrics = pd.concat([count, gauge], ignore_index=True)
    info = pd.DataFrame({"metric": ["vol", "mem"], "type": ["count", "gauge"]})
    profile = {"detectors": {"count": ["seasonal"], "gauge": ["seasonal", "drift"]},
               "detector_params": {}, "events": []}

    alerts = run_detectors(metrics, info, profile)
    used = alerts.groupby("metric")["detector"].apply(set).to_dict()
    assert used["vol"] == {"seasonal"}            # counts never get the drift detector
    assert "drift" in used["mem"]                 # gauges do
