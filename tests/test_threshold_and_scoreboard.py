import pandas as pd

from src.anomaly_agent.adapters.csv_adapter import to_utc
from src.anomaly_agent.detectors.threshold import detect_threshold
from src.anomaly_agent.evaluation.scoreboard import score


def series(values, entity="x", metric="m", start="2025-01-01 00:00"):
    ts = pd.date_range(start, periods=len(values), freq="5min")
    return pd.DataFrame({"timestamp": ts, "entity": entity, "metric": metric, "value": values})


def test_needs_min_points_in_a_row():
    # two high points (too short) then four high points (alert)
    data = series([1, 9, 9, 1, 9, 9, 9, 9, 1])
    alerts = detect_threshold(data, {"m": {"upper": 5}}, min_points=3)
    assert len(alerts) == 1
    assert alerts.loc[0, "peak_value"] == 9
    # known at the 3rd point of the run, not the 1st
    assert alerts.loc[0, "detected_at"] - alerts.loc[0, "start"] == pd.Timedelta(minutes=10)


def test_metric_without_limit_is_ignored():
    alerts = detect_threshold(series([99, 99, 99, 99]), {}, min_points=3)
    assert alerts.empty


def test_lower_limit():
    alerts = detect_threshold(series([5, 0, 0, 0, 5]), {"m": {"lower": 1}}, min_points=3)
    assert len(alerts) == 1 and alerts.loc[0, "peak_value"] == 0


def make_truth():
    truth = pd.DataFrame([
        {"type": "real", "start": pd.Timestamp("2025-01-02"), "end": pd.Timestamp("2025-01-02 01:00"),
         "series": "m:x", "is_anomaly": True},
        {"type": "sale", "start": pd.Timestamp("2025-01-03"), "end": pd.Timestamp("2025-01-04"),
         "series": "m:x", "is_anomaly": False},
    ])
    truth["series_set"] = truth["series"].str.split(";").apply(set)
    return truth


def alert(start, end, detected, entity="x"):
    return {"detector": "t", "entity": entity, "metric": "m", "start": pd.Timestamp(start),
            "end": pd.Timestamp(end), "detected_at": pd.Timestamp(detected), "peak_value": 1.0}


def test_scoreboard_hit_miss_and_false_alarm():
    alerts = pd.DataFrame([
        alert("2025-01-02 00:10", "2025-01-02 00:30", "2025-01-02 00:20"),  # true detection
        alert("2025-01-03 05:00", "2025-01-03 06:00", "2025-01-03 05:10"),  # sale day -> false alarm
        alert("2025-01-05 00:00", "2025-01-05 01:00", "2025-01-05 00:10"),  # nothing there
    ])
    summary, problems, false_alarms = score(make_truth(), alerts)
    assert summary["true_detections"] == 1 and summary["misses"] == 0
    assert summary["false_alarms"] == 2
    assert problems.loc[0, "minutes_to_detect"] == 20
    assert "normal event: sale" in set(false_alarms["why"])


def test_scoreboard_miss_when_wrong_entity():
    alerts = pd.DataFrame([alert("2025-01-02 00:10", "2025-01-02 00:30", "2025-01-02 00:20", entity="other")])
    summary, _, _ = score(make_truth(), alerts)
    assert summary["misses"] == 1 and summary["false_alarms"] == 1


def test_local_time_becomes_utc():
    # 05:30 in India is 00:00 UTC
    out = to_utc(pd.Series(["01/01/2025 05:30"]), "%d/%m/%Y %H:%M", "Asia/Kolkata")
    assert out[0] == pd.Timestamp("2025-01-01 00:00")
