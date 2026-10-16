import json

import duckdb
import pandas as pd

from src.anomaly_agent.adapters.csv_adapter import build
from src.anomaly_agent.detectors import registry
from src.anomaly_agent.real.run_real import DATASETS, nab_truth, static_limits


def test_adapter_averages_duplicate_timestamps_and_says_so(tmp_path, capsys):
    (tmp_path / "x.csv").write_text("timestamp,value\n2014-01-01 00:00:00,10\n2014-01-01 00:00:00,20\n"
                                    "2014-01-01 00:05:00,30\n", encoding="utf-8")
    (tmp_path / "changes.csv").write_text("when,svc,kind,note\n2014-01-01 00:00,s,deploy,n\n", encoding="utf-8")
    profile = tmp_path / "p.yaml"
    profile.write_text(
        "timezone: UTC\nmetrics: {m: {type: gauge}}\n"
        "sources:\n  - {file: x.csv, time_col: timestamp, time_format: '%Y-%m-%d %H:%M:%S', "
        "columns: [{pattern: '^value$', metric: m, entity: e}]}\n"
        "changes: {file: changes.csv, time_col: when, time_format: '%Y-%m-%d %H:%M', "
        "columns: {service: svc, change_type: kind, description: note}}\n", encoding="utf-8")
    build(str(profile), str(tmp_path), str(tmp_path / "out.duckdb"))
    assert "1 duplicate timestamps were averaged" in capsys.readouterr().out
    rows = duckdb.connect(str(tmp_path / "out.duckdb"), read_only=True).execute(
        "SELECT value FROM metrics ORDER BY timestamp").fetchall()
    assert rows == [(15.0,), (30.0,)]                      # 10 and 20 averaged, nothing silently lost


def test_registry_applies_per_metric_overrides(monkeypatch):
    seen = {}

    def spy(subset, **kwargs):
        seen[subset["metric"].iloc[0]] = kwargs
        return pd.DataFrame(columns=registry.ALERT_COLUMNS)

    monkeypatch.setitem(registry.DETECTORS, "seasonal", spy)
    metrics = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01"] * 2), "entity": "e",
                            "metric": ["slow", "fast"], "value": [1.0, 2.0]})
    info = pd.DataFrame({"metric": ["slow", "fast"], "type": ["count", "count"]})
    profile = {"detectors": {"count": ["seasonal"]}, "detector_params": {"seasonal": {"train_days": 8, "min_points": 3}},
               "overrides": {"slow": {"seasonal": {"train_days": 28}}}}
    registry.run_detectors(metrics, info, profile)
    assert seen["slow"]["train_days"] == 28 and seen["fast"]["train_days"] == 8   # only "slow" is overridden
    assert seen["slow"]["min_points"] == 3                                        # the rest is inherited


def test_nab_windows_become_ground_truth(tmp_path):
    key = next(iter(DATASETS))
    metric, entity = DATASETS[key]
    path = tmp_path / "w.json"
    path.write_text(json.dumps({k: [["2014-01-01 00:00:00.000000", "2014-01-02 00:00:00.000000"]]
                                for k in DATASETS}), encoding="utf-8")
    truth = nab_truth(path)
    assert len(truth) == len(DATASETS) and truth["is_anomaly"].all()
    row = truth[truth["series"] == f"{metric}:{entity}"].iloc[0]
    assert row["start"] == pd.Timestamp("2014-01-01") and row["series_set"] == {f"{metric}:{entity}"}


def test_static_limits_come_from_the_training_period_only():
    ts = pd.date_range("2025-01-01", periods=10 * 288, freq="5min")
    values = [10.0, 12.0] * (4 * 288) + [1000.0] * (2 * 288)        # a huge shift after the training days
    metrics = pd.DataFrame({"timestamp": ts, "entity": "e", "metric": "m", "value": values})
    limits = static_limits(metrics, {"overrides": {"m": {"seasonal": {"train_days": 8}}}})
    assert 10 < limits["m"]["upper"] < 20 and limits["m"]["lower"] > 0     # not dragged up by the later 1000s
