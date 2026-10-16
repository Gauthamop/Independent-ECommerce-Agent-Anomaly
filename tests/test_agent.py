import pandas as pd
import pytest

from src.anomaly_agent.agent.graph import investigate
from src.anomaly_agent.agent.incidents import group_alerts
from src.anomaly_agent.tools.metric_tools import MetricTools


def incident(start, end, entity="e1", detectors=("seasonal",)):
    return {"incident_id": "I1", "entity": entity, "metric": "m",
            "start": pd.Timestamp(start), "end": pd.Timestamp(end),
            "detected_at": pd.Timestamp(start) + pd.Timedelta(minutes=10),
            "detectors": list(detectors), "n_alerts": 1}


def test_agent_finds_the_change_right_before_the_problem(small_db):
    report, evidence = investigate(
        incident("2025-01-03 12:00", "2025-01-03 14:00"), MetricTools(small_db))
    assert [e["tool"] for e in evidence] == ["query_metric", "compare_segments", "get_recent_changes"]
    assert "svc-a" in report["top_causes"][0]["description"]
    assert report["confidence"] == "high"
    assert "e1" in report["who_affected"] and "[E2]" in report["who_affected"]
    assert all(c["evidence_ids"] for c in report["top_causes"])      # every cause cites evidence


def test_no_recent_change_means_low_confidence_not_a_guess(small_db):
    # a problem on e1 at a time with no change in the past 48 hours
    report, evidence = investigate(
        incident("2025-01-03 20:00", "2025-01-03 21:00"), MetricTools(small_db))
    # it looked wider once before giving up
    assert [e["args"].get("lookback_hours") for e in evidence if e["tool"] == "get_recent_changes"] == [6, 48]
    assert report["confidence"] == "low"
    assert report["top_causes"][0]["score"] == 0.0
    assert "external" in report["top_causes"][0]["description"]
    assert "do not roll anything back" in report["suggested_action"]


def test_step_limit_stops_and_says_not_enough_evidence(small_db):
    report, evidence = investigate(
        incident("2025-01-03 12:00", "2025-01-03 14:00"), MetricTools(small_db), max_steps=1)
    assert len(evidence) == 1
    assert report["confidence"] == "low"
    assert any("step limit" in n for n in report["notes"])
    assert "not enough evidence" in report["who_affected"]


def test_unknown_tool_is_refused(small_db):
    bad_planner = lambda state: {"tool": "drop_everything", "args": {}}
    with pytest.raises(ValueError, match="not allowed"):
        investigate(incident("2025-01-03 12:00", "2025-01-03 14:00"), MetricTools(small_db),
                    planner=bad_planner)


def test_group_alerts_merges_fragments_and_keeps_series_apart():
    t = pd.Timestamp("2025-01-01")
    row = lambda e, s, en, d: {"detector": d, "entity": e, "metric": "m", "start": t + pd.Timedelta(hours=s),
                               "end": t + pd.Timedelta(hours=en), "detected_at": t + pd.Timedelta(hours=s),
                               "peak_value": 1.0}
    alerts = pd.DataFrame([row("a", 0, 1, "seasonal"), row("a", 2, 3, "seasonal"),   # close: merge
                           row("a", 1, 10, "drift"),                                 # overlaps: merge
                           row("a", 20, 21, "seasonal"),                             # far: new incident
                           row("b", 0, 1, "seasonal")])                              # other series
    out = group_alerts(alerts, gap_minutes=120)
    assert len(out) == 3
    first = next(i for i in out if i["entity"] == "a" and i["start"] == t)
    assert first["n_alerts"] == 3 and first["detectors"] == ["drift", "seasonal"]
