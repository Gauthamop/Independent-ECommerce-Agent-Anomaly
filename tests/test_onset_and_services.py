import math

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.anomaly_agent.adapters.csv_adapter import build
from src.anomaly_agent.agent.graph import investigate
from src.anomaly_agent.agent.llm_agent import (load_service_descriptions, make_llm_judge, make_llm_planner)
from src.anomaly_agent.agent.planner import ONSET_TOLERANCE_MIN, onset_args
from src.anomaly_agent.llm.scripted import ScriptedLLM
from src.anomaly_agent.tools.metric_tools import ALLOWED_TOOLS, MetricTools

ONSET = pd.Timestamp("2025-01-04 12:00")         # when the ramp and the step really begin
ALERT = pd.Timestamp("2025-01-05 00:30")         # when a detector first noticed the slow ramp (12.5 h later)


@pytest.fixture
def ramp_db(tmp_path):
    """6 days of 5-minute data: e1 = slow ramp from ONSET (4 per day), e2 = flat, e3 = sudden step at ONSET."""
    rng = np.random.default_rng(3)
    ts = pd.date_range("2025-01-01", periods=6 * 288, freq="5min")
    wave = 3 * np.sin((ts.hour + ts.minute / 60 - 10) / 24 * 2 * np.pi)
    hours = np.clip(np.asarray((ts - ONSET) / pd.Timedelta(hours=1)), 0, None)
    series = {"e1": 50 + wave + 4 / 24 * hours, "e2": 50 + wave, "e3": 50 + wave + np.where(ts >= ONSET, 10, 0)}
    frames = [pd.DataFrame({"timestamp": ts, "entity": name, "metric": "m", "value": v + rng.normal(0, 1, len(ts)),
                            "tags": "{}"}) for name, v in series.items()]
    changes = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 08:00", ONSET]), "service": ["svc-b", "svc-a"],
        "change_type": ["deploy", "config_change"], "description": ["harmless", "the real cause"]})
    path = tmp_path / "ramp.duckdb"
    con = duckdb.connect(str(path))
    for name, frame in [("metrics", pd.concat(frames, ignore_index=True)), ("changes", changes)]:
        con.register("f", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM f")
        con.unregister("f")
    con.close()
    return str(path)


def incident(detectors, start=ALERT, entity="e1"):
    return {"incident_id": "I1", "entity": entity, "metric": "m", "start": start, "end": start + pd.Timedelta(hours=2),
            "detected_at": start + pd.Timedelta(minutes=10), "detectors": detectors, "n_alerts": 1}


# ---- the onset estimator ---------------------------------------------------------------------
def test_a_slow_ramp_is_traced_back_to_near_its_real_start(ramp_db):
    out = MetricTools(ramp_db).estimate_onset("m", "e1", ALERT)
    onset = pd.Timestamp(out["onset"])
    assert ONSET <= onset <= ONSET + pd.Timedelta(hours=4)           # late by a few hours at most, never early
    assert 8 <= out["hours_before_alert"] <= 12.5                    # far earlier than the alert itself


def test_a_sudden_step_is_found_close_to_its_start(ramp_db):
    out = MetricTools(ramp_db).estimate_onset("m", "e3", ONSET + pd.Timedelta(minutes=15))
    assert abs((pd.Timestamp(out["onset"]) - ONSET).total_seconds()) <= 30 * 60


def test_a_calm_series_has_no_earlier_onset(ramp_db):
    out = MetricTools(ramp_db).estimate_onset("m", "e2", ALERT)
    assert out["onset"] == ALERT.strftime("%Y-%m-%d %H:%M") and "no earlier" in out["note"]


def test_calm_series_rarely_gets_a_false_onset_across_many_moments(ramp_db):
    tools = MetricTools(ramp_db)
    moments = pd.date_range("2025-01-05 00:30", "2025-01-06 23:00", periods=40)
    false_onsets = sum(tools.estimate_onset("m", "e2", t)["hours_before_alert"] > 0 for t in moments)
    assert false_onsets <= 4                                    # designed for about 1%, allow 10% in this small test
    # while a real slow ramp keeps being found at every one of those moments
    assert all(tools.estimate_onset("m", "e1", t)["hours_before_alert"] > 6 for t in moments)


def test_not_enough_history_is_said_not_guessed(ramp_db):
    out = MetricTools(ramp_db).estimate_onset("m", "e1", pd.Timestamp("2025-01-01 02:00"))
    assert out["onset"] is None and out["note"]
    assert MetricTools(ramp_db).estimate_onset("m", "nobody", ALERT)["onset"] is None


def test_the_new_tool_is_allowed_and_read_only():
    assert "estimate_onset" in ALLOWED_TOOLS


def test_changes_shortly_after_an_estimated_onset_still_count_as_simultaneous(ramp_db):
    tools = MetricTools(ramp_db)
    before = ONSET - pd.Timedelta(minutes=30)                         # an estimate that came out a bit early
    assert tools.get_recent_changes(before, lookback_hours=6)["n_found"] == 0
    out = tools.get_recent_changes(before, lookback_hours=6, tolerance_minutes=60)
    assert out["n_found"] == 1 and out["changes"][0]["minutes_before"] == 0       # clamped, never negative


# ---- the agent uses it for slow problems -------------------------------------------------------
def test_a_slow_problem_is_matched_against_its_estimated_start_not_the_late_alert(ramp_db):
    report, evidence = investigate(incident(["drift", "seasonal"]), MetricTools(ramp_db))
    assert [e["tool"] for e in evidence] == ["query_metric", "compare_segments", "estimate_onset", "get_recent_changes"]
    onset = evidence[2]["result"]["onset"]
    assert evidence[3]["args"]["before"] == onset and evidence[3]["args"]["tolerance_minutes"] == ONSET_TOLERANCE_MIN
    top = report["top_causes"][0]
    assert top["service"] == "svc-a" and top["score"] >= 0.8 and report["confidence"] == "high"
    assert "estimated to have begun" in report["since_when"] and "[E3]" in report["since_when"]
    # the old way (distance from the late alert, 24 h fade) would have given much less
    old_timing = math.exp(-(ALERT - ONSET).total_seconds() / 60 / (24 * 60))
    assert top["score"] > old_timing + 0.2


def test_a_sudden_problem_does_not_use_the_estimator(ramp_db):
    _, evidence = investigate(incident(["seasonal"], start=ONSET, entity="e3"), MetricTools(ramp_db))
    assert "estimate_onset" not in [e["tool"] for e in evidence]


def test_the_llm_planner_can_ask_for_the_onset_but_code_fills_in_the_facts():
    state = {"incident": incident(["drift"]), "evidence": [], "steps": 2, "max_steps": 6}
    action = make_llm_planner(ScriptedLLM(lambda s, u: {"tool": "estimate_onset", "metric": "HACKED"}))(state)
    assert action == {"tool": "estimate_onset", "args": onset_args(state["incident"])}

    state["evidence"] = [{"id": "E1", "tool": "estimate_onset", "args": {}, "result": {"onset": "2025-01-04 13:00"}}]
    action = make_llm_planner(ScriptedLLM(lambda s, u: {"tool": "get_recent_changes", "lookback_hours": 24}))(state)
    assert action["args"]["before"] == "2025-01-04 13:00" and action["args"]["tolerance_minutes"] == ONSET_TOLERANCE_MIN


# ---- service descriptions for the judge -------------------------------------------------------
def test_the_judge_is_told_what_each_service_does(ramp_db):
    llm = ScriptedLLM(lambda s, u: {"judgements": []})
    judge = make_llm_judge(llm, {"m": "a test metric"}, {"svc-a": "sends card payments to banks"})
    investigate(incident(["drift", "seasonal"]), MetricTools(ramp_db), judge=judge)
    prompt = llm.calls[0][1]
    assert '(what it does: sends card payments to banks)' in prompt and "What this metric measures: a test metric" in prompt
    # a service the catalog does not know is simply named, with no made-up description
    llm2 = ScriptedLLM(lambda s, u: {"judgements": []})
    investigate(incident(["drift", "seasonal"]), MetricTools(ramp_db), judge=make_llm_judge(llm2, {}, {}))
    assert 'service="svc-a"' in llm2.calls[0][1] and "what it does" not in llm2.calls[0][1]


def test_service_descriptions_come_from_the_profile_via_the_adapter(tmp_path):
    (tmp_path / "x.csv").write_text("timestamp,value\n2014-01-01 00:00:00,1\n", encoding="utf-8")
    (tmp_path / "c.csv").write_text("when,svc,kind,note\n2014-01-01 00:00,s,deploy,n\n", encoding="utf-8")
    profile = tmp_path / "p.yaml"
    profile.write_text(
        "timezone: UTC\nmetrics: {m: {type: gauge}}\nservices:\n  pay: \"sends payments\"\n"
        "sources:\n  - {file: x.csv, time_col: timestamp, time_format: '%Y-%m-%d %H:%M:%S', "
        "columns: [{pattern: '^value$', metric: m, entity: e}]}\n"
        "changes: {file: c.csv, time_col: when, time_format: '%Y-%m-%d %H:%M', "
        "columns: {service: svc, change_type: kind, description: note}}\n", encoding="utf-8")
    build(str(profile), str(tmp_path), str(tmp_path / "out.duckdb"))
    assert load_service_descriptions(tmp_path / "out.duckdb") == {"pay": "sends payments"}
    assert load_service_descriptions(tmp_path / "missing.duckdb") == {}
