import pandas as pd

from src.anomaly_agent.evaluation.agent_eval import DECOYS, evaluate, summarize


def alert(start, end, entity="e1"):
    return {"detector": "seasonal", "entity": entity, "metric": "m", "start": pd.Timestamp(start),
            "end": pd.Timestamp(end), "detected_at": pd.Timestamp(start) + pd.Timedelta(minutes=10),
            "peak_value": 1.0}


def truth_row(kind, start, end, cause_type, service=None, time=None, series="m:e1", real=True):
    return {"type": kind, "start": pd.Timestamp(start), "end": pd.Timestamp(end), "series": series,
            "is_anomaly": real, "cause_type": cause_type, "cause_service": service, "cause_time": time}


def make_truth(rows):
    truth = pd.DataFrame(rows)
    truth["series_set"] = truth["series"].str.split(";").apply(set)
    return truth


def test_true_cause_is_credited_and_cause_removed_twin_is_handled(small_db):
    alerts = pd.DataFrame([alert("2025-01-03 12:00", "2025-01-03 14:00")])
    truth = make_truth([truth_row("jump", "2025-01-03 12:00", "2025-01-03 14:00",
                                  "change", "svc-a", "2025-01-03 12:00")])
    cases = evaluate(small_db, alerts, truth)
    assert list(cases["case"]) == ["jump", "jump (cause removed)"]
    assert cases.loc[0, "correct"] and cases.loc[0, "passed"] and cases.loc[0, "confidence"] == "high"
    # with the real cause deleted from the log, it must not blame the decoy confidently
    assert cases.loc[1, "confidence"] == "low" and cases.loc[1, "passed"]
    assert summarize(cases)["overconfident_wrong_answers"] == 0


def test_external_problem_is_correct_only_when_agent_says_no_change(small_db):
    alerts = pd.DataFrame([alert("2025-01-03 20:00", "2025-01-03 21:00")])
    truth = make_truth([truth_row("outside", "2025-01-03 20:00", "2025-01-03 21:00", "external")])
    cases = evaluate(small_db, alerts, truth)
    assert list(cases["case"]) == ["outside"]            # no twins unless decoys are asked for
    assert cases.loc[0, "correct"] and cases.loc[0, "confidence"] == "low" and cases.loc[0, "passed"]

    cases = evaluate(small_db, alerts, truth, decoys=DECOYS)
    # external -> one decoy twin per decoy, and no cause-removed twin
    assert list(cases["case"]) == ["outside"] + [f"outside (decoy added: {d['service']})" for d in DECOYS]
    # the timing-only agent is fooled by every unrelated change landing just before: a known weakness
    twins = cases.iloc[1:]
    assert twins["overconfident_wrong"].all() and not twins["passed"].any()


def test_wrong_cause_fails_and_confident_wrong_is_counted(small_db):
    # the truth says the cause was svc-b, but the agent (rightly, by timing) blames svc-a
    alerts = pd.DataFrame([alert("2025-01-03 12:00", "2025-01-03 14:00")])
    truth = make_truth([truth_row("jump", "2025-01-03 12:00", "2025-01-03 14:00",
                                  "change", "svc-b", "2025-01-02 08:00")])
    cases = evaluate(small_db, alerts, truth)
    assert not cases.loc[0, "correct"] and not cases.loc[0, "passed"]
    assert cases.loc[0, "overconfident_wrong"]


def test_problem_the_detectors_missed_is_a_failed_case(small_db):
    alerts = pd.DataFrame([alert("2025-01-03 12:00", "2025-01-03 14:00", entity="e2")])  # wrong series
    truth = make_truth([truth_row("jump", "2025-01-03 12:00", "2025-01-03 14:00",
                                  "change", "svc-a", "2025-01-03 12:00")])
    cases = evaluate(small_db, alerts, truth)
    assert len(cases) == 1 and not cases.loc[0, "passed"] and "missed" in cases.loc[0, "got"]
