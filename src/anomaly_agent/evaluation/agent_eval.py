"""Evaluate the agent against the answer key (step 6).

For every real problem in the ground truth we ask:
  - did the agent name the TRUE cause (service + time) as its top cause?
  - for a problem with no internal cause, did it say "no change found" with low confidence?
  - was it ever confidently wrong?
  - does every claim cite evidence?
We also run a "cause removed" test: delete the true cause from the change log and check that the
agent does NOT confidently blame something else. That is the "no real cause" case.

This file reads ground truth, like scoreboard.py. The agent itself never does.

Usage (from project root, with .venv active):
    python -m src.anomaly_agent.evaluation.agent_eval
"""
import shutil
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from src.anomaly_agent.agent.graph import investigate
from src.anomaly_agent.agent.incidents import group_alerts
from src.anomaly_agent.evaluation.scoreboard import load_truth
from src.anomaly_agent.tools.metric_tools import MetricTools


def _matches(incident, problem):
    """Same series and overlapping time windows (same rule as the scoreboard)."""
    same_series = f"{incident['metric']}:{incident['entity']}" in problem.series_set
    overlaps = incident["start"] <= problem.end and incident["end"] >= problem.start
    return same_series and overlaps


def _is_external(cause):
    return cause["service"] is None and cause["score"] == 0.0


def _cited(report):
    causes_cited = all(c["evidence_ids"] for c in report["top_causes"])
    text_cited = all("[E" in report[k] or "not enough evidence" in report[k]
                     for k in ("what_happened", "who_affected"))
    return causes_cited and text_cited


def _short(cause):
    return "no change found" if _is_external(cause) else f"{cause['service']} @ {cause['change_time']}"


def judge(report, problem, evidence, label):
    """Compare one report with one ground-truth row."""
    top = report["top_causes"][0] if report["top_causes"] else None
    conf = report["confidence"]
    cited = _cited(report)

    if top is None:
        correct, in_top3, expected = False, False, "?"
    elif problem.cause_type == "change":
        expected = f"{problem.cause_service} @ {problem.cause_time}"
        match = lambda c: c["service"] == problem.cause_service and c["change_time"] == problem.cause_time
        correct = match(top)
        in_top3 = any(match(c) for c in report["top_causes"])
    else:  # external: the right answer is "no change found"
        expected = "no change found"
        correct = _is_external(top)
        in_top3 = correct

    if problem.cause_type == "change":
        confidence_ok = conf in ("medium", "high") if correct else conf == "low"
    else:
        confidence_ok = conf == "low"
    passed = correct and confidence_ok and cited

    return {"case": label, "expected": expected, "got": _short(top) if top else "-",
            "confidence": conf, "correct": correct, "in_top3": in_top3,
            "overconfident_wrong": (not correct) and conf != "low",
            "cited": cited, "steps": len(evidence), "passed": passed}


def judge_cause_removed(report, evidence, label):
    """The true cause is gone from the log. Passing = not confidently blaming anything else."""
    top = report["top_causes"][0] if report["top_causes"] else None
    cited = _cited(report)
    conf = report["confidence"]
    return {"case": label, "expected": "no cause (removed)", "got": _short(top) if top else "-",
            "confidence": conf, "correct": None, "in_top3": None,
            "overconfident_wrong": conf != "low", "cited": cited, "steps": len(evidence),
            "passed": conf == "low" and cited}


def _without_change(db_path, service, time, workdir):
    copy = Path(workdir) / "cause_removed.duckdb"
    shutil.copy(db_path, copy)
    con = duckdb.connect(str(copy))
    con.execute("DELETE FROM changes WHERE service = ? AND timestamp = ?",
                [service, pd.Timestamp(time).to_pydatetime()])
    con.close()
    return str(copy)


def evaluate(db_path, alerts, truth, max_steps=6):
    """Run the agent on every real problem (and its cause-removed twin). Returns one row per case."""
    incidents = group_alerts(alerts)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in truth[truth["is_anomaly"]].itertuples():
            inc = next((i for i in incidents if _matches(i, p)), None)
            if inc is None:
                rows.append({"case": p.type, "expected": "-", "got": "no incident (detectors missed it)",
                             "confidence": "-", "correct": False, "in_top3": False,
                             "overconfident_wrong": False, "cited": False, "steps": 0, "passed": False})
                continue

            report, evidence = investigate(inc, MetricTools(db_path), max_steps=max_steps)
            rows.append(judge(report, p, evidence, p.type))

            if p.cause_type == "change":
                twin_db = _without_change(db_path, p.cause_service, p.cause_time, tmp)
                report, evidence = investigate(inc, MetricTools(twin_db), max_steps=max_steps)
                rows.append(judge_cause_removed(report, evidence, f"{p.type} (cause removed)"))
    return pd.DataFrame(rows)


def summarize(cases):
    real = cases[~cases["case"].str.contains("removed")]
    removed = cases[cases["case"].str.contains("removed")]
    return {
        "top_cause_correct": f"{int(real['correct'].sum())}/{len(real)}",
        "cause_in_top3": f"{int(real['in_top3'].sum())}/{len(real)}",
        "cause_removed_handled": f"{int(removed['passed'].sum())}/{len(removed)}",
        "overconfident_wrong_answers": int(cases["overconfident_wrong"].sum()),
        "all_claims_cited": bool(cases["cited"].all()),
        "cases_passed": f"{int(cases['passed'].sum())}/{len(cases)}",
    }


def main(db_path="data/clean/anomaly.duckdb", truth_path="data/ground_truth/ground_truth.csv",
         out_path="data/clean/agent_eval.csv"):
    con = duckdb.connect(db_path, read_only=True)
    alerts = con.execute("SELECT * FROM anomalies").df()
    con.close()

    cases = evaluate(db_path, alerts, load_truth(truth_path))
    cases.to_csv(out_path, index=False)

    print("\n=== AGENT EVALUATION ===")
    print(cases.drop(columns=["in_top3"]).to_string(index=False))
    print()
    for key, value in summarize(cases).items():
        print(f"{key:>28}: {value}")


if __name__ == "__main__":
    main()
