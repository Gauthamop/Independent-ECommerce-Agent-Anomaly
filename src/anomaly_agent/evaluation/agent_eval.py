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
from src.anomaly_agent.agent.planner import rule_planner
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


def _llm_fell_back(report):
    """True when the LLM judge failed (rate limit, network, bad reply) and the case was scored by
    timing alone. Such a row is NOT a measurement of the LLM, and must never pass as one."""
    return any("LLM judge unavailable" in n for n in report["notes"])


def _short(cause):
    return "no change found" if _is_external(cause) else f"{cause['service']} @ {cause['change_time']}"


def grade_case(report, problem, evidence, label):
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
            "cited": cited, "steps": len(evidence), "llm_fallback": _llm_fell_back(report), "passed": passed}


def judge_cause_removed(report, evidence, label, expected="no cause (removed)"):
    """No real cause is in the log (removed, or never there). Passing = not confidently blaming
    anything else, even a change that happened right before."""
    top = report["top_causes"][0] if report["top_causes"] else None
    cited = _cited(report)
    conf = report["confidence"]
    return {"case": label, "expected": expected, "got": _short(top) if top else "-",
            "confidence": conf, "correct": None, "in_top3": None,
            "overconfident_wrong": conf != "low", "cited": cited, "steps": len(evidence),
            "llm_fallback": _llm_fell_back(report), "passed": conf == "low" and cited}


def _without_change(db_path, service, time, workdir):
    copy = Path(workdir) / "cause_removed.duckdb"
    shutil.copy(db_path, copy)
    con = duckdb.connect(str(copy))
    con.execute("DELETE FROM changes WHERE service = ? AND timestamp = ?",
                [service, pd.Timestamp(time).to_pydatetime()])
    con.close()
    return str(copy)


# harmless changes with NO plausible link to a payment or server metric. Each lands just before the
# problem, which is exactly what fools an agent that scores only by timing.
DECOYS = (
    {"service": "search-service", "change_type": "deploy", "minutes_before": 10,
     "description": "v1.9 rollout, new ranking tweaks"},
    {"service": "cart-service", "change_type": "deploy", "minutes_before": 5,
     "description": "v4.2 rollout, button colour and layout tweaks"},
    {"service": "email-service", "change_type": "config_change", "minutes_before": 20,
     "description": "updated the marketing email template footer"},
)


def _with_decoy(db_path, start, workdir, decoy):
    """A copy of the database with one harmless, UNRELATED change just before the problem."""
    copy = Path(workdir) / f"decoy_{decoy['service']}.duckdb"
    shutil.copy(db_path, copy)
    con = duckdb.connect(str(copy))
    con.execute("INSERT INTO changes VALUES (?, ?, ?, ?)",
                [(pd.Timestamp(start) - pd.Timedelta(minutes=decoy["minutes_before"])).to_pydatetime(),
                 decoy["service"], decoy["change_type"], decoy["description"]])
    con.close()
    return str(copy)


def evaluate(db_path, alerts, truth, max_steps=6, planner=rule_planner, judge=None, decoys=()):
    """Run the agent on every real problem plus its trick twins. Returns one row per case.

    Twins: "cause removed" (the true change is deleted) and, when `decoys` is given, for problems
    with an outside cause one "decoy added" twin per decoy (an unrelated change lands just before the
    problem: a timing-only agent blames it). planner / judge default to the rule-based agent; pass
    the LLM versions to compare. decoys defaults to none, so existing callers behave as before.
    """
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

            run = lambda db: investigate(inc, MetricTools(db), planner=planner, max_steps=max_steps,
                                         judge=judge)
            report, evidence = run(db_path)
            rows.append(grade_case(report, p, evidence, p.type))

            if p.cause_type == "change":
                report, evidence = run(_without_change(db_path, p.cause_service, p.cause_time, tmp))
                rows.append(judge_cause_removed(report, evidence, f"{p.type} (cause removed)"))
            elif p.cause_type == "external":
                for decoy in decoys:
                    report, evidence = run(_with_decoy(db_path, p.start, tmp, decoy))
                    rows.append(judge_cause_removed(report, evidence, f"{p.type} (decoy added: {decoy['service']})",
                                                    expected="no confident cause (decoy added)"))
    return pd.DataFrame(rows)


def summarize(cases):
    is_removed = cases["case"].str.contains("removed")
    is_decoy = cases["case"].str.contains("decoy")
    real, removed, decoy = cases[~is_removed & ~is_decoy], cases[is_removed], cases[is_decoy]
    return {
        "top_cause_correct": f"{int(real['correct'].sum())}/{len(real)}",
        "cause_in_top3": f"{int(real['in_top3'].sum())}/{len(real)}",
        "cause_removed_handled": f"{int(removed['passed'].sum())}/{len(removed)}",
        "decoy_handled": f"{int(decoy['passed'].sum())}/{len(decoy)}",
        "llm_fallbacks": int(cases["llm_fallback"].sum()) if "llm_fallback" in cases else 0,
        "overconfident_wrong_answers": int(cases["overconfident_wrong"].sum()),
        "all_claims_cited": bool(cases["cited"].all()),
        "cases_passed": f"{int(cases['passed'].sum())}/{len(cases)}",
    }


def main(db_path="data/clean/anomaly.duckdb", truth_path="data/ground_truth/ground_truth.csv",
         out_path="data/clean/agent_eval.csv", use_llm=False):
    con = duckdb.connect(db_path, read_only=True)
    alerts = con.execute("SELECT * FROM anomalies").df()
    con.close()

    planner, judge = rule_planner, None
    if use_llm:  # Groq planner + relatedness judge (needs GROQ_API_KEY)
        from src.anomaly_agent.agent.llm_agent import llm_parts
        from src.anomaly_agent.llm.groq_llm import GroqLLM
        planner, judge = llm_parts(GroqLLM(), db_path)
        out_path = out_path.replace(".csv", "_llm.csv")

    cases = evaluate(db_path, alerts, load_truth(truth_path), planner=planner, judge=judge, decoys=DECOYS)
    cases.to_csv(out_path, index=False)

    print(f"\n=== AGENT EVALUATION ({'Groq LLM' if use_llm else 'rule-based'}) ===")
    print(cases.drop(columns=["in_top3"]).to_string(index=False))
    print()
    for key, value in summarize(cases).items():
        print(f"{key:>28}: {value}")
    if use_llm and cases["llm_fallback"].any():
        print(f"\nWARNING: the LLM judge was unavailable in {int(cases['llm_fallback'].sum())} case(s) "
              "(rate limit or network?). Those rows were scored by timing alone: do not read them as LLM results.")


if __name__ == "__main__":
    import sys

    main(use_llm="--llm" in sys.argv)
