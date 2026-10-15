"""Scoreboard: grade a detector's alerts against the answer key.

This is the ONLY place that reads ground truth. Detectors never do.
Counts: true detections, misses, false alarms, and minutes-to-detect.
"""
import pandas as pd


def load_truth(path="data/ground_truth/ground_truth.csv"):
    truth = pd.read_csv(path, parse_dates=["start", "end"])
    truth["series_set"] = truth["series"].str.split(";").apply(set)
    return truth


def _matches(alert, problem):
    """Same series and the time windows overlap."""
    same_series = f"{alert.metric}:{alert.entity}" in problem.series_set
    overlaps = alert.start <= problem.end and alert.end >= problem.start
    return same_series and overlaps


def score(truth, alerts):
    """Return (summary dict, per-problem table, false-alarm table)."""
    real = truth[truth["is_anomaly"]]
    normal = truth[~truth["is_anomaly"]]

    problem_rows = []
    for p in real.itertuples():
        hits = [a for a in alerts.itertuples() if _matches(a, p)]
        if hits:
            first = min(a.detected_at for a in hits)
            minutes = (first - p.start).total_seconds() / 60
            problem_rows.append({"problem": p.type, "found": True, "minutes_to_detect": minutes,
                                 "alerts": len(hits)})
        else:
            problem_rows.append({"problem": p.type, "found": False, "minutes_to_detect": None,
                                 "alerts": 0})
    problems = pd.DataFrame(problem_rows)

    # a false alarm is an alert that matches no real problem
    false_rows = []
    for a in alerts.itertuples():
        if any(_matches(a, p) for p in real.itertuples()):
            continue
        why = "no problem there"
        for n in normal.itertuples():
            if _matches(a, n):
                why = f"normal event: {n.type}"
        false_rows.append({"entity": a.entity, "metric": a.metric, "start": a.start,
                           "end": a.end, "why": why})
    false_alarms = pd.DataFrame(false_rows, columns=["entity", "metric", "start", "end", "why"])

    found = problems[problems["found"]]
    summary = {
        "true_detections": int(problems["found"].sum()),
        "misses": int((~problems["found"]).sum()),
        "false_alarms": len(false_alarms),
        "alerts_total": len(alerts),
        "mean_minutes_to_detect": (float(found["minutes_to_detect"].mean())
                                   if len(found) else None),
    }
    return summary, problems, false_alarms


def print_scoreboard(name, summary, problems, false_alarms):
    print(f"\n=== SCOREBOARD: {name} ===")
    print(problems.to_string(index=False))
    print()
    for key, value in summary.items():
        print(f"{key:>24}: {value if value is None else round(value, 1) if isinstance(value, float) else value}")
    if len(false_alarms):
        print("\nfalse alarms:")
        print(false_alarms.to_string(index=False))
