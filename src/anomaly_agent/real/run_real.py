"""Run the detectors on REAL data (NAB) and score them against NAB's labelled anomaly windows.

    python -m src.anomaly_agent.real.run_real

Same adapter, registry and scoreboard as the made-up shop. What is different, and why it matters:
  - NAB windows are WIDE (days long) and the labels are known to be incomplete, so:
      * "found" is lenient (any alert inside the window counts), and
      * "false alarms" are OVER-counted (a real, unlabelled problem would count as a false alarm).
    So read the numbers as a rough comparison between setups, not as an accuracy score.
  - the static baseline is a classic 3-sigma limit learned from each series' clean training period.
"""
import json
from pathlib import Path

import duckdb
import pandas as pd

from src.anomaly_agent.adapters.csv_adapter import build, load_profile
from src.anomaly_agent.detectors.registry import run_detectors
from src.anomaly_agent.detectors.threshold import detect_threshold
from src.anomaly_agent.evaluation.scoreboard import score
from src.anomaly_agent.real.download_nab import download

REAL = Path("data/real")
PROFILE = "profiles/nab.yaml"
DB = REAL / "clean" / "nab.duckdb"

# NAB label key -> (metric, entity), matching profiles/nab.yaml
DATASETS = {
    "realKnownCause/nyc_taxi.csv": ("taxi_trips", "nyc"),
    "realKnownCause/cpu_utilization_asg_misconfiguration.csv": ("cpu_percent", "asg"),
    "realKnownCause/ec2_request_latency_system_failure.csv": ("request_latency", "ec2"),
    "realKnownCause/machine_temperature_system_failure.csv": ("machine_temperature", "machine"),
}


def nab_truth(windows_path):
    """NAB's labelled windows as a ground-truth table the scoreboard understands."""
    windows = json.loads(Path(windows_path).read_text(encoding="utf-8"))
    rows = []
    for key, (metric, entity) in DATASETS.items():
        for i, (start, end) in enumerate(windows[key], 1):
            rows.append({"type": f"{metric} window {i}", "start": pd.Timestamp(start), "end": pd.Timestamp(end),
                         "series": f"{metric}:{entity}", "is_anomaly": True})
    truth = pd.DataFrame(rows)
    truth["series_set"] = truth["series"].apply(lambda s: {s})
    return truth


def _min_points(profile, metric):
    base = profile["detector_params"]["seasonal"]["min_points"]
    return profile.get("overrides", {}).get(metric, {}).get("seasonal", {}).get("min_points", base)


def static_limits(metrics, profile, k=3.0):
    """The classic static baseline: mean +/- k standard deviations of the clean training period."""
    limits = {}
    for metric, g in metrics.groupby("metric"):
        days = profile["overrides"][metric]["seasonal"]["train_days"]
        g = g.sort_values("timestamp")
        train = g[g["timestamp"] < g["timestamp"].iloc[0] + pd.Timedelta(days=days)]["value"]
        limits[metric] = {"upper": train.mean() + k * train.std(), "lower": train.mean() - k * train.std()}
    return limits


def run_setups(metrics, info, profile):
    limits = static_limits(metrics, profile)
    static = pd.concat([
        detect_threshold(metrics[metrics["metric"] == m], {m: limits[m]}, min_points=_min_points(profile, m))
        for m in limits
    ], ignore_index=True)
    seasonal_only = {**profile, "detectors": {t: ["seasonal"] for t in profile["detectors"]}}
    return {
        "1 static 3-sigma": static,
        "2 seasonal only": run_detectors(metrics, info, seasonal_only),
        "3 seasonal + drift": run_detectors(metrics, info, profile),
    }


def score_datasets(setups, truth):
    rows = []
    for metric, entity in DATASETS.values():
        t = truth[truth["series"].str.startswith(f"{metric}:")]
        for name, alerts in setups.items():
            a = alerts[alerts["metric"] == metric]
            summary, problems, _ = score(t, a)
            hits = summary["alerts_total"] - summary["false_alarms"]
            rows.append({
                "dataset": metric, "setup": name,
                "windows found": f"{summary['true_detections']}/{len(t)}",
                "alerts": summary["alerts_total"],
                "alerts outside windows": summary["false_alarms"],
                "alerts inside windows %": round(100 * hits / summary["alerts_total"]) if summary["alerts_total"] else None,
                "first hit (h after window start)": (round(problems["minutes_to_detect"].dropna().min() / 60, 1)
                                                     if problems["found"].any() else None),
            })
    return pd.DataFrame(rows)


def main():
    download()
    print(build(PROFILE, str(REAL), str(DB)))
    profile = load_profile(PROFILE)

    con = duckdb.connect(str(DB), read_only=True)
    metrics = con.execute("SELECT timestamp, entity, metric, value FROM metrics").df()
    info = con.execute("SELECT metric, type FROM metric_info").df()
    con.close()

    setups = run_setups(metrics, info, profile)
    table = score_datasets(setups, nab_truth(REAL / "nab_raw" / "combined_windows.json"))
    table.to_csv(REAL / "clean" / "results.csv", index=False)

    print("\n=== REAL DATA (NAB): detectors vs labelled windows ===")
    print(table.to_string(index=False))
    print("\nReading it: NAB windows are days wide and labels are incomplete, so 'found' is lenient and "
          "'alerts outside windows' over-counts real-but-unlabelled problems.")
    return table, setups


if __name__ == "__main__":
    main()
