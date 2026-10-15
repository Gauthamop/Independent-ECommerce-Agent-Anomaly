"""Run every detector setup on the clean data and compare them on one scoreboard.

Usage (from project root, with .venv active):
    python -m src.anomaly_agent.evaluation.run_compare
Writes the final system's alerts (registry) to the `anomalies` table.
"""
import duckdb
import pandas as pd

from src.anomaly_agent.adapters.csv_adapter import load_profile
from src.anomaly_agent.detectors.registry import run_detectors
from src.anomaly_agent.detectors.threshold import detect_threshold
from src.anomaly_agent.evaluation.scoreboard import load_truth, print_scoreboard, score


def main(profile_path="profiles/ecommerce.yaml", db_path="data/clean/anomaly.duckdb",
         truth_path="data/ground_truth/ground_truth.csv"):
    profile = load_profile(profile_path)

    con = duckdb.connect(db_path)
    metrics = con.execute("SELECT timestamp, entity, metric, value FROM metrics").df()
    info = con.execute("SELECT metric, type FROM metric_info").df()

    seasonal_only = {**profile, "detectors": {t: ["seasonal"] for t in profile["detectors"]}}
    setups = {
        "1 threshold": detect_threshold(metrics, profile["thresholds"], min_points=3),
        "2 seasonal only": run_detectors(metrics, info, seasonal_only),
        "3 seasonal + drift": run_detectors(metrics, info, profile),
    }

    truth = load_truth(truth_path)
    table, details = [], {}
    for name, alerts in setups.items():
        summary, problems, false_alarms = score(truth, alerts)
        details[name] = (summary, problems, false_alarms)
        row = {"setup": name, "found": summary["true_detections"], "missed": summary["misses"],
               "false_alarms": summary["false_alarms"], "alerts": summary["alerts_total"]}
        for p in problems.itertuples():
            row[p.problem] = None if pd.isna(p.minutes_to_detect) else round(p.minutes_to_detect)
        table.append(row)

    print("\nMinutes to detect per problem (empty = missed):")
    print(pd.DataFrame(table).to_string(index=False))

    final = setups["3 seasonal + drift"]
    con.register("final_alerts", final)
    con.execute("CREATE OR REPLACE TABLE anomalies AS SELECT * FROM final_alerts")
    con.close()

    print_scoreboard("3 seasonal + drift", *details["3 seasonal + drift"])
    print("\nfinal alerts:")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
