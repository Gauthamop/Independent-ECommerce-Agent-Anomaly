"""Run the threshold detector on the clean DuckDB data and print the scoreboard.

Usage (from project root, with .venv active):
    python -m src.anomaly_agent.evaluation.run_threshold
"""
import duckdb

from src.anomaly_agent.adapters.csv_adapter import load_profile
from src.anomaly_agent.detectors.threshold import detect_threshold
from src.anomaly_agent.evaluation.scoreboard import load_truth, print_scoreboard, score


def main(profile_path="profiles/ecommerce.yaml", db_path="data/clean/anomaly.duckdb",
         truth_path="data/ground_truth/ground_truth.csv", min_points=3):
    limits = load_profile(profile_path)["thresholds"]

    con = duckdb.connect(db_path)
    metrics = con.execute("SELECT timestamp, entity, metric, value FROM metrics").df()

    alerts = detect_threshold(metrics, limits, min_points=min_points)

    # kept apart from `anomalies`, which run_compare.py fills with the final system's alerts
    con.register("alerts", alerts)
    con.execute("CREATE OR REPLACE TABLE anomalies_threshold AS SELECT * FROM alerts")
    con.close()

    print(alerts.to_string(index=False))
    summary, problems, false_alarms = score(load_truth(truth_path), alerts)
    print_scoreboard("threshold", summary, problems, false_alarms)
    return summary


if __name__ == "__main__":
    main()
