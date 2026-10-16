"""Investigate every incident in the `anomalies` table and print a report for each.

Usage (from project root, with .venv active):
    python -m src.anomaly_agent.agent.run_agent
Run run_compare first, so the `anomalies` table is filled.
"""
import json
from pathlib import Path

import duckdb

from src.anomaly_agent.agent.graph import investigate
from src.anomaly_agent.agent.incidents import group_alerts
from src.anomaly_agent.agent.report import Report
from src.anomaly_agent.tools.metric_tools import MetricTools


def main(db_path="data/clean/anomaly.duckdb", out_path="data/clean/reports.json", max_steps=6):
    con = duckdb.connect(db_path, read_only=True)
    alerts = con.execute("SELECT * FROM anomalies").df()
    con.close()

    incidents = group_alerts(alerts)
    print(f"{len(alerts)} alerts -> {len(incidents)} incidents\n")

    tools = MetricTools(db_path)
    saved = []
    for inc in incidents:
        report, evidence = investigate(inc, tools, max_steps=max_steps)
        print(Report(**report).to_text(), "\n")
        saved.append({"incident": {k: str(v) for k, v in inc.items()}, "report": report})

    Path(out_path).write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print(f"saved {len(saved)} reports to {out_path}")


if __name__ == "__main__":
    main()
