"""Group raw alerts into incidents, so the agent investigates a problem once, not 13 times.

Alerts on the same series that overlap or sit close together (gap_minutes) are one incident.
Different detectors on the same series merge too.
"""
import pandas as pd


def group_alerts(alerts, gap_minutes=120):
    gap = pd.Timedelta(minutes=gap_minutes)
    incidents = []
    for (entity, metric), g in alerts.groupby(["entity", "metric"]):
        g = g.sort_values("start")
        current = None
        for a in g.itertuples():
            if current is not None and a.start <= current["end"] + gap:
                current["end"] = max(current["end"], a.end)
                current["detected_at"] = min(current["detected_at"], a.detected_at)
                current["detectors"].add(a.detector)
                current["n_alerts"] += 1
            else:
                if current is not None:
                    incidents.append(current)
                current = {"entity": entity, "metric": metric, "start": a.start, "end": a.end,
                           "detected_at": a.detected_at, "detectors": {a.detector}, "n_alerts": 1}
        if current is not None:
            incidents.append(current)

    out = []
    for i, inc in enumerate(sorted(incidents, key=lambda x: x["start"]), start=1):
        out.append({
            "incident_id": f"I{i}",
            "entity": inc["entity"], "metric": inc["metric"],
            "start": pd.Timestamp(inc["start"]), "end": pd.Timestamp(inc["end"]),
            "detected_at": pd.Timestamp(inc["detected_at"]),
            "detectors": sorted(inc["detectors"]), "n_alerts": inc["n_alerts"],
        })
    return out
