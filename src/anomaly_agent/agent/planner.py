"""The planner decides the NEXT tool call from what the agent already knows.

This one is rule-based, so the agent runs offline and is easy to test. An LLM planner can replace
it later: any function `planner(state) -> {"tool": ..., "args": {...}} | None` works.
Returning None means "I have enough evidence, write the report".
"""
import pandas as pd

WINDOW_CAP_HOURS = 6   # only look at the first hours of an incident (long ones overlap their own baseline)
ABRUPT_LOOKBACK_H = 6  # a sudden problem is usually caused by something just before it
WIDE_LOOKBACK_H = 48   # if nothing is found, look wider once before giving up
GRADUAL_LOOKBACK_H = 72  # slow problems are noticed late, so look further back


def incident_window(incident):
    start = pd.Timestamp(incident["start"])
    end = min(pd.Timestamp(incident["end"]), start + pd.Timedelta(hours=WINDOW_CAP_HOURS))
    return start, end + pd.Timedelta(minutes=5)  # end is exclusive in the tools


def is_gradual(incident):
    return "drift" in incident["detectors"]


def _iso(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def _calls(state, tool):
    return [e for e in state["evidence"] if e["tool"] == tool]


def rule_planner(state):
    inc = state["incident"]
    start, win_end = incident_window(inc)

    if not _calls(state, "query_metric"):
        return {"tool": "query_metric", "args": {
            "metric": inc["metric"], "entity": inc["entity"], "start": _iso(start), "end": _iso(win_end)}}

    if not _calls(state, "compare_segments"):
        return {"tool": "compare_segments", "args": {
            "metric": inc["metric"], "start": _iso(start), "end": _iso(win_end)}}

    changes = _calls(state, "get_recent_changes")
    if not changes:
        hours = GRADUAL_LOOKBACK_H if is_gradual(inc) else ABRUPT_LOOKBACK_H
        return {"tool": "get_recent_changes", "args": {"before": _iso(start), "lookback_hours": hours}}

    last = changes[-1]["result"]
    if last["n_found"] == 0 and last["lookback_hours"] < WIDE_LOOKBACK_H:
        return {"tool": "get_recent_changes", "args": {"before": _iso(start), "lookback_hours": WIDE_LOOKBACK_H}}

    return None
