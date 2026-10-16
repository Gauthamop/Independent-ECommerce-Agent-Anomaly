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


ONSET_TOLERANCE_MIN = 60  # an estimated onset may be a bit early, so accept changes this much after it


def estimated_onset(state):
    """The onset (ISO text) found by an estimate_onset call, or None."""
    for e in _calls(state, "estimate_onset"):
        if e["result"].get("onset"):
            return e["result"]["onset"]
    return None


def onset_args(inc):
    start, _ = incident_window(inc)
    return {"metric": inc["metric"], "entity": inc["entity"], "detected_at": _iso(start),
            "lookback_hours": GRADUAL_LOOKBACK_H}


def changes_args(state, hours):
    """Arguments for get_recent_changes. A slow problem is noticed late, so once its onset has been
    estimated we look for changes around THAT time, not around the late first alert."""
    onset = estimated_onset(state)
    if onset:
        return {"before": onset, "lookback_hours": hours, "tolerance_minutes": ONSET_TOLERANCE_MIN}
    start, _ = incident_window(state["incident"])
    return {"before": _iso(start), "lookback_hours": hours}


def rule_planner(state):
    inc = state["incident"]
    start, win_end = incident_window(inc)

    if not _calls(state, "query_metric"):
        return {"tool": "query_metric", "args": {
            "metric": inc["metric"], "entity": inc["entity"], "start": _iso(start), "end": _iso(win_end)}}

    if not _calls(state, "compare_segments"):
        return {"tool": "compare_segments", "args": {
            "metric": inc["metric"], "start": _iso(start), "end": _iso(win_end)}}

    if is_gradual(inc) and not _calls(state, "estimate_onset"):
        return {"tool": "estimate_onset", "args": onset_args(inc)}

    changes = _calls(state, "get_recent_changes")
    if not changes:
        hours = GRADUAL_LOOKBACK_H if is_gradual(inc) else ABRUPT_LOOKBACK_H
        return {"tool": "get_recent_changes", "args": changes_args(state, hours)}

    last = changes[-1]["result"]
    if last["n_found"] == 0 and last["lookback_hours"] < WIDE_LOOKBACK_H:
        return {"tool": "get_recent_changes", "args": changes_args(state, WIDE_LOOKBACK_H)}

    return None
