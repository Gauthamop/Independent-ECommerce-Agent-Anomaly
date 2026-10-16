"""Turn the evidence the agent collected into a report. Every claim cites its evidence ids [E1].

If a claim has no tool result behind it, the report says "not enough evidence" instead.
"""
import math

import pandas as pd
from pydantic import BaseModel

from src.anomaly_agent.agent.planner import is_gradual

MIN_CAUSE_SCORE = 0.05     # weaker candidates are not worth listing


class Cause(BaseModel):
    description: str
    score: float
    evidence_ids: list[str]
    service: str | None = None       # the change's service; None for "no change found"
    change_time: str | None = None   # the change's time, "YYYY-MM-DD HH:MM"


class Report(BaseModel):
    incident_id: str
    what_happened: str
    since_when: str
    who_affected: str
    top_causes: list[Cause]
    confidence: str
    suggested_action: str
    evidence: list[str]
    notes: list[str]

    def to_text(self):
        lines = [
            f"=== INCIDENT {self.incident_id} ===",
            f"What happened : {self.what_happened}",
            f"Since when    : {self.since_when}",
            f"Who is hit    : {self.who_affected}",
            "Top causes    :",
        ]
        for i, c in enumerate(self.top_causes, 1):
            lines.append(f"  {i}. {c.description} (score {c.score}) {' '.join(f'[{e}]' for e in c.evidence_ids)}")
        lines += [f"Confidence    : {self.confidence}", f"Suggested     : {self.suggested_action}", "Evidence      :"]
        lines += [f"  {e}" for e in self.evidence]
        if self.notes:
            lines += ["Notes         :"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


def _first(state, tool):
    for e in state["evidence"]:
        if e["tool"] == tool:
            return e
    return None


def _level(x):
    return "high" if x >= 0.7 else "medium" if x >= 0.35 else "low"


def _summary(e):
    r = e["result"]
    args = ", ".join(f"{k}={v}" for k, v in e["args"].items())
    if e["tool"] == "query_metric":
        detail = f"n_points={r['n_points']}, mean={r.get('mean')}, min={r.get('min')}, max={r.get('max')}"
    elif e["tool"] == "compare_segments":
        detail = f"{r['n_entities']} series, {r.get('n_changed')} changed"
    else:
        detail = f"{r['n_found']} change(s) found"
    return f"[{e['id']}] {e['tool']}({args}) -> {detail}"


def build_report(state):
    inc = state["incident"]
    e_q, e_c = _first(state, "query_metric"), _first(state, "compare_segments")
    change_ev = [e for e in state["evidence"] if e["tool"] == "get_recent_changes"]
    gradual = is_gradual(inc)
    notes = []

    # ---- what happened --------------------------------------------------------------
    own = None
    if e_c:
        own = next((t for t in e_c["result"].get("top", []) if t["entity"] == inc["entity"]), None)
    if e_q and e_q["result"]["n_points"] > 0:
        q = e_q["result"]
        up = own is None or own["change_pct"] >= 0
        extreme = q["max"] if up else q["min"]
        what = f"{inc['metric']} for {inc['entity']} averaged {q['mean']} in the incident window (extreme {extreme}) [{e_q['id']}]"
        if own:
            what += (f", against {own['baseline_mean']} at the same time a day earlier "
                     f"({own['change_pct']:+.0f}%) [{e_c['id']}]")
    else:
        what = "not enough evidence: no data was retrieved for this incident"

    since = (f"{pd.Timestamp(inc['start'])} UTC (first alert), detected {pd.Timestamp(inc['detected_at'])} UTC "
             f"by {', '.join(inc['detectors'])}")
    if gradual:
        since += "; this is a slow change, so it probably began earlier than the first alert"

    # ---- who is affected ------------------------------------------------------------
    localized = False
    if e_c and e_c["result"].get("top"):
        r = e_c["result"]
        limit = r["changed_threshold_pct"]
        hit = [t for t in r["top"] if abs(t["change_pct"]) >= limit]
        names = ", ".join(f"{t['entity']} {t['tags'] or ''} ({t['change_pct']:+.0f}%)".replace("  ", " ") for t in hit)
        localized = r["n_changed"] == 1
        who = f"{r['n_changed']} of {r['n_entities']} series changed (by {limit}% or more): {names or 'none'}"
        if localized:
            who += f". The others look normal (median change {r['rest_median_abs_change_pct']}%)"
        who += f" [{e_c['id']}]"
    else:
        who = "not enough evidence: the segments were not compared"

    # ---- causes ---------------------------------------------------------------------
    candidates = {}
    for e in change_ev:
        for ch in e["result"]["changes"]:
            scale = 24 * 60 if gradual else 60  # minutes: how fast "just before" fades
            score = math.exp(-ch["minutes_before"] / scale)
            key = (ch["time"], ch["service"])
            if score >= MIN_CAUSE_SCORE and (key not in candidates or candidates[key][0] < score):
                desc = (f"{ch['service']} {ch['change_type']} at {ch['time']}: {ch['description']} "
                        f"({ch['minutes_before']} min before onset)")
                candidates[key] = (score, desc, e["id"], ch["service"], ch["time"])
    causes = [Cause(description=d, score=round(s, 2), evidence_ids=[eid], service=svc, change_time=t)
              for s, d, eid, svc, t in sorted(candidates.values(), reverse=True)[:3]]

    if not change_ev:
        notes.append("Recent changes were not checked (step limit reached), so no cause can be given: not enough evidence.")
        confidence = "low"
    elif not causes:
        causes = [Cause(description="No recent change explains this; it is probably an external cause "
                                    "(not enough evidence for an internal one)",
                        score=0.0, evidence_ids=[e["id"] for e in change_ev])]
        confidence = "low"
    else:
        confidence = _level(causes[0].score * (1.0 if localized else 0.7))

    missing = [t for t, e in (("query_metric", e_q), ("compare_segments", e_c)) if e is None]
    if missing and change_ev:
        notes.append(f"Not checked: {', '.join(missing)}.")

    # ---- suggested action -----------------------------------------------------------
    top = causes[0] if causes else None
    if top is None:
        action = "Evidence is incomplete. Run the investigation again with more steps before acting."
    elif top.score > 0:
        action = (f"Review or roll back the top candidate ({top.description.split(': ')[0]}), "
                  f"then watch {inc['metric']} for {inc['entity']}.")
    else:
        action = (f"No related change found. Check external factors for {inc['entity']} and escalate "
                  f"for manual review; do not roll anything back on this evidence.")

    return Report(incident_id=inc["incident_id"], what_happened=what, since_when=since,
                  who_affected=who, top_causes=causes, confidence=confidence,
                  suggested_action=action, evidence=[_summary(e) for e in state["evidence"]],
                  notes=notes)
