"""Agent tools: plain Python over the clean DuckDB. Read-only, parameterized SQL, small results.

Three tools answer the three questions an investigator asks:
  query_metric      -> "what did this series do?"
  compare_segments  -> "who is affected, and who is not?"
  get_recent_changes-> "what changed just before?"

Rules: the database is opened read-only, every value goes in as a SQL parameter (never pasted
into the SQL text), and results are short, rounded, and JSON-friendly so an agent can read them.
No business words here: only metric / entity / tags / changes from the canonical tables.
"""
import json
import math

import duckdb
import numpy as np
import pandas as pd


def _ts(value):
    """Accept ISO strings or Timestamps."""
    return pd.Timestamp(value)


def _iso(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def _num(x, digits=4):
    return None if x is None or pd.isna(x) else round(float(x), digits)


class MetricTools:
    def __init__(self, db_path):
        self.db_path = str(db_path)

    def _query(self, sql, params=()):
        con = duckdb.connect(self.db_path, read_only=True)  # cannot write, by construction
        try:
            return con.execute(sql, list(params)).df()
        finally:
            con.close()

    # ---- tool 1 -------------------------------------------------------------------------
    def query_metric(self, metric, entity, start, end, max_points=12):
        """Summary and a coarse series of one metric for one entity in [start, end)."""
        start, end = _ts(start), _ts(end)
        df = self._query(
            "SELECT timestamp, value FROM metrics "
            "WHERE metric = ? AND entity = ? AND timestamp >= ? AND timestamp < ? "
            "ORDER BY timestamp",
            [metric, entity, start.to_pydatetime(), end.to_pydatetime()],
        )
        out = {"metric": metric, "entity": entity, "start": _iso(start), "end": _iso(end),
               "n_points": len(df)}
        if df.empty:
            out["note"] = "no data in this window"
            return out

        # shrink to at most max_points buckets so the result stays small
        size = int(np.ceil(len(df) / max_points))
        bucket = np.arange(len(df)) // size
        series = df.groupby(bucket).agg(time=("timestamp", "first"), mean=("value", "mean"))
        out.update({
            "mean": _num(df["value"].mean()), "min": _num(df["value"].min()),
            "max": _num(df["value"].max()), "first": _num(df["value"].iloc[0]),
            "last": _num(df["value"].iloc[-1]),
            "series": [{"time": _iso(r.time), "mean": _num(r.mean)} for r in series.itertuples()],
        })
        return out

    # ---- tool 2 -------------------------------------------------------------------------
    def compare_segments(self, metric, start, end, baseline_shift_hours=24, top_n=5,
                         material_pct=3.0):
        """Compare every entity of a metric in [start, end) with the same window shifted back.

        Shifting back by a whole day keeps the daily wave out of the comparison.
        An entity counts as "changed" when it moved at least material_pct AND at least 5x the
        typical move of the other entities (so normal noise does not count).
        """
        start, end = _ts(start), _ts(end)
        shift = pd.Timedelta(hours=baseline_shift_hours)
        b_start, b_end = start - shift, end - shift
        df = self._query(
            "SELECT entity, tags, "
            "  avg(CASE WHEN timestamp >= ? AND timestamp < ? THEN value END) AS window_mean, "
            "  avg(CASE WHEN timestamp >= ? AND timestamp < ? THEN value END) AS baseline_mean "
            "FROM metrics WHERE metric = ? AND timestamp >= ? AND timestamp < ? "
            "GROUP BY entity, tags",
            [start.to_pydatetime(), end.to_pydatetime(), b_start.to_pydatetime(),
             b_end.to_pydatetime(), metric, min(start, b_start).to_pydatetime(),
             max(end, b_end).to_pydatetime()],
        )
        out = {"metric": metric, "window": [_iso(start), _iso(end)],
               "baseline_window": [_iso(b_start), _iso(b_end)], "n_entities": len(df)}
        if df.empty:
            out["note"] = "no data"
            return out

        change = df["window_mean"] - df["baseline_mean"]
        denom = df["baseline_mean"].abs().replace(0, np.nan)
        df["change_pct"] = 100 * change / denom
        df = df.dropna(subset=["change_pct"]).copy()
        df["abs_pct"] = df["change_pct"].abs()
        df = df.sort_values("abs_pct", ascending=False)

        rest = df["abs_pct"].iloc[1:].median()
        rest = 0.0 if pd.isna(rest) else float(rest)
        limit = max(material_pct, 5 * rest)
        out["rest_median_abs_change_pct"] = _num(rest, 1)
        out["changed_threshold_pct"] = _num(limit, 1)
        out["n_changed"] = int((df["abs_pct"] >= limit).sum())
        out["top"] = [
            {"entity": r.entity, "tags": json.loads(r.tags) if r.tags else {},
             "window_mean": _num(r.window_mean), "baseline_mean": _num(r.baseline_mean),
             "change_pct": _num(r.change_pct, 1)}
            for r in df.head(top_n).itertuples()
        ]
        return out

    # ---- tool 3 -------------------------------------------------------------------------
    def get_recent_changes(self, before, lookback_hours=6, limit=5, tolerance_minutes=0):
        """Changes (deploys, config edits) in the window before a time, closest first.

        tolerance_minutes: also accept changes up to this long AFTER `before`. Useful when `before` is
        an estimate (see estimate_onset) that might be a little early. Such a change counts as
        0 minutes before (it happened at the same time, as far as the estimate can tell).
        """
        before = _ts(before)
        since = before - pd.Timedelta(hours=lookback_hours)
        until = before + pd.Timedelta(minutes=tolerance_minutes)
        df = self._query(
            "SELECT timestamp, service, change_type, description FROM changes "
            "WHERE timestamp <= ? AND timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
            [until.to_pydatetime(), since.to_pydatetime(), int(limit)],
        )
        return {
            "before": _iso(before), "lookback_hours": lookback_hours, "tolerance_minutes": tolerance_minutes,
            "n_found": len(df),
            "changes": [
                {"time": _iso(r.timestamp), "service": r.service, "change_type": r.change_type,
                 "description": r.description,
                 "minutes_before": max(0, int((before - pd.Timestamp(r.timestamp)).total_seconds() // 60))}
                for r in df.itertuples()
            ],
        }

    # ---- tool 4 -------------------------------------------------------------------------
    def estimate_onset(self, metric, entity, detected_at, lookback_hours=72, allowance=0.25):
        """Estimate when a SLOW change really began. Its first alert comes late: a leak is noticed
        hours after it starts, so matching changes against the alert time would look too far away.

        Method (CUSUM): take the day-over-day change d(t) = x(t) - x(t - 24h), which removes the daily
        wave. Learn the normal size of d from the early, assumed-quiet half of the window. Then run a
        one-sided CUSUM (it only adds up changes bigger than `allowance` x the normal wobble) up to the
        alert. The estimate is the LAST moment the CUSUM was still at zero: after that, the change kept
        building. It tends to be a little late for very slow ramps and is close for sudden steps.
        """
        detected = _ts(detected_at)
        start = detected - pd.Timedelta(hours=lookback_hours + 24)
        df = self._query(
            "SELECT timestamp, value FROM metrics WHERE metric = ? AND entity = ? "
            "AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
            [metric, entity, start.to_pydatetime(), detected.to_pydatetime()],
        )
        out = {"metric": metric, "entity": entity, "detected_at": _iso(detected), "onset": None,
               "method": "CUSUM on the day-over-day change"}
        if df.empty:
            out["note"] = "no data"
            return out
        s = df.set_index("timestamp")["value"]
        d = (s - s.shift(freq="24h")).dropna()
        d = d[d.index >= detected - pd.Timedelta(hours=lookback_hours)]
        half = d[d.index < d.index[0] + (detected - d.index[0]) / 2] if len(d) else d
        if len(half) < 20:
            out["note"] = "not enough history to estimate an onset"
            return out
        mu0 = half.median()
        sigma = max(1.4826 * (half - mu0).abs().median(), 1e-9)

        total, zero_at = 0.0, d.index[0]
        for t, v in d.items():
            total = max(0.0, total + (v - mu0) - allowance * sigma)
            if total == 0.0:
                zero_at = t
        # Pure noise leaves the running sum above zero by chance. With allowance k (in wobbles), the chance
        # it exceeds h wobbles is about exp(-2*k*h); asking for ~1% false alarms gives h = ln(100) / (2*k)
        # (10 wobbles for k = 0.25). A real build-up is far larger than that.
        sustained = total >= math.log(100) / (2 * allowance) * sigma
        onset = zero_at if sustained else detected
        out.update({
            "onset": _iso(onset),
            "hours_before_alert": round((detected - onset).total_seconds() / 3600, 1),
            "normal_daily_change": _num(mu0), "normal_wobble": _num(sigma),
            "note": ("the series was already drifting before the alert" if sustained and onset < detected
                     else "no earlier sustained change found: the alert time is the start"),
        })
        return out


ALLOWED_TOOLS = ("query_metric", "compare_segments", "get_recent_changes", "estimate_onset")
