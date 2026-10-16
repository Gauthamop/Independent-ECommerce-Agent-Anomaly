"""Anomaly Agent explorer: look at every step of the pipeline in one place.

Run from the project root, with .venv active:
    streamlit run src/anomaly_agent/app/app.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # so `src.anomaly_agent...` imports work under streamlit

import altair as alt  # noqa: E402
import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.anomaly_agent.adapters.csv_adapter import build, load_profile  # noqa: E402
from src.anomaly_agent.agent.graph import investigate  # noqa: E402
from src.anomaly_agent.agent.incidents import group_alerts  # noqa: E402
from src.anomaly_agent.agent.report import Report  # noqa: E402
from src.anomaly_agent.detectors.registry import run_detectors  # noqa: E402
from src.anomaly_agent.detectors.threshold import detect_threshold  # noqa: E402
from src.anomaly_agent.evaluation.agent_eval import _without_change, evaluate, summarize  # noqa: E402
from src.anomaly_agent.evaluation.scoreboard import load_truth, score  # noqa: E402
from src.anomaly_agent.generator.make_raw import write_raw  # noqa: E402
from src.anomaly_agent.tools.metric_tools import MetricTools  # noqa: E402

DB = ROOT / "data" / "clean" / "anomaly.duckdb"
RAW = ROOT / "data" / "raw"
TRUTH = ROOT / "data" / "ground_truth" / "ground_truth.csv"
PROFILE = ROOT / "profiles" / "ecommerce.yaml"

# validated categorical order (fixed: a series keeps its colour whatever else is on screen)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DETECTOR_COLOR = {"seasonal": SERIES[0], "drift": SERIES[1], "threshold": SERIES[2]}
MUTED = "#898781"

alt.data_transformers.disable_max_rows()


# ----------------------------------------------------------------------------- data loading
def have_data():
    return DB.exists() and TRUTH.exists()


@st.cache_data
def load_core(db_mtime):
    con = duckdb.connect(str(DB), read_only=True)
    tables = set(con.execute("SHOW TABLES").df()["name"])
    out = {
        "metrics": con.execute("SELECT timestamp, entity, metric, value, tags FROM metrics").df(),
        "info": con.execute("SELECT metric, type FROM metric_info").df(),
        "changes": con.execute("SELECT * FROM changes ORDER BY timestamp").df(),
        "anomalies": con.execute("SELECT * FROM anomalies").df() if "anomalies" in tables else pd.DataFrame(),
    }
    con.close()
    return out


@st.cache_data
def load_truth_cached(mtime):
    return load_truth(TRUTH)


def rebuild_everything():
    """make raw files -> adapter -> detectors -> anomalies table (what the command line steps do)."""
    write_raw(ROOT / "data")
    build(str(PROFILE), str(RAW), str(DB))
    profile = load_profile(PROFILE)
    con = duckdb.connect(str(DB))
    metrics = con.execute("SELECT timestamp, entity, metric, value FROM metrics").df()
    info = con.execute("SELECT metric, type FROM metric_info").df()
    alerts = run_detectors(metrics, info, profile)
    con.register("alerts", alerts)
    con.execute("CREATE OR REPLACE TABLE anomalies AS SELECT * FROM alerts")
    con.close()
    st.cache_data.clear()


def stat_row(items):
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.metric(label, value)


# ----------------------------------------------------------------------------- page: overview
def page_overview(core, truth, profile):
    st.header("Overview")
    st.write(
        "This app shows every step of the anomaly pipeline. Pick a page on the left. "
        "Everything here is **fake data we generated**, so we know the right answers."
    )
    st.code(
        "fake world (world.py) -> messy raw files (make_raw.py)\n"
        "   -> adapter + profile -> clean DuckDB tables (metrics, changes)\n"
        "   -> detectors -> alerts -> incidents\n"
        "   -> LangGraph agent (tools: query_metric, compare_segments, get_recent_changes) -> report\n"
        "   -> scoreboard / agent evaluation (the only code that reads the answer key)",
        language="text",
    )
    metrics, anomalies = core["metrics"], core["anomalies"]
    incidents = group_alerts(anomalies) if len(anomalies) else []
    stat_row([
        ("Metric rows", f"{len(metrics):,}"),
        ("Series", f"{metrics.groupby(['entity', 'metric']).ngroups}"),
        ("Alerts", f"{len(anomalies)}"),
        ("Incidents", f"{len(incidents)}"),
        ("Changes logged", f"{len(core['changes'])}"),
    ])

    st.subheader("The four metrics")
    series = (metrics.groupby("metric").agg(series=("entity", "nunique"), rows=("value", "size"),
                                            example_entity=("entity", "first")).reset_index()
              .merge(core["info"], on="metric"))
    st.dataframe(series[["metric", "type", "series", "rows", "example_entity"]], hide_index=True,
                 width="stretch")
    st.caption("type: count = how many, rate = a share between 0 and 1, gauge = a level that goes up and down. "
               "The detector registry picks detectors by this type.")

    st.subheader("What we planted in the data (the answer key)")
    st.dataframe(truth[["problem_id", "type", "start", "end", "series", "is_anomaly", "true_cause"]],
                 hide_index=True, width="stretch")
    st.caption("`is_anomaly = False` is the sale day: it looks scary but is normal. Detectors never read this table.")

    st.subheader("Things to try")
    st.markdown(
        "1. **Explore metrics**: show `orders` and find the sale day. Then show `decline_rate` for all 6 series "
        "and find the one that breaks. Switch resolution to *1 hour*: what happens to the short fraud burst?\n"
        "2. **Detectors**: lower `z_threshold` and watch false alarms appear. Raise the hand-set limits and watch "
        "the threshold detector miss things.\n"
        "3. **Agent**: pick the memory-leak incident and hide the true change from the log. Does the agent blame "
        "something else? How confident is it?\n"
        "4. **Evaluation**: see which cases pass and why."
    )


# ----------------------------------------------------------------------------- page: data flow
def page_flow(core, truth, profile):
    st.header("How messy files become clean tables")
    st.write("The adapter reads the profile (`profiles/ecommerce.yaml`) and turns each raw file into the same "
             "tall table: `timestamp, entity, metric, value, tags`. All times become UTC.")

    st.subheader("1. Raw files (deliberately messy)")
    for src in profile["sources"]:
        path = RAW / src["file"]
        with st.expander(f"{src['file']}  (time column: `{src['time_col']}`, format `{src['time_format']}`)", expanded=False):
            if path.exists():
                raw = pd.read_csv(path, nrows=4)
                st.dataframe(raw, hide_index=True, width="stretch")
                st.caption(f"{len(raw.columns)} columns. Times are India local time (IST).")
            for rule in src["columns"]:
                st.code(f"column pattern: {rule['pattern']}\n-> metric: {rule['metric']}, entity: {rule['entity']}",
                        language="text")
    with st.expander(f"{profile['changes']['file']} (the change log)"):
        st.dataframe(pd.read_csv(RAW / profile["changes"]["file"]), hide_index=True, width="stretch")

    st.subheader("2. Clean `metrics` table")
    metric = st.selectbox("Show a sample for", sorted(core["metrics"]["metric"].unique()), key="flow_metric")
    st.dataframe(core["metrics"].query("metric == @metric").head(6), hide_index=True, width="stretch")
    st.caption("`entity` is built from the column name (e.g. `dclnrt_bank_a_north` -> `bank_a/north`). "
               "`tags` keeps the parts, so the agent can filter by them.")

    st.subheader("3. Clean `changes` table (UTC)")
    st.dataframe(core["changes"], hide_index=True, width="stretch")

    with st.expander("The profile (the only place with business words)"):
        st.code(PROFILE.read_text(encoding="utf-8"), language="yaml")


# ----------------------------------------------------------------------------- page: explore
# which alerts the Explore page can draw -> the setup name used by compute_setups
ALERT_SOURCES = {
    "Final system (seasonal + drift)": "3 seasonal + drift",
    "Threshold only (hand-set limits)": "1 threshold",
    "Seasonal only": "2 seasonal only",
}


@st.cache_data
def default_alert_sets(db_mtime):
    """Alerts of every setup with the profile's own settings (first call takes a few seconds)."""
    profile = load_profile(PROFILE)
    seasonal = profile["detector_params"]["seasonal"]
    limits = tuple((m, float(s["upper"])) for m, s in profile["thresholds"].items())
    results = compute_setups(db_mtime, float(seasonal["z_threshold"]), int(seasonal["min_points"]),
                             int(seasonal["train_days"]), True, limits)
    return {name: r["alerts"] for name, r in results.items()}


def page_explore(core, truth, profile):
    st.header("Explore the metrics")
    metrics = core["metrics"]

    c1, c2, c3 = st.columns([1, 1, 1])
    metric = c1.selectbox("Metric", sorted(metrics["metric"].unique()))
    all_entities = sorted(metrics.loc[metrics["metric"] == metric, "entity"].unique())
    chosen = c2.multiselect("Series", all_entities, default=all_entities[:3])
    res = c3.selectbox("Resolution", ["5 min", "15 min", "1 hour"], index=1)

    sub = metrics[metrics["metric"] == metric]
    tmin, tmax = sub["timestamp"].min().to_pydatetime(), sub["timestamp"].max().to_pydatetime()
    t0, t1 = st.slider("Time range (UTC)", min_value=tmin, max_value=tmax, value=(tmin, tmax),
                       format="MMM D HH:mm")

    o1, o2, o3, o4 = st.columns(4)
    show_truth = o1.checkbox("Show answer-key windows", value=True)
    show_alerts = o2.checkbox("Show detector alerts", value=True)
    alert_source = o3.selectbox("Alerts from", list(ALERT_SOURCES), disabled=not show_alerts)
    limit = profile["thresholds"].get(metric, {}).get("upper")
    show_limit = o4.checkbox("Show hand-set limit", value=False, disabled=limit is None)

    if alert_source == "Final system (seasonal + drift)":
        anomalies = core["anomaly_view"]
    else:
        anomalies = default_alert_sets(DB.stat().st_mtime)[ALERT_SOURCES[alert_source]]

    if not chosen:
        st.info("Pick at least one series.")
        return

    df = sub[sub["entity"].isin(chosen) & (sub["timestamp"] >= t0) & (sub["timestamp"] <= t1)]
    if df.empty:
        st.info("No data in this time range for the chosen series.")
        return
    minutes = {"5 min": 5, "15 min": 15, "1 hour": 60}[res]
    if minutes > 5:
        # average each series in fixed time buckets (index by time first: the filtered rows are not 0,1,2...)
        df = (df.set_index("timestamp").groupby("entity")["value"].resample(f"{minutes}min").mean()
              .dropna().reset_index())

    # a series keeps its colour by its place among ALL series of this metric
    colors = alt.Scale(domain=all_entities, range=[SERIES[i % len(SERIES)] for i in range(len(all_entities))])
    x = alt.X("timestamp:T", title=None, axis=alt.Axis(grid=False))
    tip = [alt.Tooltip("timestamp:T", title="time", format="%b %d %H:%M"), "entity:N",
           alt.Tooltip("value:Q", format=".4g")]
    line = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=x, y=alt.Y("value:Q", title=metric, scale=alt.Scale(zero=False), axis=alt.Axis(gridOpacity=0.2)),
        color=alt.Color("entity:N", scale=colors, legend=alt.Legend(title=None, orient="top")))
    hover = line.mark_point(size=60, opacity=0).encode(tooltip=tip)
    layers = []

    if show_truth:
        rows = truth[truth["series_set"].apply(lambda s: any(k.startswith(f"{metric}:") for k in s))]
        bands = pd.DataFrame({
            "start": rows["start"], "end": rows["end"],
            "label": [f"{t}{'' if a else ' (normal)'}" for t, a in zip(rows["type"], rows["is_anomaly"])],
        })
        bands = bands[(bands["end"] >= t0) & (bands["start"] <= t1)]
        if len(bands):
            layers.append(alt.Chart(bands).mark_rect(opacity=0.18, color=MUTED).encode(x="start:T", x2="end:T"))
            layers.append(alt.Chart(bands).mark_text(align="left", baseline="top", dx=4, dy=2, color=MUTED)
                          .encode(x="start:T", y=alt.value(0), text="label:N"))
    if show_limit and limit is not None:
        layers.append(alt.Chart(pd.DataFrame({"y": [limit]})).mark_rule(strokeDash=[4, 4], color=MUTED).encode(y="y:Q"))

    top = alt.layer(*layers, line, hover).properties(height=340)
    charts = [top]

    if show_alerts:
        al = anomalies[(anomalies["metric"] == metric) & anomalies["entity"].isin(chosen)
                       & (anomalies["end"] >= t0) & (anomalies["start"] <= t1)] if len(anomalies) else anomalies
        if len(al) == 0:
            st.info(f"No alerts from '{alert_source}' for these series in this time range. "
                    "That is not a bug: the guard saw nothing odd here. "
                    "Try another 'Alerts from' choice to compare guards.")
        else:
            strip = alt.Chart(al).mark_bar(height=10, cornerRadius=2).encode(
                x=alt.X("start:T", title=None), x2="end:T", y=alt.Y("entity:N", title=None),
                color=alt.Color("detector:N", scale=alt.Scale(domain=list(DETECTOR_COLOR), range=list(DETECTOR_COLOR.values())),
                                legend=alt.Legend(title="alerts by", orient="top")),
                tooltip=["detector:N", "entity:N", alt.Tooltip("start:T", format="%b %d %H:%M"),
                         alt.Tooltip("end:T", format="%b %d %H:%M"), alt.Tooltip("peak_value:Q", format=".4g")],
            ).properties(height=26 * max(1, al["entity"].nunique()) + 20)
            charts.append(strip)

    st.altair_chart(alt.vconcat(*charts).resolve_scale(x="shared", color="independent"), width="stretch")
    st.caption("Grey bands are the answer key (a viewer-only overlay; detectors never see it). "
               "Bars below the chart are the alerts from the guard chosen in 'Alerts from' "
               "(blue = seasonal, orange = drift, green = threshold).")
    with st.expander("Table view of the plotted numbers"):
        st.dataframe(df, hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- page: detectors
@st.cache_data
def compute_setups(db_mtime, z, min_points, train_days, pool, limits_tuple):
    core = load_core(db_mtime)
    profile = load_profile(PROFILE)
    profile["thresholds"] = {m: {"upper": v} for m, v in limits_tuple}
    for name in ("seasonal", "drift"):
        profile["detector_params"][name]["min_points"] = min_points
        profile["detector_params"][name]["train_days"] = train_days
    profile["detector_params"]["seasonal"]["z_threshold"] = z
    profile["detector_params"]["seasonal"]["pool_spread"] = pool

    metrics, info = core["metrics"][["timestamp", "entity", "metric", "value"]], core["info"]
    seasonal_only = {**profile, "detectors": {t: ["seasonal"] for t in profile["detectors"]}}
    setups = {
        "1 threshold": detect_threshold(metrics, profile["thresholds"], min_points=min_points),
        "2 seasonal only": run_detectors(metrics, info, seasonal_only),
        "3 seasonal + drift": run_detectors(metrics, info, profile),
    }
    truth = load_truth(TRUTH)
    results = {}
    for name, alerts in setups.items():
        summary, problems, false_alarms = score(truth, alerts)
        results[name] = {"alerts": alerts, "summary": summary, "problems": problems, "false_alarms": false_alarms}
    return results


def page_detectors(core, truth, profile):
    st.header("Detectors and the scoreboard")
    st.write("Three detector setups run on the same clean data. The scoreboard then opens the answer key "
             "and counts true detections, misses and false alarms. Move the sliders and see what changes.")

    seasonal = profile["detector_params"]["seasonal"]
    with st.sidebar:
        st.subheader("Detector settings")
        z = st.slider("seasonal z_threshold (how far from normal)", 2.0, 8.0, float(seasonal["z_threshold"]), 0.5)
        min_points = st.slider("min_points in a row (5 min each)", 1, 6, int(seasonal["min_points"]))
        train_days = st.slider("train_days (learning period)", 4, 14, int(seasonal["train_days"]))
        pool = st.checkbox("pool_spread (steadier spread estimate)", value=True)
        st.caption("Hand-set limits for the threshold detector")
        limits = {}
        for m, spec in profile["thresholds"].items():
            limits[m] = st.number_input(f"{m} upper limit", value=float(spec["upper"]), format="%g")

    results = compute_setups(DB.stat().st_mtime, z, min_points, train_days, pool, tuple(limits.items()))

    rows = []
    for name, r in results.items():
        s = r["summary"]
        row = {"setup": name, "found": s["true_detections"], "missed": s["misses"],
               "false alarms": s["false_alarms"], "alerts": s["alerts_total"]}
        for p in r["problems"].itertuples():
            row[f"{p.problem} (min)"] = None if pd.isna(p.minutes_to_detect) else round(p.minutes_to_detect)
        rows.append(row)
    st.subheader("Scoreboard")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("(min) = minutes from the true start until the detector *knew*. Empty = missed. "
               "Lower is better. Default settings: threshold finds the leak after 7.5 days; seasonal + drift in ~12 hours.")

    pick = st.selectbox("Look closer at", list(results), index=2)
    r = results[pick]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Per problem**")
        st.dataframe(r["problems"], hide_index=True, width="stretch")
    with c2:
        st.markdown("**False alarms**")
        if len(r["false_alarms"]):
            st.dataframe(r["false_alarms"], hide_index=True, width="stretch")
        else:
            st.success("None: every alert overlaps a real problem.")

    alerts = r["alerts"].copy()
    if len(alerts):
        alerts["series"] = alerts["metric"] + " / " + alerts["entity"]
        chart = alt.Chart(alerts).mark_bar(height=12, cornerRadius=2).encode(
            x=alt.X("start:T", title=None), x2="end:T", y=alt.Y("series:N", title=None),
            color=alt.Color("detector:N", scale=alt.Scale(domain=list(DETECTOR_COLOR), range=list(DETECTOR_COLOR.values())),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["detector:N", "series:N", alt.Tooltip("start:T", format="%b %d %H:%M"),
                     alt.Tooltip("end:T", format="%b %d %H:%M"), alt.Tooltip("peak_value:Q", format=".4g")],
        ).properties(height=30 * alerts["series"].nunique() + 40)
        st.markdown("**Alert timeline**")
        st.altair_chart(chart, width="stretch")
        with st.expander("All alerts"):
            st.dataframe(alerts.drop(columns="series"), hide_index=True, width="stretch")
    else:
        st.info("This setup raised no alerts.")


# ----------------------------------------------------------------------------- page: agent
CONF_LABEL = {"high": ":green[● High]", "medium": ":orange[● Medium]", "low": ":red[● Low]"}


def page_agent(core, truth, profile):
    st.header("The investigating agent")
    anomalies = core["anomaly_view"]
    if anomalies.empty:
        st.warning("No alerts yet. Use 'Rebuild data' in the sidebar.")
        return

    incidents = group_alerts(anomalies)
    st.write(f"{len(anomalies)} alerts were grouped into **{len(incidents)} incidents**. "
             "The agent investigates one incident at a time with three read-only tools.")

    labels = {f"{i['incident_id']}: {i['metric']} / {i['entity']} from {i['start']:%b %d %H:%M}": i for i in incidents}
    inc = labels[st.selectbox("Incident", list(labels))]

    c1, c2 = st.columns(2)
    max_steps = c1.slider("Step limit (tool calls)", 1, 8, 6)
    changes = core["changes"]
    options = ["(keep the full change log)"] + [
        f"{r.timestamp:%Y-%m-%d %H:%M} · {r.service} · {r.description}" for r in changes.itertuples()]
    hidden = c2.selectbox("Hide one change from the agent", options)

    with tempfile.TemporaryDirectory() as tmp:
        db = str(DB)
        if hidden != options[0]:
            row = changes.iloc[options.index(hidden) - 1]
            db = _without_change(str(DB), row["service"], row["timestamp"], tmp)
        report, evidence = investigate(inc, MetricTools(db), max_steps=max_steps)
    rep = Report(**report)

    st.subheader(f"Report for {rep.incident_id}")
    st.markdown(f"**Confidence:** {CONF_LABEL[rep.confidence]}")
    st.markdown(f"**What happened:** {rep.what_happened}")
    st.markdown(f"**Since when:** {rep.since_when}")
    st.markdown(f"**Who is affected:** {rep.who_affected}")
    st.markdown("**Top causes:**")
    for c in rep.top_causes:
        st.markdown(f"- {c.description} · score **{c.score}** · evidence {', '.join(c.evidence_ids)}")
        st.progress(min(max(c.score, 0.0), 1.0))
    st.markdown(f"**Suggested action:** {rep.suggested_action}")
    for n in rep.notes:
        st.warning(n)

    st.subheader("Evidence trail (what the agent actually did)")
    st.caption("Plan -> act -> plan -> ... -> report. Every claim above cites one of these results.")
    for e in evidence:
        with st.expander(f"{e['id']} · {e['tool']}"):
            st.markdown("**Arguments**")
            st.json(e["args"])
            st.markdown("**Result**")
            st.json(e["result"])

    if st.checkbox("Reveal the answer key for this incident (the agent never sees it)"):
        hit = truth[truth["series_set"].apply(lambda s: f"{inc['metric']}:{inc['entity']}" in s)
                    & (truth["start"] <= inc["end"]) & (truth["end"] >= inc["start"])]
        if hit.empty:
            st.info("No ground-truth row matches this incident.")
        else:
            st.dataframe(hit[["type", "start", "end", "is_anomaly", "true_cause", "cause_type",
                              "cause_service", "cause_time"]], hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- page: evaluation
@st.cache_data
def run_eval(db_mtime, truth_mtime):
    core = load_core(db_mtime)
    return evaluate(str(DB), core["anomalies"], load_truth(TRUTH))


def page_eval(core, truth, profile):
    st.header("Agent evaluation")
    st.write("Each real problem is investigated and compared with the answer key. "
             "A **cause removed** row deletes the true change from the log: the agent should then "
             "*not* confidently blame something else.")
    if core["anomalies"].empty:
        st.warning("No alerts yet. Use 'Rebuild data' in the sidebar.")
        return
    cases = run_eval(DB.stat().st_mtime, TRUTH.stat().st_mtime)
    summary = summarize(cases)
    stat_row([("Top cause correct", summary["top_cause_correct"]),
              ("Cause removed, handled", summary["cause_removed_handled"]),
              ("Confident & wrong", summary["overconfident_wrong_answers"]),
              ("Claims cited", "yes" if summary["all_claims_cited"] else "NO"),
              ("Cases passed", summary["cases_passed"])])
    view = cases.drop(columns=["in_top3"]).copy()
    view["passed"] = view["passed"].map({True: "✓ pass", False: "✗ fail"})
    st.dataframe(view, hide_index=True, width="stretch")
    st.caption("This is a handful of cases on one synthetic world, so treat it as a sanity check, not a guarantee.")
    with st.expander("The answer key"):
        st.dataframe(truth.drop(columns=["series_set"]), hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- main
PAGES = {
    "Overview": page_overview,
    "How data flows": page_flow,
    "Explore metrics": page_explore,
    "Detectors": page_detectors,
    "Agent": page_agent,
    "Evaluation": page_eval,
}


def main():
    st.set_page_config(page_title="Anomaly Agent Explorer", layout="wide")
    st.sidebar.title("Anomaly Agent")
    if st.sidebar.button("Rebuild data (raw -> clean -> alerts)"):
        with st.spinner("Rebuilding..."):
            rebuild_everything()
        st.sidebar.success("Rebuilt.")

    if not have_data():
        st.title("Anomaly Agent Explorer")
        st.warning("No data yet. Click **Rebuild data** in the sidebar (takes about 20 seconds).")
        return

    page = st.sidebar.radio("Page", list(PAGES))
    core = load_core(DB.stat().st_mtime)
    # the alerts the final system raised (what the agent investigates)
    core = {**core, "anomaly_view": core["anomalies"]}
    truth = load_truth_cached(TRUTH.stat().st_mtime)
    profile = load_profile(PROFILE)
    PAGES[page](core, truth, profile)


if __name__ == "__main__":  # so the LLM / real-data app can import these pages without running this app
    main()
