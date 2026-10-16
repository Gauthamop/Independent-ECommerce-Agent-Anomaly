"""Anomaly Agent explorer, version 2: adds the Groq LLM agent, the rules-vs-LLM evaluation, and a
page for REAL data. The original app (app.py) is separate and unchanged; this one reuses its pages.

Run from the project root, with .venv active:
    streamlit run src/anomaly_agent/app/app_llm.py
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
from src.anomaly_agent.agent.llm_agent import llm_parts  # noqa: E402
from src.anomaly_agent.agent.planner import rule_planner  # noqa: E402
from src.anomaly_agent.agent.report import Report  # noqa: E402
from src.anomaly_agent.app import app as base  # noqa: E402  (the original app's pages and helpers)
from src.anomaly_agent.evaluation.agent_eval import (DECOYS, _with_decoy, _without_change,  # noqa: E402
                                                     evaluate, summarize)
from src.anomaly_agent.evaluation.scoreboard import load_truth  # noqa: E402
from src.anomaly_agent.llm.groq_llm import GroqLLM, get_model  # noqa: E402
from src.anomaly_agent.llm.recording import RecordingLLM  # noqa: E402
from src.anomaly_agent.real import run_real  # noqa: E402
from src.anomaly_agent.real.download_nab import download  # noqa: E402
from src.anomaly_agent.tools.metric_tools import MetricTools  # noqa: E402

SERIES, DETECTOR_COLOR, MUTED, CONF_LABEL = base.SERIES, base.DETECTOR_COLOR, base.MUTED, base.CONF_LABEL
LLM_CSV = ROOT / "data" / "clean" / "agent_eval_llm.csv"
REAL_DIR = ROOT / "data" / "real"
REAL_DB = REAL_DIR / "clean" / "nab.duckdb"
REAL_PROFILE = ROOT / "profiles" / "nab.yaml"
NO_KEY_HELP = ("To enable the LLM: copy `.env.example` to `.env` in the project folder, put your Groq key "
               "after `GROQ_API_KEY=`, save, and rerun this page. The key stays on your machine.")


def _pass_mark(value):
    return "✓ pass" if bool(value) else "✗ fail"


# ============================================================================== page: agent
def _run_agent(inc, hidden, decoy_name, max_steps, use_llm):
    """One investigation. hidden: (service, time) to delete from the log. decoy_name: an unrelated
    change to add just before the problem. Returns (report dict, evidence, recorded LLM calls)."""
    planner, judge, recorder = rule_planner, None, None
    if use_llm:
        recorder = RecordingLLM(GroqLLM())
        planner, judge = llm_parts(recorder, base.DB)
    with tempfile.TemporaryDirectory() as tmp:
        db = str(base.DB)
        if hidden:
            db = _without_change(db, hidden[0], hidden[1], tmp)
        if decoy_name:
            db = _with_decoy(db, inc["start"], tmp, next(d for d in DECOYS if d["service"] == decoy_name))
        report, evidence = investigate(inc, MetricTools(db), planner=planner, max_steps=max_steps, judge=judge)
    return report, evidence, (recorder.calls if recorder else [])


@st.cache_data(show_spinner="Asking Groq (planner and judge)...")
def _run_agent_llm(db_mtime, model, inc, hidden, decoy_name, max_steps):
    """Cached, so moving other widgets does not repeat the LLM calls (they take time and quota)."""
    return _run_agent(inc, hidden, decoy_name, max_steps, True)


def _show_report(report, evidence, heading):
    rep = Report(**report)
    st.subheader(heading)
    st.markdown(f"**Confidence:** {CONF_LABEL[rep.confidence]}")
    st.markdown(f"**What happened:** {rep.what_happened}")
    st.markdown(f"**Since when:** {rep.since_when}")
    st.markdown(f"**Who is affected:** {rep.who_affected}")
    st.markdown("**Top causes:**")
    for c in rep.top_causes:
        st.markdown(f"- {c.description} · final score **{c.score}** · evidence {', '.join(c.evidence_ids)}")
        if c.relatedness is not None:
            st.caption(f"timing score {c.timing_score} × LLM relatedness {c.relatedness} (kept at least 20% of "
                       f"the timing score). LLM reason: “{c.llm_reason}”")
        st.progress(min(max(c.score, 0.0), 1.0))
    st.markdown(f"**Suggested action:** {rep.suggested_action}")
    for n in rep.notes:
        st.warning(n)
    with st.expander("Evidence trail (every tool call)"):
        for e in evidence:
            st.markdown(f"**{e['id']} · {e['tool']}**")
            st.json({"arguments": e["args"], "result": e["result"]}, expanded=False)


def page_agent_llm(core, truth, profile):
    st.header("The investigating agent: rules or Groq LLM")
    anomalies = core["anomaly_view"]
    if anomalies.empty:
        st.warning("No alerts yet. Use 'Rebuild data' in the sidebar.")
        return
    incidents = group_alerts(anomalies)
    st.write("The agent looks at one incident with three read-only tools. With the LLM switched on, the LLM "
             "(1) chooses the next tool and (2) rates how plausibly each recent change could affect this metric. "
             "Every tool call and number still comes from the data, and every LLM answer is validated.")

    labels = {f"{i['incident_id']}: {i['metric']} / {i['entity']} from {i['start']:%b %d %H:%M}": i for i in incidents}
    inc = labels[st.selectbox("Incident", list(labels))]

    c1, c2, c3 = st.columns(3)
    max_steps = c1.slider("Step limit (tool calls)", 1, 8, 6)
    changes = core["changes"]
    options = ["(keep the full change log)"] + [
        f"{r.timestamp:%Y-%m-%d %H:%M} · {r.service} · {r.description}" for r in changes.itertuples()]
    hidden_choice = c2.selectbox("Hide one change from the agent", options)
    hidden = None
    if hidden_choice != options[0]:
        row = changes.iloc[options.index(hidden_choice) - 1]
        hidden = (row["service"], pd.Timestamp(row["timestamp"]))
    decoy_options = ["(none)"] + [f"{d['service']}: {d['description']}" for d in DECOYS]
    decoy_choice = c3.selectbox("Add an UNRELATED change just before the problem", decoy_options)
    decoy_name = None if decoy_choice == decoy_options[0] else DECOYS[decoy_options.index(decoy_choice) - 1]["service"]

    llm_ok = GroqLLM.available()
    use_llm = st.checkbox(f"Use the Groq LLM (planner + relatedness judge) · model: {get_model()}",
                          value=False, disabled=not llm_ok)
    if not llm_ok:
        st.info(NO_KEY_HELP)

    rule_report, rule_evidence, _ = _run_agent(inc, hidden, decoy_name, max_steps, False)
    if not use_llm:
        _show_report(rule_report, rule_evidence, f"Report for {inc['incident_id']} (rule-based agent)")
        st.caption("Try the 'unrelated change' box: the rule-based agent scores only by timing, so it trusts "
                   "a harmless deploy that happened just before. Then switch the LLM on.")
        return

    try:
        report, evidence, calls = _run_agent_llm(base.DB.stat().st_mtime, get_model(), inc, hidden,
                                                 decoy_name, max_steps)
    except Exception as e:  # key rejected, network down, ... never crash the page
        st.error(f"The LLM agent could not run: {type(e).__name__}: {str(e)[:200]}")
        _show_report(rule_report, rule_evidence, "Rule-based agent (fallback)")
        return

    left, right = st.columns(2)
    with left:
        _show_report(report, evidence, f"Report for {inc['incident_id']} (Groq LLM agent)")
    with right:
        _show_report(rule_report, rule_evidence, "Same incident, rule-based agent")

    top = lambda r: (f"{r['top_causes'][0]['service']} @ {r['top_causes'][0]['change_time']}"
                     if r["top_causes"][0]["service"] else "no change found")
    st.dataframe(pd.DataFrame([
        {"agent": "rule-based", "top cause": top(rule_report), "confidence": rule_report["confidence"],
         "score": rule_report["top_causes"][0]["score"]},
        {"agent": "Groq LLM", "top cause": top(report), "confidence": report["confidence"],
         "score": report["top_causes"][0]["score"]},
    ]), hide_index=True, width="stretch")

    with st.expander(f"What the LLM was asked and what it answered ({len(calls)} calls)"):
        if calls:
            st.markdown("**Instructions given to the planner / judge**")
            st.code("\n\n".join(sorted({c['system'] for c in calls})), language="text")
        for i, c in enumerate(calls, 1):
            st.markdown(f"**Call {i}**")
            st.code(c["user"], language="text")
            if "error" in c:
                st.error(c["error"])
            else:
                st.json(c["reply"])
    if st.checkbox("Reveal the answer key for this incident (the agent never sees it)", key="llm_reveal"):
        hit = truth[truth["series_set"].apply(lambda s: f"{inc['metric']}:{inc['entity']}" in s)
                    & (truth["start"] <= inc["end"]) & (truth["end"] >= inc["start"])]
        st.dataframe(hit[["type", "start", "end", "is_anomaly", "true_cause", "cause_type"]],
                     hide_index=True, width="stretch")


# ============================================================================== page: evaluation
@st.cache_data
def _eval_rules(db_mtime, truth_mtime):
    core = base.load_core(db_mtime)
    return evaluate(str(base.DB), core["anomalies"], load_truth(base.TRUTH), decoys=DECOYS)


def _eval_tiles(cases):
    s = summarize(cases)
    base.stat_row([("Top cause correct", s["top_cause_correct"]), ("Cause removed, handled", s["cause_removed_handled"]),
                   ("Unrelated decoy, handled", s["decoy_handled"]), ("Confident & wrong", s["overconfident_wrong_answers"]),
                   ("Cases passed", s["cases_passed"])])


def page_eval_llm(core, truth, profile):
    st.header("Agent evaluation: rules vs Groq LLM")
    st.write("Each real problem is investigated and graded against the answer key. Trick cases: **cause removed** "
             "(the true change is deleted) and **decoy added** (an unrelated change, like a search-ranking tweak or "
             "an email footer edit, lands just before an outside attack). A good agent must not confidently blame it.")
    if core["anomalies"].empty:
        st.warning("No alerts yet. Use 'Rebuild data' in the sidebar.")
        return

    rules = _eval_rules(base.DB.stat().st_mtime, base.TRUTH.stat().st_mtime)
    st.subheader("Rule-based agent (scores a change only by timing)")
    _eval_tiles(rules)

    st.subheader("Groq LLM agent")
    llm_ok = GroqLLM.available()
    if not llm_ok:
        st.info(NO_KEY_HELP)
    if st.button("Run the Groq evaluation (about 30 LLM calls)", disabled=not llm_ok):
        with st.spinner("Asking Groq..."):
            try:
                planner, judge = llm_parts(GroqLLM(), base.DB)
                cases = evaluate(str(base.DB), core["anomalies"], load_truth(base.TRUTH),
                                 planner=planner, judge=judge, decoys=DECOYS)
                cases.to_csv(LLM_CSV, index=False)
                st.session_state["llm_eval"] = cases
            except Exception as e:
                st.error(f"The Groq evaluation failed: {type(e).__name__}: {str(e)[:200]}")
    llm_cases = st.session_state.get("llm_eval")
    if llm_cases is None and LLM_CSV.exists():
        llm_cases = pd.read_csv(LLM_CSV)
        llm_cases["passed"] = llm_cases["passed"].astype(bool)
        st.caption(f"Showing the last saved run ({LLM_CSV.name}).")
    if llm_cases is None:
        st.caption("No Groq results yet. Press the button (it uses your Groq quota).")
        return
    _eval_tiles(llm_cases)

    both = rules[["case", "confidence", "passed"]].merge(
        llm_cases[["case", "confidence", "passed"]], on="case", how="outer", suffixes=(" · rules", " · Groq"))
    for col in ("passed · rules", "passed · Groq"):
        both[col] = both[col].map(lambda v: "-" if pd.isna(v) else _pass_mark(v))
    st.subheader("Side by side")
    st.dataframe(both, hide_index=True, width="stretch")
    st.caption("Small benchmark on one made-up world: a sanity check, not a guarantee. The judge's rubric was written "
               "after seeing the first decoy fail, so the other two decoys are the fairer test.")
    with st.expander("Full Groq results"):
        st.dataframe(llm_cases.drop(columns=[c for c in ("in_top3",) if c in llm_cases]), hide_index=True, width="stretch")


# ============================================================================== page: real data
def _real_ready():
    return REAL_DB.exists() and (REAL_DIR / "nab_raw" / "combined_windows.json").exists()


@st.cache_data(show_spinner="Running the detectors on real data...")
def _real_results(db_mtime):
    con = duckdb.connect(str(REAL_DB), read_only=True)
    metrics = con.execute("SELECT timestamp, entity, metric, value FROM metrics").df()
    info = con.execute("SELECT metric, type FROM metric_info").df()
    con.close()
    profile = load_profile(REAL_PROFILE)
    setups = run_real.run_setups(metrics, info, profile)
    truth = run_real.nab_truth(REAL_DIR / "nab_raw" / "combined_windows.json")
    return metrics, setups, truth, run_real.score_datasets(setups, truth)


def _real_chart(series, windows, alerts, metric):
    hourly = series.set_index("timestamp")["value"].resample("1h").mean().dropna().reset_index()
    layers = []
    if len(windows):
        bands = pd.DataFrame({"start": windows["start"], "end": windows["end"], "label": "NAB labelled window"})
        layers.append(alt.Chart(bands).mark_rect(opacity=0.18, color=MUTED).encode(x="start:T", x2="end:T"))
    line = alt.Chart(hourly).mark_line(strokeWidth=1.5, color=SERIES[0]).encode(
        x=alt.X("timestamp:T", title=None, axis=alt.Axis(grid=False)),
        y=alt.Y("value:Q", title=f"{metric} (hourly mean)", scale=alt.Scale(zero=False), axis=alt.Axis(gridOpacity=0.2)))
    hover = line.mark_point(size=40, opacity=0).encode(
        tooltip=[alt.Tooltip("timestamp:T", format="%b %d %H:%M"), alt.Tooltip("value:Q", format=".4g")])
    charts = [alt.layer(*layers, line, hover).properties(height=320)]
    if len(alerts):
        strip = alt.Chart(alerts).mark_bar(height=10, cornerRadius=2).encode(
            x=alt.X("start:T", title=None), x2="end:T", y=alt.Y("detector:N", title=None),
            color=alt.Color("detector:N", scale=alt.Scale(domain=list(DETECTOR_COLOR), range=list(DETECTOR_COLOR.values())),
                            legend=alt.Legend(title="alerts by", orient="top")),
            tooltip=["detector:N", alt.Tooltip("start:T", format="%b %d %H:%M"), alt.Tooltip("end:T", format="%b %d %H:%M"),
                     alt.Tooltip("peak_value:Q", format=".4g")]).properties(height=70)
        charts.append(strip)
    return alt.vconcat(*charts).resolve_scale(x="shared", color="independent")


def page_real(core, truth, profile):
    st.header("Real data: the Numenta Anomaly Benchmark (NAB)")
    st.write("Four real series (NYC taxi demand, an AWS server's CPU, EC2 request latency, an industrial machine's "
             "temperature) with anomaly windows labelled by people. Same adapter, detectors and scoreboard as the "
             "made-up shop; only the profile (`profiles/nab.yaml`) is new.")
    st.warning("Read with care: NAB windows are days wide and the labels are incomplete. So **'found' is lenient** and "
               "**'alerts outside windows' over-counts** real-but-unlabelled problems. Compare the setups with each "
               "other; do not read these as accuracy scores.")
    if not _real_ready():
        st.info("The real data is not on this machine yet (it is downloaded from GitHub and git-ignored).")
        if st.button("Download NAB data and build the real-data tables"):
            with st.spinner("Downloading and building..."):
                download(REAL_DIR / "nab_raw")
                build(str(REAL_PROFILE), str(REAL_DIR), str(REAL_DB))
            st.cache_data.clear()
            st.rerun()
        return

    metrics, setups, nab_truth, table = _real_results(REAL_DB.stat().st_mtime)
    st.subheader("Scoreboard: three setups on each real series")
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Static 3-sigma = a fixed limit of mean ± 3 standard deviations from the series' clean start. "
               "'first hit' is measured from the START of a days-wide window, so it is not a detection delay; a "
               "negative value means an alert began before the window.")

    names = {metric: metric for metric, _ in run_real.DATASETS.values()}
    c1, c2 = st.columns(2)
    metric = c1.selectbox("Series", list(names))
    setup = c2.selectbox("Alerts from", list(setups), index=2)
    series = metrics[metrics["metric"] == metric]
    windows = nab_truth[nab_truth["series"].str.startswith(f"{metric}:")]
    alerts = setups[setup][setups[setup]["metric"] == metric]
    st.altair_chart(_real_chart(series, windows, alerts, metric), width="stretch")
    st.caption(f"{len(alerts)} alerts from '{setup}'. Grey bands are NAB's labelled windows (an answer-key overlay "
               "for you; detectors never see them).")

    st.subheader("What this teaches")
    st.markdown(
        "- **The made-up world was kinder than real life.** The seasonal setup that raised 0 false alarms there "
        "raises hundreds of short alerts on the real CPU and machine-temperature series.\n"
        "- **Two real causes:** short noisy blips on 5-minute data (a longer `min_points` helps), and a level that "
        "*shifts over time*, so 'normal' learned once from the first week goes stale. Re-learning periodically is "
        "the natural next step.\n"
        "- **Seasonal is much better on the taxi series** (all 5 holiday/event windows found), where weekly patterns "
        "are strong, and weak where there is no real seasonality.\n"
        "- **Real exports are messy:** duplicate timestamps were found and averaged by the adapter.")

    st.subheader("Try the investigating agent on a real incident")
    seasonal = setups["3 seasonal + drift"]
    pool = group_alerts(seasonal[seasonal["metric"] == metric]) if (seasonal["metric"] == metric).any() else []
    if not pool:
        st.info("This setup raised no alerts for the chosen series.")
        return
    labels = {f"{i['incident_id']}: from {i['start']:%b %d %H:%M} for {(i['end'] - i['start']).total_seconds() / 3600:.1f} h": i
              for i in pool[:80]}
    inc = labels[st.selectbox("Incident", list(labels))]
    report, evidence = investigate(inc, MetricTools(str(REAL_DB)))
    st.code(Report(**report).to_text(), language="text")
    st.caption("Only the taxi series has a change log (a hand-written calendar of NYC events). The agent scores a "
               "change by how soon it was *before* the incident, but a holiday lasts all day, so it often fails to "
               "link a real holiday to its own alert. An event needs a duration, not just a start time.")


# ============================================================================== main
PAGES = {
    **{name: page for name, page in base.PAGES.items() if name not in ("Agent", "Evaluation")},
    "Agent (rules or Groq LLM)": page_agent_llm,
    "Evaluation (rules vs Groq)": page_eval_llm,
    "Real data (NAB)": page_real,
}


def main():
    st.set_page_config(page_title="Anomaly Agent Explorer (LLM + real data)", layout="wide")
    st.sidebar.title("Anomaly Agent")
    st.sidebar.caption("Version 2: Groq LLM agent + real data. The original app is separate: "
                       "`streamlit run src/anomaly_agent/app/app.py`")
    if st.sidebar.button("Rebuild data (raw -> clean -> alerts)"):
        with st.spinner("Rebuilding..."):
            base.rebuild_everything()
        st.sidebar.success("Rebuilt.")

    page = st.sidebar.radio("Page", list(PAGES))
    if page != "Real data (NAB)" and not base.have_data():
        st.title("Anomaly Agent Explorer")
        st.warning("No data yet. Click **Rebuild data** in the sidebar (takes about 20 seconds).")
        return
    if page == "Real data (NAB)":
        PAGES[page](None, None, None)
        return
    core = base.load_core(base.DB.stat().st_mtime)
    core = {**core, "anomaly_view": core["anomalies"]}
    PAGES[page](core, base.load_truth_cached(base.TRUTH.stat().st_mtime), load_profile(base.PROFILE))


main()
