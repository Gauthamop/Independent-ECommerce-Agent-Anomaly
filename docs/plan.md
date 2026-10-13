# Project: anomaly-agent

E-commerce anomaly detection and root-cause agent. 2-week learning project,
built by me (basic-intermediate ML, LangChain/LangGraph). Synthetic data.

## How to help me (IMPORTANT)
- I am learning. Do NOT write or edit project files unless I explicitly ask.
- Explain concepts, give hints, review code I wrote, explain errors, quiz me.
- When I'm stuck: hint first, then review my attempt. No full solutions.
- Simple English. Ask me to predict results before I run things.
- Boilerplate with no learning value (Streamlit layout, plot styling) is OK to write.

## The problem
Static threshold alerts cause: (1) false alarms on seasonal spikes,
(2) late detection of fraud and slow degradation, (3) no root-cause context
(TTR 18+ hours). Goal: fewer false alarms, earlier detection, automatic root cause.

## Architecture / data flow
Messy raw files -> adapter (driven by profile YAML) -> DuckDB canonical tables
-> detectors -> anomalies table -> agent tools -> LangGraph agent -> report/dashboard

Canonical tables:
- metrics: timestamp, entity, metric, value, tags
- events: individual records (e.g. transactions)
- changes: deploys/config changes (the agent's "what changed?" source)

## Synthetic world (30 days, 5-minute data)
Metrics: orders/min, payment decline rate, latency, server memory.
Tags: bank, region. Daily cycle, weekend dip, one sale-day spike (NOT an anomaly).
Injected problems (logged in data/ground_truth):
1. Bad deploy -> decline rate jumps for ONE bank (true cause is in deploy_log)
2. Slow memory leak
3. Short fraud-like burst
Raw files are deliberately messy: odd column names, wide format, local timezone.

## Build order
1. Threshold detector + scoreboard (true detections, false alarms, misses,
   minutes-to-detect). Every later detector must beat this.
2. Hand-made seasonal baseline (per hour-of-week median/spread) + sale-day flag
3. Drift detector (rolling slope) + detector registry chosen by metric type
   (rate/count/gauge) from the profile
4. Agent tools (plain Python over DuckDB, with pytest): query_metric,
   compare_segments, get_recent_changes
5. LangGraph agent: receive anomaly -> plan -> call tools -> enough evidence?
   -> write report. Max step limit.
   Report: what happened, since when, who is affected, top 3 causes,
   evidence, confidence, suggested action.
6. Evaluate agent on every injected problem + one "no real cause" case
7. Small fraud model (Kaggle credit card data, time-based split,
   logistic regression vs XGBoost, PR-AUC)
8. Minimal Streamlit page, then README

## Rules
- Folders: data/raw, data/clean, data/ground_truth, src/anomaly_agent/
  {generator,adapters,detectors,evaluation,tools,agent,fraud,app}, profiles/, tests/
- Detectors must NEVER read ground truth.
- No e-commerce words (bank, decline_rate) hard-coded in detectors/, tools/, agent/.
  They live only in profiles/ecommerce.yaml and the adapter.
- Always split time data by time, never randomly.
- Agent tools are read-only. Every agent claim must cite a tool result;
  otherwise say "not enough evidence".
- Use parameterized SQL. Tools return small, readable results.
- Stack: Python, pandas, DuckDB, pydantic, pytest, LangGraph, Streamlit.

## If I fall behind, cut in this order
feedback button -> Streamlit (use notebook) -> fraud model -> drift polish.
NEVER cut: synthetic data + ground truth, adapter, scoreboard, agent evaluation.

## Daily habit
Predict -> build by hand -> break it -> write 3-5 lines in LEARNING_LOG.md.

## Current status
Day 0 done (repo, env, structure). Next: Day 1, synthetic data generator.