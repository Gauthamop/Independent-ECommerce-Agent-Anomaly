# Project: anomaly-agent

E-commerce anomaly detection and root-cause agent. 2-week learning project.

## How I want you to help (IMPORTANT)
- I am learning. Do NOT write or edit project files unless I explicitly ask.
- Explain concepts, give hints, review code I wrote, explain errors, quiz me.
- When I'm stuck, give a hint first, not the full solution.
- Keep explanations in simple English.

## What the system does
1. Watch: learn "normal" (including sale-day spikes) and flag real problems.
2. Investigate: a LangGraph agent uses tools to find the likely cause.
3. Report: short summary for the analyst.

## Data flow
Messy raw files -> adapter -> DuckDB -> detectors -> agent tools -> report

## Folder layout
- data/raw: messy generated input. data/clean: DuckDB after adapter.
- data/ground_truth: answer key. Detectors must NEVER read it.
- src/anomaly_agent/: generator, adapters, detectors, evaluation, tools, agent, fraud, app
- profiles/ecommerce.yaml: all e-commerce-specific names live here.

## Rules
- No e-commerce words (bank, decline_rate) hard-coded in detectors/, tools/, agent/.
- Split time-series and fraud data by time, never randomly.
- Agent tools are read-only. Every agent claim must cite a tool result.
- Tools: Python, pandas, DuckDB, pydantic, pytest, LangGraph.

## Current status
Day 1 done (generator, adapter, threshold detector, scoreboard). Next: seasonal baseline detector.