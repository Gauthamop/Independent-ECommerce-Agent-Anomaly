"""How steady is the LLM judge? Ask the SAME question several times and look at the spread.

A confidence level flips when a score crosses a boundary (0.35 or 0.7), so a judge whose score wobbles
between runs makes the confidence wobble too. Run this before trusting any single LLM result.

    python -m src.anomaly_agent.evaluation.judge_stability [repeats] [--services]

It pauses between calls and stops at the first rate-limit error (free tiers have a daily token limit;
a reasoning model spends many tokens per call, so keep the repeat count small).
"""
import sys
import tempfile
import time
from statistics import mean

import duckdb

from src.anomaly_agent.agent.graph import investigate
from src.anomaly_agent.agent.incidents import group_alerts
from src.anomaly_agent.agent.llm_agent import load_metric_descriptions, load_service_descriptions, make_llm_judge
from src.anomaly_agent.evaluation.agent_eval import DECOYS, _with_decoy
from src.anomaly_agent.llm.groq_llm import GroqLLM, get_model
from src.anomaly_agent.llm.recording import RecordingLLM
from src.anomaly_agent.tools.metric_tools import MetricTools


def _evidence_state(inc, db):
    """Collect the evidence the judge would see, using the rule-based agent (no LLM calls)."""
    _, evidence = investigate(inc, MetricTools(db))
    return {"incident": inc, "evidence": evidence, "steps": len(evidence), "max_steps": 6}


def main(db_path="data/clean/anomaly.duckdb", repeats=5, use_services=False, pause=3.0):
    con = duckdb.connect(db_path, read_only=True)
    alerts = con.execute("SELECT * FROM anomalies").df()
    con.close()
    incidents = {(i["metric"], i["entity"]): i for i in group_alerts(alerts)}
    fraud = incidents[("decline_rate", "bank_c/south")]
    llm = RecordingLLM(GroqLLM())
    judge = make_llm_judge(llm, load_metric_descriptions(db_path),
                           load_service_descriptions(db_path) if use_services else None)

    with tempfile.TemporaryDirectory() as tmp:
        cases = {  # name -> (state, the service whose score we read)
            "TRUE bad deploy": (_evidence_state(incidents[("decline_rate", "bank_a/north")], db_path), "payment-gateway"),
            "TRUE memory leak": (_evidence_state(incidents[("memory_pct", "server_2")], db_path), "checkout-service"),
            "DECOY search-service": (_evidence_state(fraud, _with_decoy(db_path, fraud["start"], tmp, DECOYS[0])),
                                     "search-service"),
        }
        print(f"model {get_model()}, {repeats} repeats each, service catalog {'ON' if use_services else 'off'}")
        for name, (state, service) in cases.items():
            scores = []
            for _ in range(repeats):
                judgement, _notes = judge(state)
                got = [v["relatedness"] for k, v in judgement.items() if k.startswith(service + "|")]
                if not got:
                    error = llm.calls[-1].get("error", "")
                    print(f"  {name}: no answer ({error[:120]})")
                    if "RateLimit" in error or "429" in error:
                        print("  Rate limit reached: stop here and try again after the limit resets.")
                        return
                    continue
                scores.append(got[0])
                time.sleep(pause)
            if scores:
                print(f"  {name:22} {scores}  min {min(scores)} max {max(scores)} mean {mean(scores):.2f}")


if __name__ == "__main__":
    numbers = [a for a in sys.argv[1:] if a.isdigit()]
    main(repeats=int(numbers[0]) if numbers else 5, use_services="--services" in sys.argv)
