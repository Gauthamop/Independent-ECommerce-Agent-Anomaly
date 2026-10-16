"""The LangGraph agent.

    START -> plan -> act -> plan -> ... -> report -> END

plan   : the planner picks the next tool call, or None when the evidence is enough
         (it is also forced to stop at max_steps)
act    : run that tool (read-only, only from the allowed list) and store the result as evidence
report : write the report from the evidence only
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.anomaly_agent.agent.planner import rule_planner
from src.anomaly_agent.agent.report import build_report
from src.anomaly_agent.tools.metric_tools import ALLOWED_TOOLS


class AgentState(TypedDict, total=False):
    incident: dict
    evidence: list
    steps: int
    max_steps: int
    action: dict
    report: dict


def build_graph(tools, planner=rule_planner):
    def plan(state):
        if state["steps"] >= state["max_steps"]:
            return {"action": None}  # step limit: stop and report what we have
        return {"action": planner(state)}

    def act(state):
        action = state["action"]
        if action["tool"] not in ALLOWED_TOOLS:
            raise ValueError(f"tool not allowed: {action['tool']}")
        result = getattr(tools, action["tool"])(**action["args"])
        evidence = state["evidence"] + [{
            "id": f"E{len(state['evidence']) + 1}",
            "tool": action["tool"], "args": action["args"], "result": result,
        }]
        return {"evidence": evidence, "steps": state["steps"] + 1}

    def report(state):
        return {"report": build_report(state).model_dump()}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("act", act)
    graph.add_node("report", report)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", lambda s: "act" if s.get("action") else "report",
                                {"act": "act", "report": "report"})
    graph.add_edge("act", "plan")
    graph.add_edge("report", END)
    return graph.compile()


def investigate(incident, tools, planner=rule_planner, max_steps=6):
    """Run the agent on one incident. Returns (Report-as-dict, evidence list)."""
    app = build_graph(tools, planner)
    final = app.invoke({"incident": incident, "evidence": [], "steps": 0, "max_steps": max_steps})
    return final["report"], final["evidence"]
