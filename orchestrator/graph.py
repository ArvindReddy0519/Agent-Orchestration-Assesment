from langgraph.graph import END, START, StateGraph

from orchestrator.nodes import coder, planner, reviewer
from orchestrator.state import OrchestratorState


def build_graph():
    """Wire planner → coder → reviewer and return a compiled graph."""
    builder = StateGraph(OrchestratorState)

    builder.add_node("planner", planner)
    builder.add_node("coder", coder)
    builder.add_node("reviewer", reviewer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "coder")
    builder.add_edge("coder", "reviewer")
    builder.add_edge("reviewer", END)

    return builder.compile()
