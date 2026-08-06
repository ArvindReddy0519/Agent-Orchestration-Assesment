from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from orchestrator.nodes import (
    approval_node,
    coder,
    planner,
    reviewer,
    route_after_approval,
)
from orchestrator.state import OrchestratorState


def build_graph(*, checkpointer: MemorySaver | None = None):
    """Build planner → approval → coder → reviewer with human-in-the-loop checkpoint."""
    builder = StateGraph(OrchestratorState)

    builder.add_node("planner", planner)
    builder.add_node("approval", approval_node)
    builder.add_node("coder", coder)
    builder.add_node("reviewer", reviewer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "approval")
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"planner": "planner", "coder": "coder"},
    )
    builder.add_edge("coder", "reviewer")
    builder.add_edge("reviewer", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=saver)
