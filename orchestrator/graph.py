from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from orchestrator.nodes import (
    approval_node,
    coder,
    increment_retry,
    planner,
    reviewer,
    route_after_approval,
    route_after_reviewer,
)
from orchestrator.state import OrchestratorState


def build_graph(*, checkpointer: MemorySaver | None = None):
    """Build planner → approval → coder ↔ reviewer (bounded retries) with HITL."""
    builder = StateGraph(OrchestratorState)

    builder.add_node("planner", planner)
    builder.add_node("approval", approval_node)
    builder.add_node("coder", coder)
    builder.add_node("reviewer", reviewer)
    builder.add_node("increment_retry", increment_retry)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "approval")
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"planner": "planner", "coder": "coder"},
    )
    builder.add_edge("coder", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {"increment_retry": "increment_retry", "__end__": END},
    )
    builder.add_edge("increment_retry", "coder")

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=saver)
