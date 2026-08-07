from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from orchestrator.nodes import (
    approval_node,
    assemble_code,
    coder,
    fan_out_tasks,
    increment_retry,
    invalidate_tasks,
    planner,
    reviewer,
    route_after_approval,
    route_after_reviewer,
    schedule_tasks,
    summarize_reviews,
)
from orchestrator.state import OrchestratorState


def build_graph(*, checkpointer: MemorySaver | None = None):
    """Build task-graph orchestration with HITL, per-artifact review, and bounded retries."""
    builder = StateGraph(OrchestratorState)

    builder.add_node("planner", planner)
    builder.add_node("approval", approval_node)
    builder.add_node("schedule_tasks", schedule_tasks)
    builder.add_node("coder", coder)
    builder.add_node("reviewer", reviewer)
    builder.add_node("assemble_code", assemble_code)
    builder.add_node("summarize_reviews", summarize_reviews)
    builder.add_node("invalidate_tasks", invalidate_tasks)
    builder.add_node("increment_retry", increment_retry)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "approval")
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"planner": "planner", "schedule_tasks": "schedule_tasks"},
    )
    builder.add_conditional_edges(
        "schedule_tasks",
        fan_out_tasks,
        ["coder", "assemble_code"],
    )
    builder.add_edge("coder", "reviewer")
    builder.add_edge("reviewer", "schedule_tasks")
    builder.add_edge("assemble_code", "summarize_reviews")
    builder.add_conditional_edges(
        "summarize_reviews",
        route_after_reviewer,
        {"invalidate_tasks": "invalidate_tasks", "__end__": END},
    )
    builder.add_edge("invalidate_tasks", "increment_retry")
    builder.add_edge("increment_retry", "schedule_tasks")

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=saver)
