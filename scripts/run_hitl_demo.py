"""Demonstrate human-in-the-loop plan approval via LangGraph interrupt/resume.

Requires a Groq API key (free tier) for LLM nodes (planner, coder, reviewer):

    export GROQ_API_KEY='your-key-here'

Optional model override (default: llama-3.3-70b-versatile):

    export GROQ_MODEL='llama-3.3-70b-versatile'

Examples:

    python scripts/run_hitl_demo.py --scenario approved
    python scripts/run_hitl_demo.py --scenario rejected
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.graph import build_graph
from orchestrator.llm import ensure_groq_api_key


def _print_interrupt(payload: dict) -> None:
    print("\n--- Plan approval required (graph paused) ---")
    print(payload.get("message", ""))
    print("\nGenerated plan:")
    print(payload.get("plan", ""))
    print("---\n")


def run_until_interrupt(requirement: str) -> tuple[object, dict]:
    """Compile with MemorySaver, run until approval_node calls interrupt()."""
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke(
        {"requirement": requirement, "retry_count": 0, "errors": ""},
        config,
    )

    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if interrupts:
        _print_interrupt(interrupts[0].value)

    snapshot = graph.get_state(config)
    print(f"Next node(s) waiting: {snapshot.next}")

    return graph, config


def resume(graph: object, config: dict, decision: str) -> dict:
    """Resume after interrupt with Command(resume=...)."""
    return graph.invoke(Command(resume=decision), config)


def demo_approved(requirement: str) -> None:
    graph, config = run_until_interrupt(requirement)

    print("Resuming with Command(resume='Approved')")
    final = resume(graph, config, "Approved")
    _print_state(final)


def demo_rejected_then_approved(requirement: str) -> None:
    graph, config = run_until_interrupt(requirement)

    print("Resuming with Command(resume='Rejected: Please change X')")
    paused_again = resume(graph, config, "Rejected: Please change X")
    if isinstance(paused_again, dict) and paused_again.get("__interrupt__"):
        _print_interrupt(paused_again["__interrupt__"][0].value)

    print("Resuming with Command(resume='Approved')")
    final = resume(graph, config, "Approved")
    _print_state(final)


def _print_state(state: dict) -> None:
    print("\nFinal state:")
    for key in ("requirement", "plan", "human_feedback", "code", "errors", "retry_count"):
        if key in state:
            print(f"  {key}: {state[key]!r}")


def main() -> None:
    ensure_groq_api_key()

    parser = argparse.ArgumentParser(
        description=(
            "Human-in-the-loop orchestrator demo (Groq LLM). "
            "Set GROQ_API_KEY before running."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=("approved", "rejected"),
        default="approved",
        help="approved: single resume; rejected: reject then approve on replan",
    )
    parser.add_argument(
        "--requirement",
        default="Build a REST API for todos",
        help="Initial requirement passed to the planner",
    )
    args = parser.parse_args()

    if args.scenario == "approved":
        demo_approved(args.requirement)
    else:
        demo_rejected_then_approved(args.requirement)


if __name__ == "__main__":
    main()
