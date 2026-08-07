"""Run the orchestration workflow with a dynamic requirement and interactive HITL approval.

Requires a Groq API key for LLM nodes (planner, coder, reviewer):

    export GROQ_API_KEY='your-key-here'

Or place GROQ_API_KEY in a `.env` file in the project root (loaded automatically).

Examples:

    python scripts/run_scenarios.py --requirement "Build a URL shortener REST API"
    python scripts/run_scenarios.py
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from orchestrator.graph import build_graph
from orchestrator.file_writer import save_generated_java_code
from orchestrator.llm import ensure_groq_api_key


def _print_interrupt(payload: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("PLAN APPROVAL REQUIRED — graph paused")
    print("=" * 60)
    print(payload.get("message", ""))
    print("\n--- Generated plan ---\n")
    print(payload.get("plan", ""))
    print("\n" + "-" * 60)


def _prompt_approval() -> str:
    print(
        "\nEnter your decision:\n"
        "  Approved\n"
        "  Rejected: <your feedback for the planner>\n"
    )
    while True:
        try:
            decision = input("Decision> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            raise SystemExit(130) from None

        if not decision:
            print("Please enter 'Approved' or 'Rejected: <feedback>'.")
            continue
        lowered = decision.lower()
        if lowered == "approved" or lowered.startswith("rejected"):
            return decision
        print("Invalid input. Use 'Approved' or 'Rejected: <feedback>'.")


def _prompt_requirement() -> str:
    """Read a multi-line requirement from the terminal."""
    print(
        "Enter your requirement below. Press Enter on an empty line when finished:\n"
    )
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines:
            break
        lines.append(line)

    requirement = "\n".join(lines).strip()
    if not requirement:
        print("No requirement provided.", file=sys.stderr)
        raise SystemExit(1)
    return requirement


def _resolve_requirement(cli_requirement: str | None) -> str:
    if cli_requirement and cli_requirement.strip():
        return cli_requirement.strip()
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    return _prompt_requirement()


def _summarize_update(node: str, update: dict[str, Any]) -> str:
    if node == "planner" and "tasks" in update:
        count = len(update.get("tasks", []))
        return f"execution plan generated ({count} tasks)"
    if node == "coder" and "completed_tasks" in update:
        task_ids = ", ".join(update["completed_tasks"].keys())
        return f"task completed: {task_ids}"
    if node == "reviewer" and "artifact_reviews" in update:
        paths = ", ".join(update["artifact_reviews"].keys())
        return f"artifact(s) reviewed: {paths}"
    if node == "assemble_code" and "code" in update:
        code = update["code"]
        return f"assembled codebase ({len(code)} chars)"
    if node == "summarize_reviews":
        if "review_summary" in update:
            summary = update["review_summary"].strip()
            if update.get("errors", "").strip():
                return f"review summary: findings recorded ({len(summary)} chars)"
            return "review summary: all artifacts APPROVED"
        if "errors" in update and update["errors"].strip():
            return "review summary: findings recorded"
    if node == "increment_retry" and "retry_count" in update:
        return f"retry_count -> {update['retry_count']}"
    if node == "approval" and "human_feedback" in update:
        return f"human_feedback recorded: {update['human_feedback']!r}"
    keys = ", ".join(update.keys()) if update else "(no state change)"
    return f"updated: {keys}"


def run_scenario(requirement: str) -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    inputs: dict[str, Any] | Command = {
        "requirement": requirement,
        "retry_count": 0,
        "errors": "",
        "completed_tasks": {},
        "artifacts": {},
        "artifact_reviews": {},
        "tasks": [],
        "architecture_decisions": [],
    }

    preview = requirement[:80].replace("\n", " ")
    if len(requirement) > 80:
        preview += "..."
    logs: list[str] = [
        f"requirement={preview!r}",
        f"thread_id={config['configurable']['thread_id']}",
        "workflow started",
    ]

    while True:
        interrupted = False
        for chunk in graph.stream(inputs, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
                _print_interrupt(payload)
                decision = _prompt_approval()
                logs.append(f"[approval] human decision: {decision!r}")
                inputs = Command(resume=decision)
                interrupted = True
                break

            for node, update in chunk.items():
                summary = _summarize_update(node, update)
                logs.append(f"[{node}] {summary}")

        if not interrupted:
            break

    snapshot = graph.get_state(config)
    final_state = snapshot.values if snapshot else {}

    print("\n" + "=" * 60)
    print("EXECUTION LOG")
    print("=" * 60)
    for entry in logs:
        print(f"  {entry}")
    if snapshot and snapshot.next:
        print(f"  next node(s): {snapshot.next}")

    print("\n" + "=" * 60)
    print("FINAL STATE SUMMARY")
    print("=" * 60)
    for key in ("retry_count", "errors", "human_feedback", "tasks", "review_summary"):
        if key in final_state:
            if key == "tasks":
                print(f"  tasks: {len(final_state[key])} task(s)")
            else:
                print(f"  {key}: {final_state[key]!r}")

    print("\n" + "=" * 60)
    print("GENERATED CODE")
    print("=" * 60)
    generated_code = final_state.get("code", "// (no code produced)")
    print(generated_code)

    written_files = save_generated_java_code(generated_code)
    if written_files:
        print("\n" + "=" * 60)
        print("WRITTEN FILES")
        print("=" * 60)
        for path in written_files:
            print(f"  {path}")
    else:
        print("\n(no ### File: blocks found — project not written to disk)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the multi-agent orchestration workflow with a dynamic requirement "
            "and interactive plan approval. Set GROQ_API_KEY before running."
        ),
    )
    parser.add_argument(
        "--requirement",
        help="Natural-language product requirement passed to the planner",
    )
    args = parser.parse_args()

    ensure_groq_api_key()
    requirement = _resolve_requirement(args.requirement)
    run_scenario(requirement)


if __name__ == "__main__":
    main()
