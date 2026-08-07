"""Run URL shortener orchestration scenarios with interactive human-in-the-loop approval.

Requires a Groq API key for LLM nodes (planner, coder, reviewer):

    export GROQ_API_KEY='your-key-here'

Or place GROQ_API_KEY in a `.env` file in the project root (loaded automatically).

Examples:

    python scripts/run_scenarios.py --scenario greenfield
    python scripts/run_scenarios.py --scenario brownfield
    python scripts/run_scenarios.py --scenario ambiguous
"""

from __future__ import annotations

import argparse
import sys
import textwrap
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

SCENARIO_REQUIREMENTS: dict[str, str] = {
    "greenfield": textwrap.dedent(
        """
        Build a new URL shortener REST API from scratch using Java 17, Spring Boot 3,
        and an in-memory or H2 JPA store for the MVP.

        Functional requirements:
        - POST /api/shorten accepts {"url": "<long url>", "alias": "<optional custom slug>"}
          and returns {"shortUrl": "...", "targetUrl": "..."}.
        - GET /{slug} redirects (HTTP 307) to the original URL.
        - Reject invalid URLs, duplicate aliases, and reserved slugs (e.g. "health", "actuator").
        - Include GET /actuator/health or equivalent health endpoint.

        Non-functional requirements:
        - Use DTOs with Jakarta Bean Validation (@Valid, @NotBlank, @URL).
        - Layered architecture: controller, service, repository, entity.
        - Include @ControllerAdvice for consistent error responses.
        - Provide JUnit 5 tests with MockMvc.
        - Structure persistence so Redis or PostgreSQL can replace H2 later.
        """
    ).strip(),
    "brownfield": textwrap.dedent(
        """
        Extend the existing Spring Boot URL shortener service below. Do not rewrite from
        scratch; preserve current endpoints and behavior, then add the requested features.

        Existing service (Spring Boot 3 + in-memory JPA):

        ```java
        // UrlController.java — simplified legacy MVP
        @RestController
        public class UrlController {
            private final UrlService urlService;

            public UrlController(UrlService urlService) {
                this.urlService = urlService;
            }

            @PostMapping("/shorten")
            public ShortenResponse shorten(@Valid @RequestBody ShortenRequest request) {
                return urlService.shorten(request.getUrl());
            }

            @GetMapping("/{slug}")
            public ResponseEntity<Void> redirect(@PathVariable String slug) {
                return ResponseEntity.status(HttpStatus.TEMPORARY_REDIRECT)
                    .location(URI.create(urlService.resolve(slug)))
                    .build();
            }
        }
        ```

        Required enhancements:
        - Optional custom alias on POST /shorten; return 409 when alias already exists.
        - Track click counts per slug; expose GET /stats/{slug}.
        - Add TTL support via optional expiresInSeconds on shorten; expired links return 410.
        - Keep backward compatibility for clients that only send {"url": "..."}.
        - Add/update JUnit 5 tests for new behavior.
        """
    ).strip(),
    "ambiguous": textwrap.dedent(
        """
        We need a URL shortener for our startup. Users should be able to make links
        shorter and share them. It should be fast, secure, and "enterprise ready."
        Maybe add analytics if that's easy. Use whatever stack you think is best.
        We might need auth later but not sure yet. Ship something we can demo next week.
        """
    ).strip(),
}


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


def _summarize_update(node: str, update: dict[str, Any]) -> str:
    if node == "planner" and "plan" in update:
        plan = update["plan"]
        preview = plan[:120].replace("\n", " ")
        suffix = "..." if len(plan) > 120 else ""
        return f"plan generated ({len(plan)} chars): {preview}{suffix}"
    if node == "coder" and "code" in update:
        code = update["code"]
        lines = code.count("\n") + 1 if code else 0
        return f"code generated ({lines} lines, {len(code)} chars)"
    if node == "reviewer" and "errors" in update:
        errors = update["errors"].strip()
        if errors:
            finding_count = errors.count("\n") + 1
            return f"review failed ({finding_count} finding(s))"
        return "review passed (APPROVED)"
    if node == "increment_retry" and "retry_count" in update:
        return f"retry_count -> {update['retry_count']}"
    if node == "approval" and "human_feedback" in update:
        return f"human_feedback recorded: {update['human_feedback']!r}"
    keys = ", ".join(update.keys()) if update else "(no state change)"
    return f"updated: {keys}"


def run_scenario(scenario: str) -> None:
    requirement = SCENARIO_REQUIREMENTS[scenario]
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    inputs: dict[str, Any] | Command = {
        "requirement": requirement,
        "retry_count": 0,
        "errors": "",
    }

    logs: list[str] = [
        f"scenario={scenario}",
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
    for key in ("retry_count", "errors", "human_feedback"):
        if key in final_state:
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
            "Run URL shortener orchestration scenarios with interactive plan approval. "
            "Set GROQ_API_KEY before running."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_REQUIREMENTS),
        required=True,
        help="Scenario requirement to load: greenfield, brownfield, or ambiguous",
    )
    args = parser.parse_args()

    ensure_groq_api_key()
    run_scenario(args.scenario)


if __name__ == "__main__":
    main()
