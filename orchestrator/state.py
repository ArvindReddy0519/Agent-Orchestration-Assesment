from typing import TypedDict


class OrchestratorState(TypedDict, total=False):
    """Shared state for the SDLC orchestration graph."""

    requirement: str
    plan: str
    code: str
    errors: list[str]
    human_feedback: str
