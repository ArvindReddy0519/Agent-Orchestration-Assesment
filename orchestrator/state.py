from typing import TypedDict


class OrchestratorState(TypedDict, total=False):
    """Shared state for the SDLC orchestration graph."""

    requirement: str
    plan: str
    code: str
    errors: str
    retry_count: int
    human_feedback: str
