from typing import Annotated, TypedDict

from orchestrator.tasks import merge_dicts


class OrchestratorState(TypedDict, total=False):
    """Shared state for the SDLC orchestration graph."""

    requirement: str
    normalized_requirement: str
    plan: str
    architecture_decisions: list[str]
    tasks: list[dict]
    completed_tasks: Annotated[dict[str, str], merge_dicts]
    artifacts: Annotated[dict[str, str], merge_dicts]
    artifact_reviews: Annotated[dict[str, str], merge_dicts]
    current_task_id: str
    code: str
    review_summary: str
    errors: str
    retry_count: int
    human_feedback: str
