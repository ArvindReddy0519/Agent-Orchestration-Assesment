"""Structured execution plan models and task-graph helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

class Task(BaseModel):
    id: str
    description: str
    dependencies: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    architecture_decisions: list[str] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)


def merge_dicts(left: dict[str, str] | None, right: dict[str, str] | None) -> dict[str, str]:
    """Merge dictionaries for parallel task completion updates."""
    if not right:
        return dict(left or {})
    if "__replace__" in right:
        return {key: value for key, value in right.items() if key != "__replace__"}
    merged = dict(left or {})
    merged.update(right)
    return merged


def tasks_from_dicts(raw: list[dict[str, Any]] | None) -> list[Task]:
    if not raw:
        return []
    return [Task.model_validate(item) for item in raw]


def tasks_to_dicts(tasks: list[Task]) -> list[dict[str, Any]]:
    return [task.model_dump() for task in tasks]


def parse_execution_plan(text: str) -> ExecutionPlan:
    """Parse planner LLM output into a validated execution plan."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Planner response did not contain JSON.")
    payload = json.loads(stripped[start : end + 1])
    return ExecutionPlan.model_validate(payload)


def format_plan_summary(plan: ExecutionPlan) -> str:
    """Compact human-readable summary for HITL approval."""
    lines = ["Architecture decisions:"]
    lines.extend(f"  - {decision}" for decision in plan.architecture_decisions)
    lines.append("\nTasks:")
    for task in plan.tasks:
        deps = ", ".join(task.dependencies) if task.dependencies else "none"
        lines.append(f"  - [{task.id}] {task.description} (deps: {deps})")
    return "\n".join(lines)


def get_ready_task_ids(tasks: list[Task], completed: set[str]) -> list[str]:
    """Return tasks whose dependencies are satisfied and that are not yet completed."""
    ready: list[str] = []
    for task in tasks:
        if task.id in completed:
            continue
        if all(dep in completed for dep in task.dependencies):
            ready.append(task.id)
    return ready


def all_tasks_completed(tasks: list[Task], completed: set[str]) -> bool:
    return bool(tasks) and all(task.id in completed for task in tasks)


def find_task(tasks: list[Task], task_id: str) -> Task | None:
    for task in tasks:
        if task.id == task_id:
            return task
    return None


def expand_downstream(tasks: list[Task], root_ids: set[str]) -> set[str]:
    """Return root task ids plus all transitive dependents."""
    affected = set(root_ids)
    while True:
        expanded = False
        for task in tasks:
            if task.id in affected:
                continue
            if any(dep in affected for dep in task.dependencies):
                affected.add(task.id)
                expanded = True
        if not expanded:
            break
    return affected


def apply_partial_replan(existing: list[Task], revised: ExecutionPlan, completed: set[str]) -> list[Task]:
    """Merge a revised plan while preserving completed tasks outside the revision."""
    revised_ids = {task.id for task in revised.tasks}
    preserved = [task for task in existing if task.id in completed and task.id not in revised_ids]

    merged: list[Task] = list(preserved)
    seen = {task.id for task in merged}
    for task in revised.tasks:
        if task.id not in seen:
            merged.append(task)
            seen.add(task.id)
    return merged


def assemble_code_from_artifacts(artifacts: dict[str, str]) -> str:
    """Merge per-task file artifacts into the legacy multi-file code string."""
    if not artifacts:
        return ""
    blocks: list[str] = []
    for path in sorted(artifacts):
        ext = "xml" if path.endswith(".xml") else "java"
        if path.endswith((".yml", ".yaml", ".properties")):
            ext = path.rsplit(".", 1)[-1]
        blocks.append(f"### File: {path}\n```{ext}\n{artifacts[path].rstrip()}\n```")
    return "\n\n".join(blocks)


TASK_KEYWORD_MAP: dict[str, frozenset[str]] = {
    "entity": frozenset({"entity", "jpa", "domain", "model"}),
    "repository": frozenset({"repository", "jpa"}),
    "service": frozenset({"service", "transaction"}),
    "controller": frozenset({"controller", "rest", "endpoint", "mockmvc"}),
    "dto": frozenset({"dto", "request", "response", "validation"}),
    "exception": frozenset({"exception", "controlleradvice", "exceptions"}),
    "pom": frozenset({"pom", "dependency", "maven"}),
    "tests": frozenset({"test", "junit", "mockito"}),
    "application_main": frozenset({"application", "springbootapplication"}),
    "application_config": frozenset({"application", "properties", "yml", "config"}),
    "security": frozenset({"security", "jwt", "auth"}),
    "database": frozenset({"database", "schema", "datasource", "hikari"}),
    "payment_processing": frozenset({"stripe", "payment"}),
}

TASK_PATH_HINTS: dict[str, frozenset[str]] = {
    "pom": frozenset({"pom.xml"}),
    "entity": frozenset({"/model/", "/entity/", "/domain/"}),
    "repository": frozenset({"/repository/"}),
    "service": frozenset({"/service/"}),
    "controller": frozenset({"/controller/"}),
    "dto": frozenset({"/dto/"}),
    "exception": frozenset({"/exception", "/exceptions/"}),
    "application_config": frozenset(
        {"application.properties", "application.yml", "application.yaml", "/config/"}
    ),
    "tests": frozenset({"/test/"}),
    "security": frozenset({"/security/", "jwt", "auth"}),
    "database": frozenset({"database", "schema", "datasource", "hikari"}),
    "payment_processing": frozenset({"stripe", "payment"}),
}


def task_review_keywords(task_id: str) -> set[str]:
    """Return path/error keywords associated with a task id."""
    normalized = task_id.lower()
    keywords = set(TASK_KEYWORD_MAP.get(normalized, frozenset()))
    keywords.add(normalized)
    return keywords


def path_belongs_to_task(path: str, task_id: str, artifact_paths: set[str] | None = None) -> bool:
    """Return True when a file path is likely owned by the given task."""
    if artifact_paths:
        normalized_paths = {item.lower() for item in artifact_paths}
        lowered = path.lower()
        if lowered in normalized_paths:
            return True

    lowered = path.lower()
    hints = TASK_PATH_HINTS.get(task_id.lower(), frozenset({f"/{task_id.lower()}/"}))
    return any(hint in lowered for hint in hints)


def filter_errors_for_task(
    errors: str,
    task_id: str,
    artifact_paths: set[str] | None = None,
) -> list[str]:
    """Return error lines relevant to a single task (not the full project dump)."""
    if not errors.strip():
        return []

    matched: list[str] = []
    for line in errors.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        bracket = re.search(r"\[([^\]]+)\]", stripped)
        if bracket:
            if path_belongs_to_task(bracket.group(1), task_id, artifact_paths):
                matched.append(stripped)
            continue

        lowered = stripped.lower()
        keywords = task_review_keywords(task_id)
        if any(re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in keywords):
            matched.append(stripped)

    return matched


def infer_invalidated_task_ids(errors: str, tasks: list[Task]) -> set[str]:
    """Best-effort mapping from reviewer errors to task ids."""
    if not errors.strip():
        return set()

    matched: set[str] = set()
    for line in errors.splitlines():
        bracket = re.search(r"\[([^\]]+)\]", line)
        if not bracket:
            continue
        path = bracket.group(1)
        for task in tasks:
            if path_belongs_to_task(path, task.id):
                matched.add(task.id)

    if matched:
        return matched

    lowered = errors.lower()
    for task in tasks:
        keywords = task_review_keywords(task.id)
        if any(keyword in lowered for keyword in keywords):
            matched.add(task.id)

    return matched or {task.id for task in tasks}
