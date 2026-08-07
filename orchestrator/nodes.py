from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send, interrupt

from orchestrator.file_writer import parse_generated_java_files
from orchestrator.llm import (
    aggregate_artifact_reviews,
    get_chat_model,
    invoke_with_rate_limit_retry,
    message_text,
)
from orchestrator.state import OrchestratorState
from orchestrator.tasks import (
    ExecutionPlan,
    Task,
    all_tasks_completed,
    apply_partial_replan,
    assemble_code_from_artifacts,
    expand_downstream,
    filter_errors_for_task,
    find_task,
    format_plan_summary,
    get_ready_task_ids,
    infer_invalidated_task_ids,
    parse_execution_plan,
    tasks_from_dicts,
    tasks_to_dicts,
)

MAX_REVIEW_RETRIES = 3
DEPENDENCY_CONTEXT_CHAR_BUDGET = 6000
DEPENDENCY_FILE_CHAR_LIMIT = 800
REVIEW_CONTEXT_CHAR_BUDGET = 4000

PLANNER_SYSTEM_PROMPT = (
    "You are a Principal Java Architect. Decompose the requirement into a structured "
    "execution plan for Java 17 / Spring Boot 3.\n\n"
    "Output ONLY valid JSON (no markdown prose) with this schema:\n"
    "{\n"
    '  "architecture_decisions": ["short decision strings"],\n'
    '  "tasks": [\n'
    '    {"id": "pom", "description": "Create pom.xml with Spring Boot 3 deps", "dependencies": []},\n'
    '    {"id": "entity", "description": "Create domain entity classes", "dependencies": ["pom"]}\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- One focused deliverable per task (controller, service, repository, entity, dto, "
    "exception handler, config, tests, pom.xml, application.yml).\n"
    "- Use stable snake_case or short ids (pom, entity, repository, service, controller, "
    "dto, exception, application_config, tests).\n"
    "- Keep each description under 200 characters.\n"
    "- Declare dependencies explicitly so independent tasks (e.g. dto + entity + pom) "
    "can run in parallel.\n"
    "- Prefer 8-15 granular tasks over one monolithic step."
)

PARTIAL_REPLAN_PROMPT = (
    "You are a Principal Java Architect updating an existing execution plan.\n"
    "Preserve completed work. Revise ONLY tasks affected by human feedback.\n"
    "Output ONLY valid JSON using the same schema as the initial planner.\n"
    "Do not regenerate tasks that are already completed unless feedback requires it."
)

CODER_SYSTEM_PROMPT = (
    "You are a Senior Java Developer implementing ONE task at a time for a Java 17 / "
    "Spring Boot 3 project.\n\n"
    "Output requirements:\n"
    "- Emit only the files needed for the current task using this format per file:\n"
    "  ### File: src/main/java/com/example/.../Example.java\n"
    "  ```java\n"
    "  // code\n"
    "  ```\n"
    "- Use constructor injection, Jakarta validation, and layered architecture.\n"
    "- Reuse patterns from dependency task outputs; do not duplicate unrelated files.\n"
    "- When review findings are provided, fix issues in files you own for this task.\n"
    "- Do not include prose outside file blocks."
)

ARTIFACT_REVIEWER_PROMPT = (
    "You are a Lead Java Security & QA Auditor reviewing ONE project file at a time.\n"
    "Audit the file against the requirement, task scope, and Spring Boot 3 best practices.\n"
    "Check syntax, dependencies, validation, HTTP codes, JPA usage, and test quality as relevant.\n"
    "If the file is clean with no material issues, respond with exactly: APPROVED\n"
    "Otherwise return a concise bulleted list prefixed with '- '."
)

def has_errors(state: OrchestratorState) -> bool:
    return bool(state.get("errors", "").strip())


def _normalize_requirement(requirement: str) -> str:
    collapsed = " ".join(requirement.split())
    if len(collapsed) <= 500:
        return collapsed
    return collapsed[:497] + "..."


def _build_plan_from_state(state: OrchestratorState) -> ExecutionPlan:
    return ExecutionPlan(
        architecture_decisions=state.get("architecture_decisions", []),
        tasks=tasks_from_dicts(state.get("tasks")),
    )


def planner(state: OrchestratorState) -> OrchestratorState:
    """Produce a structured JSON execution plan from the requirement."""
    requirement = state.get("requirement", "").strip()
    feedback = state.get("human_feedback", "").strip()
    existing_tasks = tasks_from_dicts(state.get("tasks"))
    completed_ids = set(state.get("completed_tasks", {}))

    if not requirement:
        return {
            "plan": "No requirement provided.",
            "tasks": [],
            "architecture_decisions": [],
        }

    is_partial_replan = bool(
        feedback and feedback.lower().startswith("rejected") and existing_tasks
    )

    human_parts = [f"Requirement:\n{_normalize_requirement(requirement)}"]
    system_prompt = PLANNER_SYSTEM_PROMPT

    if is_partial_replan:
        system_prompt = PARTIAL_REPLAN_PROMPT
        human_parts.append(f"\nHuman feedback:\n{feedback}")
        human_parts.append("\nExisting tasks:")
        for task in existing_tasks:
            status = "completed" if task.id in completed_ids else "pending"
            human_parts.append(
                f"- {task.id}: {task.description} (status={status}, deps={task.dependencies})"
            )

    llm = get_chat_model()
    response = invoke_with_rate_limit_retry(
        llm,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(human_parts)),
        ],
    )

    try:
        execution_plan = parse_execution_plan(message_text(response.content))
    except (ValueError, Exception):
        execution_plan = ExecutionPlan(
            architecture_decisions=["Spring Boot 3 layered monolith"],
            tasks=[
                Task(
                    id="pom",
                    description="Create pom.xml with Spring Boot 3 dependencies",
                    dependencies=[],
                ),
                Task(
                    id="application_main",
                    description="Create @SpringBootApplication entry point",
                    dependencies=["pom"],
                ),
                Task(
                    id="entity",
                    description="Create JPA entity classes",
                    dependencies=["pom"],
                ),
                Task(
                    id="repository",
                    description="Create Spring Data JPA repositories",
                    dependencies=["entity"],
                ),
                Task(
                    id="dto",
                    description="Create request/response DTOs with validation",
                    dependencies=["pom"],
                ),
                Task(
                    id="service",
                    description="Create service interface and implementation",
                    dependencies=["repository", "dto"],
                ),
                Task(
                    id="controller",
                    description="Create REST controllers",
                    dependencies=["service", "dto"],
                ),
                Task(
                    id="exception",
                    description="Create @ControllerAdvice exception handling",
                    dependencies=["controller"],
                ),
                Task(
                    id="tests",
                    description="Create JUnit 5 and MockMvc tests",
                    dependencies=["controller", "service"],
                ),
            ],
        )

    if is_partial_replan:
        merged_tasks = apply_partial_replan(
            existing_tasks, execution_plan, set(state.get("completed_tasks", {}))
        )
    else:
        merged_tasks = execution_plan.tasks

    plan_summary = format_plan_summary(
        ExecutionPlan(
            architecture_decisions=execution_plan.architecture_decisions,
            tasks=merged_tasks,
        )
    )

    return {
        "normalized_requirement": _normalize_requirement(requirement),
        "architecture_decisions": execution_plan.architecture_decisions,
        "tasks": tasks_to_dicts(merged_tasks),
        "plan": plan_summary,
        "errors": "",
    }


def approval_node(state: OrchestratorState) -> OrchestratorState:
    decision = interrupt(
        {
            "type": "plan_approval",
            "message": (
                "Review the task execution plan. Resume with 'Approved' or "
                "'Rejected: <feedback>'."
            ),
            "plan": state.get("plan", ""),
            "tasks": state.get("tasks", []),
            "architecture_decisions": state.get("architecture_decisions", []),
        }
    )
    return {"human_feedback": str(decision).strip()}


def route_after_approval(
    state: OrchestratorState,
) -> Literal["planner", "schedule_tasks"]:
    feedback = state.get("human_feedback", "")
    if feedback.lower().startswith("rejected"):
        return "planner"
    return "schedule_tasks"


def schedule_tasks(state: OrchestratorState) -> OrchestratorState:
    """Select the next wave of ready tasks; no LLM call."""
    return {}


def _coder_send_state(state: OrchestratorState, task_id: str) -> OrchestratorState:
    """Build coder input; LangGraph Send does not auto-merge parent state."""
    return {
        "current_task_id": task_id,
        "tasks": state.get("tasks", []),
        "requirement": state.get("requirement", ""),
        "normalized_requirement": state.get("normalized_requirement", ""),
        "architecture_decisions": state.get("architecture_decisions", []),
        "completed_tasks": state.get("completed_tasks", {}),
        "artifact_reviews": state.get("artifact_reviews", {}),
        "errors": state.get("errors", ""),
    }


def _cap_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated for token budget]"


def _dependency_context(state: OrchestratorState, task_id: str) -> str:
    task = find_task(tasks_from_dicts(state.get("tasks")), task_id)
    if not task:
        return ""

    remaining = DEPENDENCY_CONTEXT_CHAR_BUDGET
    parts: list[str] = []
    for dep_id in task.dependencies:
        if remaining <= 0:
            break

        dep_files = _artifacts_for_task(state, dep_id)
        if dep_files:
            for path, content in sorted(dep_files.items()):
                if remaining <= 0:
                    break
                limit = min(DEPENDENCY_FILE_CHAR_LIMIT, remaining)
                snippet = _cap_text(content.strip(), limit)
                part = f"--- {path} (from task '{dep_id}') ---\n{snippet}"
                parts.append(part)
                remaining -= len(part)
            continue

        dep_output = state.get("completed_tasks", {}).get(dep_id, "").strip()
        if not dep_output:
            continue
        limit = min(DEPENDENCY_FILE_CHAR_LIMIT, remaining)
        preview = _cap_text(dep_output, limit)
        part = f"--- Dependency task '{dep_id}' output ---\n{preview}"
        parts.append(part)
        remaining -= len(part)

    return "\n\n".join(parts)


def _review_context_for_task(state: OrchestratorState, task_id: str) -> str:
    """Review findings scoped to the current task's artifacts."""
    task_paths = set(_artifacts_for_task(state, task_id).keys())
    sections: list[str] = []

    for path in sorted(task_paths):
        review = state.get("artifact_reviews", {}).get(path, "").strip()
        if review and review.upper() != "APPROVED":
            sections.append(f"File: {path}\n{review}")

    if sections:
        return _cap_text("\n\n".join(sections), REVIEW_CONTEXT_CHAR_BUDGET)

    relevant_errors = filter_errors_for_task(
        state.get("errors", ""),
        task_id,
        task_paths or None,
    )
    if not relevant_errors:
        return ""

    return _cap_text("\n".join(relevant_errors), REVIEW_CONTEXT_CHAR_BUDGET)


def fan_out_tasks(state: OrchestratorState) -> list[Send] | Literal["assemble_code"]:
    """Dispatch the next ready task to the coder (one per wave for Groq TPM limits)."""
    tasks = tasks_from_dicts(state.get("tasks"))
    completed = set(state.get("completed_tasks", {}))
    ready = get_ready_task_ids(tasks, completed)

    if ready:
        return [Send("coder", _coder_send_state(state, ready[0]))]

    if all_tasks_completed(tasks, completed):
        return "assemble_code"

    pending = [
        task.id for task in tasks if task.id not in completed
    ]
    raise RuntimeError(
        f"Task scheduling deadlock; pending tasks with unsatisfied dependencies: {pending}"
    )


def coder(state: OrchestratorState) -> OrchestratorState:
    """Generate code for a single task with minimal context."""
    task_id = state.get("current_task_id", "").strip()
    tasks = tasks_from_dicts(state.get("tasks"))
    task = find_task(tasks, task_id)

    if not task:
        return {"completed_tasks": {task_id or "unknown": "// Task not found."}}

    human_parts = [
        f"Normalized requirement:\n{state.get('normalized_requirement') or state.get('requirement', '')}",
        "Architecture decisions:",
        *[f"- {item}" for item in state.get("architecture_decisions", [])],
        f"\nCurrent task [{task.id}]:\n{task.description}",
    ]

    dep_context = _dependency_context(state, task.id)
    if dep_context:
        human_parts.append(f"\nCompleted dependency outputs:\n{dep_context}")

    review_context = _review_context_for_task(state, task.id)
    if review_context:
        human_parts.append(
            f"\nReview findings to fix for this task:\n{review_context}"
        )

    llm = get_chat_model()
    response = invoke_with_rate_limit_retry(
        llm,
        [
            SystemMessage(content=CODER_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(human_parts)),
        ],
    )
    output = message_text(response.content).strip()
    parsed_files = parse_generated_java_files(output)

    return {
        "completed_tasks": {task.id: output},
        "artifacts": parsed_files,
        "current_task_id": task.id,
    }


def _collect_artifacts(state: OrchestratorState) -> dict[str, str]:
    """Gather file artifacts from state, falling back to completed task outputs."""
    artifacts = dict(state.get("artifacts", {}))
    if artifacts:
        return artifacts

    for output in state.get("completed_tasks", {}).values():
        if output.strip():
            artifacts.update(parse_generated_java_files(output))
    return artifacts


def assemble_code(state: OrchestratorState) -> OrchestratorState:
    """Merge artifacts into the legacy code field for review and file writing."""
    artifacts = _collect_artifacts(state)
    completed = state.get("completed_tasks", {})

    if artifacts:
        code = assemble_code_from_artifacts(artifacts)
    elif completed:
        # Parser found no ### File blocks; preserve raw coder output for review.
        code = "\n\n".join(output for output in completed.values() if output.strip())
    else:
        code = ""

    return {
        "code": code,
        "artifacts": {"__replace__": "1", **artifacts},
    }


def _artifacts_for_task(state: OrchestratorState, task_id: str) -> dict[str, str]:
    """Return file artifacts produced by a single coder task."""
    task_output = state.get("completed_tasks", {}).get(task_id, "").strip()
    if not task_output:
        return {}

    files = parse_generated_java_files(task_output)
    if files:
        return files
    return {f"task:{task_id}": task_output}


def reviewer(state: OrchestratorState) -> OrchestratorState:
    """Review each artifact from the current coder task, one file at a time."""
    task_id = state.get("current_task_id", "").strip()
    tasks = tasks_from_dicts(state.get("tasks"))
    task = find_task(tasks, task_id)
    files = _artifacts_for_task(state, task_id)

    if not files:
        return {}

    requirement = state.get("normalized_requirement") or state.get("requirement", "").strip()
    llm = get_chat_model()
    reviews: dict[str, str] = {}

    for path, content in files.items():
        response = invoke_with_rate_limit_retry(
            llm,
            [
                SystemMessage(content=ARTIFACT_REVIEWER_PROMPT),
                HumanMessage(
                    content=(
                        f"Requirement:\n{requirement or '(not specified)'}\n\n"
                        f"Task id: {task_id}\n"
                        f"Task scope: {task.description if task else '(unknown)'}\n\n"
                        f"File: {path}\n"
                        f"Contents:\n{content}"
                    )
                ),
            ],
        )
        reviews[path] = message_text(response.content).strip()
        status = reviews[path][:80].replace("\n", " ")
        print(f"DEBUG: Reviewed artifact {path!r} -> {status}")

    return {"artifact_reviews": reviews}


def summarize_reviews(state: OrchestratorState) -> OrchestratorState:
    """Aggregate per-artifact reviews into a final engineering summary."""
    artifact_reviews = state.get("artifact_reviews", {})
    if not artifact_reviews:
        return {
            "errors": "No artifacts were reviewed.",
            "review_summary": "",
        }

    return aggregate_artifact_reviews(artifact_reviews)


def invalidate_tasks(state: OrchestratorState) -> OrchestratorState:
    """Mark affected tasks pending and drop their artifacts before retry."""
    tasks = tasks_from_dicts(state.get("tasks"))
    root_ids = infer_invalidated_task_ids(state.get("errors", ""), tasks)
    affected = expand_downstream(tasks, root_ids)

    completed = {
        task_id: output
        for task_id, output in state.get("completed_tasks", {}).items()
        if task_id not in affected
    }
    artifacts = _collect_artifacts({"completed_tasks": completed, "artifacts": {}})

    affected_paths: set[str] = set()
    for task_id in affected:
        affected_paths.update(_artifacts_for_task(state, task_id).keys())

    remaining_reviews = {
        path: review
        for path, review in state.get("artifact_reviews", {}).items()
        if path not in affected_paths
    }

    return {
        "completed_tasks": {"__replace__": "1", **completed},
        "artifacts": {"__replace__": "1", **artifacts},
        "artifact_reviews": {"__replace__": "1", **remaining_reviews},
        "code": assemble_code_from_artifacts(artifacts),
    }


def increment_retry(state: OrchestratorState) -> OrchestratorState:
    return {"retry_count": state.get("retry_count", 0) + 1}


def route_after_reviewer(
    state: OrchestratorState,
) -> Literal["invalidate_tasks", "__end__"]:
    if not has_errors(state):
        return "__end__"

    if state.get("retry_count", 0) < MAX_REVIEW_RETRIES:
        return "invalidate_tasks"

    return "__end__"
