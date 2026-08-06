from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from orchestrator.llm import (
    extract_python_code,
    get_chat_model,
    message_text,
    parse_review_findings,
)
from orchestrator.state import OrchestratorState


def planner(state: OrchestratorState) -> OrchestratorState:
    """Produce an implementation plan from the requirement."""
    requirement = state.get("requirement", "").strip()
    feedback = state.get("human_feedback", "").strip()

    if not requirement:
        return {"plan": "No requirement provided."}

    human_parts = [f"Requirement:\n{requirement}"]
    if feedback and feedback.lower().startswith("rejected"):
        human_parts.append(
            f"\nThe previous plan was rejected. Revise the plan using this feedback:\n{feedback}"
        )

    llm = get_chat_model()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a Senior Software Architect. Given a product requirement, "
                    "produce a detailed step-by-step task list for implementation. "
                    "Include concrete deliverables, API endpoints, data models, and "
                    "testing steps where relevant. Use numbered steps and keep the plan "
                    "actionable for engineers."
                )
            ),
            HumanMessage(content="\n".join(human_parts)),
        ]
    )
    return {"plan": message_text(response.content).strip()}


def approval_node(state: OrchestratorState) -> OrchestratorState:
    """Pause for human review of the plan before coding begins."""
    plan = state.get("plan", "")
    decision = interrupt(
        {
            "type": "plan_approval",
            "message": (
                "Review the generated plan. To continue, resume with "
                "'Approved' or 'Rejected: <your feedback>'."
            ),
            "plan": plan,
        }
    )
    return {"human_feedback": str(decision).strip()}


def route_after_approval(
    state: OrchestratorState,
) -> Literal["planner", "coder"]:
    """Send rejected plans back to the planner; approved plans proceed to coding."""
    feedback = state.get("human_feedback", "")
    if feedback.lower().startswith("rejected"):
        return "planner"
    return "coder"


def coder(state: OrchestratorState) -> OrchestratorState:
    """Generate or update code according to the plan."""
    plan = state.get("plan", "").strip()
    requirement = state.get("requirement", "").strip()

    if not plan:
        return {"code": "# No plan available to implement."}

    llm = get_chat_model()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a Senior Software Engineer. Implement the approved plan as "
                    "complete, runnable Python using FastAPI. Output only raw Python source "
                    "code (no prose). Include imports, app setup, routes, and models as "
                    "needed. Do not wrap the code in markdown unless necessary."
                )
            ),
            HumanMessage(
                content=(
                    f"Original requirement:\n{requirement or '(not specified)'}\n\n"
                    f"Approved implementation plan:\n{plan}"
                )
            ),
        ]
    )
    code = extract_python_code(message_text(response.content))
    return {"code": code}


def reviewer(state: OrchestratorState) -> OrchestratorState:
    """Review code and record errors or approval signals."""
    code = state.get("code", "").strip()
    plan = state.get("plan", "").strip()
    requirement = state.get("requirement", "").strip()

    if not code:
        return {"errors": ["No code was generated to review."]}

    llm = get_chat_model()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a QA and application security auditor. Review the FastAPI "
                    "Python code against the requirement and plan. Report bugs, missing "
                    "features, unsafe patterns, and test gaps. If there are no material "
                    "issues, respond with exactly: NO_FINDINGS. Otherwise list each "
                    "finding on its own line prefixed with '- '."
                )
            ),
            HumanMessage(
                content=(
                    f"Requirement:\n{requirement or '(not specified)'}\n\n"
                    f"Plan:\n{plan or '(not specified)'}\n\n"
                    f"Code:\n{code}"
                )
            ),
        ]
    )
    findings = parse_review_findings(message_text(response.content))
    return {"errors": findings}
