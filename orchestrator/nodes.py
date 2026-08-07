from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from orchestrator.llm import (
    format_review_errors,
    get_chat_model,
    message_text,
)
from orchestrator.state import OrchestratorState

MAX_REVIEW_RETRIES = 3

PLANNER_SYSTEM_PROMPT = (
    "You are a Principal Java Architect. Given a product requirement, produce a "
    "detailed, step-by-step implementation plan for a Java 17 / Spring Boot 3 "
    "URL shortener service. Break the work into standard Spring Boot components:\n"
    "- REST Controllers (@RestController, request mappings, @Valid input)\n"
    "- Service interfaces and @Service implementations\n"
    "- Spring Data JPA repositories\n"
    "- Domain entities (@Entity, relationships, constraints)\n"
    "- DTOs and request/response records for API boundaries\n"
    "- Exception handlers with @ControllerAdvice and @ExceptionHandler\n"
    "- pom.xml dependencies (spring-boot-starter-web, data-jpa, validation, test)\n"
    "- JUnit 5 tests (@SpringBootTest, MockMvc, repository/service unit tests)\n"
    "Use numbered steps, name concrete classes/packages, list REST endpoints with HTTP "
    "verbs and status codes, and keep the plan actionable for engineers."
)

CODER_SYSTEM_PROMPT = (
    "You are a Senior Java Developer. Implement the approved plan as idiomatic, "
    "production-ready Java 17 and Spring Boot 3 code for a URL shortener service.\n\n"
    "Output requirements:\n"
    "- Emit every source file using this exact format for each file:\n"
    "  ### File: src/main/java/com/example/urlshortener/controller/UrlController.java\n"
    "  ```java\n"
    "  // code here\n"
    "  ```\n"
    "- Include pom.xml, application properties/yml, all Java sources, and JUnit 5 tests.\n"
    "- Use constructor injection, Jakarta validation (@Valid), proper HTTP status codes, "
    "and layered architecture (controller -> service -> repository).\n"
    "- When prior review findings are provided, fix every listed issue in the regenerated files.\n"
    "- Do not include prose outside file blocks."
)

REVIEWER_SYSTEM_PROMPT = (
    "You are a Lead Java Security & QA Auditor. Review the generated Java 17 / "
    "Spring Boot 3 project against the requirement and plan.\n\n"
    "Audit for:\n"
    "- Spring Boot best practices (constructor injection, no field @Autowired)\n"
    "- Correct JPA entity/repository annotations and transaction boundaries\n"
    "- Request validation via @Valid and constraint annotations\n"
    "- Proper HTTP response codes and exception mapping via @ControllerAdvice\n"
    "- Exception safety, input sanitization, and missing test coverage\n"
    "- pom.xml dependency correctness for Spring Boot 3\n\n"
    "If the code is clean with no material issues, respond with exactly: APPROVED\n"
    "Otherwise return a detailed bulleted list; prefix each finding with '- '."
)


def has_errors(state: OrchestratorState) -> bool:
    """Strict check: non-empty errors string means review failed."""
    return bool(state.get("errors", "").strip())


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
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
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
    """Generate or update multi-file Java/Spring Boot code according to the plan."""
    plan = state.get("plan", "").strip()
    requirement = state.get("requirement", "").strip()
    prior_errors = state.get("errors", "").strip()

    if not plan:
        return {"code": "// No plan available to implement."}

    human_parts = [
        f"Original requirement:\n{requirement or '(not specified)'}",
        f"Approved implementation plan:\n{plan}",
    ]
    if prior_errors:
        human_parts.append(
            f"Previous review findings (fix these issues in every affected file):\n{prior_errors}"
        )

    llm = get_chat_model()
    response = llm.invoke(
        [
            SystemMessage(content=CODER_SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(human_parts)),
        ]
    )
    return {"code": message_text(response.content).strip()}


def reviewer(state: OrchestratorState) -> OrchestratorState:
    """Review code; set errors when issues exist, clear errors on pass."""
    code = state.get("code", "").strip()
    plan = state.get("plan", "").strip()
    requirement = state.get("requirement", "").strip()

    if not code:
        return {"errors": "No code was generated to review."}

    llm = get_chat_model()
    response = llm.invoke(
        [
            SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Requirement:\n{requirement or '(not specified)'}\n\n"
                    f"Plan:\n{plan or '(not specified)'}\n\n"
                    f"Generated project files:\n{code}"
                )
            ),
        ]
    )
    errors = format_review_errors(message_text(response.content))
    return {"errors": errors}


def increment_retry(state: OrchestratorState) -> OrchestratorState:
    """Bump retry_count before sending execution back to the coder."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def route_after_reviewer(
    state: OrchestratorState,
) -> Literal["increment_retry", "__end__"]:
    """Route to bounded coder retries or safe-stop at END."""
    if not has_errors(state):
        return "__end__"

    retry_count = state.get("retry_count", 0)
    if retry_count < MAX_REVIEW_RETRIES:
        return "increment_retry"

    return "__end__"
