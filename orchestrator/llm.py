"""Chat model factory for orchestrator nodes."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import APIError, AuthenticationError, Groq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
SUPPORTED_GROQ_MODELS = frozenset(
    {
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    }
)


def _verify_groq_connectivity(api_key: str) -> None:
    """Validate the bearer token using the official Groq Python client."""
    try:
        client = Groq(api_key=api_key)
        client.models.list()
        print("DEBUG: Groq API connectivity check status = 200 (models.list succeeded)")
    except AuthenticationError as exc:
        raise ValueError(
            "GROQ_API_KEY is invalid. Edit the `.env` file in the project root "
            "with a valid key from https://console.groq.com/keys"
        ) from exc
    except APIError as exc:
        print(f"WARNING: Groq API connectivity check failed: {exc}")
        print("WARNING: Continuing anyway; LangChain may still succeed.")
    except Exception as exc:
        print(f"WARNING: Groq connectivity check error: {exc}")
        print("WARNING: Continuing anyway; LangChain may still succeed.")


def _resolve_groq_model() -> str:
    """Return a supported Groq model name, falling back to the default if needed."""
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    if model not in SUPPORTED_GROQ_MODELS:
        print(
            f"DEBUG: Unsupported GROQ_MODEL={model!r}; "
            f"using {DEFAULT_GROQ_MODEL!r} instead"
        )
        return DEFAULT_GROQ_MODEL
    return model


def ensure_groq_api_key() -> str:
    """Verify Groq credentials are present before invoking the graph."""
    api_key = os.getenv("GROQ_API_KEY", "").strip("'\" ")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Create a free key at https://console.groq.com/ "
            "and export it: export GROQ_API_KEY='your-key-here'"
        )

    print(
        f"DEBUG: Loaded GROQ_API_KEY = {api_key[:7]}...{api_key[-4:]} "
        f"(Length: {len(api_key)})"
    )

    if "your_actual_key" in api_key or len(api_key) < 20:
        raise ValueError(
            "GROQ_API_KEY looks invalid (placeholder or too short). Edit the `.env` file "
            "in the project root with a real key from https://console.groq.com/keys"
        )

    _verify_groq_connectivity(api_key)
    return api_key


def get_chat_model() -> BaseChatModel:
    """Return a cached Groq chat model configured from environment variables."""
    global _chat_model
    if _chat_model is None:
        api_key = ensure_groq_api_key()
        model = _resolve_groq_model()
        print(f"DEBUG: Initializing ChatGroq with model={model!r}")
        _chat_model = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=0,
        )
    return _chat_model


def invoke_with_rate_limit_retry(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    *,
    max_retries: int = 5,
):
    """Invoke the chat model, backing off on Groq TPM rate limits."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            message = str(exc).lower()
            if "413" in message or "too large" in message or "reduce your message size" in message:
                raise
            if "rate_limit" not in message and "429" not in message:
                raise
            if attempt == max_retries - 1:
                raise
            wait_seconds = 25.0
            match = re.search(r"try again in ([\d.]+)s", str(exc), re.IGNORECASE)
            if match:
                wait_seconds = float(match.group(1)) + 1.0
            print(
                f"WARNING: Groq rate limit hit (attempt {attempt + 1}/{max_retries}); "
                f"sleeping {wait_seconds:.0f}s..."
            )
            time.sleep(wait_seconds)


_chat_model: BaseChatModel | None = None


def message_text(content: object) -> str:
    """Normalize AIMessage.content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def extract_python_code(text: str) -> str:
    """Pull Python from a markdown fence if the model wrapped the answer."""
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def parse_review_findings(text: str) -> list[str]:
    """Parse reviewer output into individual finding strings."""
    stripped = text.strip()
    if not stripped or stripped.upper() in ("NO_FINDINGS", "APPROVED"):
        return []

    findings: list[str] = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        if line:
            findings.append(line)
    return findings


def format_review_errors(text: str) -> str:
    """Normalize reviewer LLM output to a single errors string (empty if pass)."""
    findings = parse_review_findings(text)
    if not findings:
        return ""
    return "\n".join(f"- {finding}" for finding in findings)


_FAILURE_KEYWORDS = frozenset(
    {
        "security",
        "vulnerability",
        "critical",
        "syntax",
        "compile",
        "invalid",
        "missing",
        "incorrect",
        "broken",
        " must ",
        "required",
        "sql injection",
        "xss",
        "csrf",
        "runtime error",
        "does not",
        "doesn't",
        " fail",
        "failed",
        "error",
    }
)

_WARNING_KEYWORDS = frozenset(
    {
        "consider",
        "recommend",
        " should ",
        "could",
        "minor",
        "style",
        "optional",
        "improve",
        "prefer",
        "warning",
        "nit",
    }
)


def classify_review_finding(finding: str) -> str:
    """Classify a single review finding as 'failure' or 'warning'."""
    lowered = f" {finding.lower()} "
    if any(keyword in lowered for keyword in _FAILURE_KEYWORDS):
        return "failure"
    if any(keyword in lowered for keyword in _WARNING_KEYWORDS):
        return "warning"
    return "failure"


def aggregate_artifact_reviews(artifact_reviews: dict[str, str]) -> dict[str, str]:
    """Build review_summary and errors from per-artifact reviews without an LLM."""
    approved_paths: list[str] = []
    warning_entries: list[tuple[str, list[str]]] = []
    failure_entries: list[tuple[str, list[str]]] = []

    for path in sorted(artifact_reviews):
        result = artifact_reviews[path].strip()
        if result.upper() == "APPROVED":
            approved_paths.append(path)
            continue

        findings = parse_review_findings(result)
        if not findings:
            findings = [result] if result else ["Review returned no actionable findings."]

        warning_findings: list[str] = []
        failure_findings: list[str] = []
        for finding in findings:
            if classify_review_finding(finding) == "warning":
                warning_findings.append(finding)
            else:
                failure_findings.append(finding)

        if failure_findings:
            failure_entries.append((path, failure_findings))
        if warning_findings:
            warning_entries.append((path, warning_findings))

    reviewed_count = len(artifact_reviews)
    approved_count = len(approved_paths)
    warning_artifact_count = len(warning_entries)
    failure_artifact_count = len(failure_entries)

    lines = [
        "Engineering Review Summary",
        "==========================",
        (
            f"Reviewed: {reviewed_count} | Approved: {approved_count} | "
            f"Warnings: {warning_artifact_count} | Failures: {failure_artifact_count}"
        ),
    ]

    if approved_paths:
        lines.extend(["", "Approved artifacts:"])
        lines.extend(f"- {path}" for path in approved_paths)

    if warning_entries:
        lines.extend(["", "Warnings:"])
        for path, findings in warning_entries:
            lines.append(f"- {path}:")
            lines.extend(f"  - {finding}" for finding in findings)

    if failure_entries:
        lines.extend(["", "Failures:"])
        for path, findings in failure_entries:
            lines.append(f"- {path}:")
            lines.extend(f"  - {finding}" for finding in findings)

    summary = "\n".join(lines)

    if not warning_entries and not failure_entries:
        return {"errors": "", "review_summary": summary}

    error_lines: list[str] = []
    for path, findings in failure_entries:
        for finding in findings:
            error_lines.append(f"- [{path}] {finding}")
    for path, findings in warning_entries:
        for finding in findings:
            error_lines.append(f"- [{path}] {finding}")

    return {"errors": "\n".join(error_lines), "review_summary": summary}
