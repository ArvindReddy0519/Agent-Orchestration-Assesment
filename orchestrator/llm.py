"""Chat model factory for orchestrator nodes."""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def ensure_groq_api_key() -> str:
    """Verify Groq credentials are present before invoking the graph."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Create a free key at https://console.groq.com/ "
            "and export it: export GROQ_API_KEY='your-key-here'"
        )
    return api_key


def get_chat_model() -> BaseChatModel:
    """Return a Groq chat model configured from environment variables."""
    api_key = ensure_groq_api_key()
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        api_key=api_key,
        temperature=0,
    )


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
    if not stripped or stripped.upper() == "NO_FINDINGS":
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
