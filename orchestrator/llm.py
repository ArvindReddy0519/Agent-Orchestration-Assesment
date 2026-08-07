"""Chat model factory for orchestrator nodes."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import APIError, AuthenticationError, Groq
from langchain_core.language_models.chat_models import BaseChatModel
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
    """Return a Groq chat model configured from environment variables."""
    api_key = ensure_groq_api_key()
    model = _resolve_groq_model()
    print(f"DEBUG: Initializing ChatGroq with model={model!r}")
    return ChatGroq(
        model=model,
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
