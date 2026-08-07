"""Parse and persist multi-file Java / Spring Boot output from the coder node."""

from __future__ import annotations

import re
from pathlib import Path

FILE_BLOCK_PATTERN = re.compile(
    r"###\s*File:\s*(?P<path>\S+)\s*\n```(?:java|xml)?\s*\n(?P<content>.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def parse_generated_java_files(code_str: str) -> dict[str, str]:
    """Extract relative path -> source content mappings from coder LLM output."""
    files: dict[str, str] = {}
    for match in FILE_BLOCK_PATTERN.finditer(code_str):
        rel_path = match.group("path").strip()
        content = match.group("content").strip()
        if rel_path and content:
            files[rel_path] = f"{content}\n"
    return files


def save_generated_java_code(
    code_str: str,
    output_dir: str = "./generated_project",
) -> list[str]:
    """Parse `### File: ...` headers and write .java / pom.xml files to disk."""
    root = Path(output_dir)
    written: list[str] = []

    for rel_path, content in parse_generated_java_files(code_str).items():
        dest = root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        written.append(str(dest.resolve()))

    return written
