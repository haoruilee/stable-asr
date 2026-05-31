"""Markdown report helpers for early experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MarkdownReport:
    title: str
    sections: list[tuple[str, str]] = field(default_factory=list)

    def add_section(self, heading: str, body: str) -> None:
        self.sections.append((heading, body))

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for heading, body in self.sections:
            lines.extend([f"## {heading}", "", body.rstrip(), ""])
        return "\n".join(lines).rstrip() + "\n"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")


def dict_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])

