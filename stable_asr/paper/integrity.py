"""Artifact hash manifests for paper bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ArtifactDigest:
    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ArtifactIntegrityReport:
    ok: bool
    root: str
    files: list[ArtifactDigest]
    missing: list[str]
    mismatched: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "root": self.root,
            "files": [digest.to_dict() for digest in self.files],
            "missing": self.missing,
            "mismatched": self.mismatched,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Stable-ASR Artifact Integrity",
            "",
            f"status: `{'OK' if self.ok else 'FAILED'}`",
            f"root: `{self.root}`",
            f"files: `{len(self.files)}`",
            "",
        ]
        if self.missing:
            lines.extend(["## Missing", ""])
            lines.extend(f"- `{path}`" for path in self.missing)
            lines.append("")
        if self.mismatched:
            lines.extend(["## Mismatched", ""])
            lines.extend(f"- `{path}`" for path in self.mismatched)
            lines.append("")
        lines.extend(
            [
                "## Files",
                "",
                "| path | bytes | sha256 |",
                "| --- | ---: | --- |",
            ]
        )
        for digest in self.files:
            lines.append(f"| `{digest.path}` | {digest.size_bytes} | `{digest.sha256}` |")
        lines.append("")
        return "\n".join(lines)


def artifact_integrity_manifest(
    paths: Iterable[str | Path],
    *,
    root: str | Path,
) -> ArtifactIntegrityReport:
    """Hash artifact files relative to a bundle root."""

    root_path = Path(root)
    files: list[ArtifactDigest] = []
    missing: list[str] = []
    for raw_path in sorted({str(path) for path in paths}):
        path = Path(raw_path)
        resolved = path if path.is_absolute() else root_path / path
        display = _display_path(path, root_path)
        if not resolved.exists() or not resolved.is_file():
            missing.append(display)
            continue
        files.append(
            ArtifactDigest(
                path=display,
                size_bytes=resolved.stat().st_size,
                sha256=_sha256_file(resolved),
            )
        )
    return ArtifactIntegrityReport(
        ok=not missing,
        root=str(root_path),
        files=files,
        missing=missing,
        mismatched=[],
    )


def write_artifact_integrity(
    report: ArtifactIntegrityReport,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, str]:
    """Write a JSON and Markdown artifact integrity report."""

    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def load_artifact_integrity(path: str | Path) -> ArtifactIntegrityReport:
    """Load an artifact integrity report from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    files = [
        ArtifactDigest(
            path=str(item["path"]),
            size_bytes=int(item["size_bytes"]),
            sha256=str(item["sha256"]),
        )
        for item in payload.get("files", [])
        if isinstance(item, dict)
    ]
    missing = [str(item) for item in payload.get("missing", [])]
    mismatched = [str(item) for item in payload.get("mismatched", [])]
    return ArtifactIntegrityReport(
        ok=bool(payload.get("ok")) and not missing and not mismatched,
        root=str(payload.get("root", Path(path).parent)),
        files=files,
        missing=missing,
        mismatched=mismatched,
    )


def verify_artifact_integrity(
    manifest_path: str | Path,
    *,
    root: str | Path | None = None,
) -> ArtifactIntegrityReport:
    """Verify a previously written artifact integrity JSON manifest."""

    manifest_path = Path(manifest_path)
    expected = load_artifact_integrity(manifest_path)
    root_path = Path(root) if root is not None else _resolve_verify_root(manifest_path, expected.root, expected.files)
    checked: list[ArtifactDigest] = []
    missing: list[str] = list(expected.missing)
    mismatched: list[str] = list(expected.mismatched)
    for digest in expected.files:
        path = Path(digest.path)
        resolved = path if path.is_absolute() else root_path / path
        if not resolved.exists() or not resolved.is_file():
            missing.append(digest.path)
            continue
        actual = ArtifactDigest(
            path=digest.path,
            size_bytes=resolved.stat().st_size,
            sha256=_sha256_file(resolved),
        )
        checked.append(actual)
        if actual.size_bytes != digest.size_bytes or actual.sha256 != digest.sha256:
            mismatched.append(digest.path)
    return ArtifactIntegrityReport(
        ok=not missing and not mismatched,
        root=str(root_path),
        files=checked,
        missing=missing,
        mismatched=mismatched,
    )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        if path.is_absolute():
            return str(path)
        return str(path)


def _resolve_verify_root(manifest_path: Path, root: str, files: list[ArtifactDigest]) -> Path:
    stored = Path(root)
    if stored.is_absolute():
        return stored
    candidates = [stored, manifest_path.parent / stored, manifest_path.parent]
    best = max(candidates, key=lambda candidate: _existing_digest_count(candidate, files))
    return best


def _existing_digest_count(root: Path, files: list[ArtifactDigest]) -> int:
    count = 0
    for digest in files:
        path = Path(digest.path)
        resolved = path if path.is_absolute() else root / path
        if resolved.exists() and resolved.is_file():
            count += 1
    return count
