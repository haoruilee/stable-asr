"""Provenance manifests for paper artifact bundles."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from stable_asr import __version__


DEFAULT_PROVENANCE_CONFIGS = (
    "configs/paper/paper_smoke.json",
    "configs/paper/paper_parity_checklist.json",
    "configs/paper/final_experiments.json",
    "configs/final/paper_final.json",
    "configs/final/asr_command_compare.json",
    "configs/benchmarks/stable_asr_v0.json",
    "configs/datasets/stable_asr_sources.json",
    "configs/adapters/stable_asr_adapters.json",
    "configs/references/asr_collections.json",
    "configs/scenarios/stable_asr_voiceworld_v0.json",
    "configs/roadmap/stable_asr_roadmap.json",
)


@dataclass(frozen=True)
class ProvenanceFile:
    path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class GitProvenance:
    available: bool
    root: str | None
    branch: str | None
    commit: str | None
    remote: str | None
    dirty: bool
    status: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "root": self.root,
            "branch": self.branch,
            "commit": self.commit,
            "remote": self.remote,
            "dirty": self.dirty,
            "status": self.status,
        }


@dataclass(frozen=True)
class PaperProvenanceReport:
    generated_at_utc: str
    stable_asr_version: str
    python_version: str
    platform: str
    cwd: str
    output_dir: str
    git: GitProvenance
    results: ProvenanceFile
    configs: list[ProvenanceFile]
    commands: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "stable_asr_version": self.stable_asr_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "cwd": self.cwd,
            "output_dir": self.output_dir,
            "git": self.git.to_dict(),
            "results": self.results.to_dict(),
            "configs": [item.to_dict() for item in self.configs],
            "commands": self.commands,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Stable-ASR Paper Provenance",
            "",
            f"generated_at_utc: `{self.generated_at_utc}`",
            f"stable_asr_version: `{self.stable_asr_version}`",
            f"python_version: `{self.python_version}`",
            f"platform: `{self.platform}`",
            f"cwd: `{self.cwd}`",
            f"output_dir: `{self.output_dir}`",
            "",
            "## Git",
            "",
            f"available: `{self.git.available}`",
            f"root: `{self.git.root or ''}`",
            f"branch: `{self.git.branch or ''}`",
            f"commit: `{self.git.commit or ''}`",
            f"remote: `{self.git.remote or ''}`",
            f"dirty: `{self.git.dirty}`",
            "",
        ]
        if self.git.status:
            lines.extend(["Status:", ""])
            lines.extend(f"- `{entry}`" for entry in self.git.status)
            lines.append("")
        lines.extend(
            [
                "## Results",
                "",
                "| path | exists | bytes | sha256 |",
                "| --- | --- | ---: | --- |",
                _file_row(self.results),
                "",
                "## Configs",
                "",
                "| path | exists | bytes | sha256 |",
                "| --- | --- | ---: | --- |",
            ]
        )
        lines.extend(_file_row(item) for item in self.configs)
        lines.extend(["", "## Commands", ""])
        lines.extend(f"- `{command}`" for command in self.commands)
        lines.append("")
        return "\n".join(lines)


def paper_bundle_provenance(
    results_path: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    config_paths: Iterable[str | Path] = DEFAULT_PROVENANCE_CONFIGS,
) -> PaperProvenanceReport:
    """Build provenance metadata for a paper bundle."""

    repo_root = Path(repo_root)
    output_dir = Path(output_dir)
    results_path = Path(results_path)
    commands = [
        "stable-asr reproduce-paper --config configs/paper/paper_smoke.json",
        f"stable-asr paper-bundle --results {results_path} --output-dir {output_dir}",
        f"stable-asr paper-audit --results {results_path} --artifacts-dir {output_dir}",
        f"stable-asr paper-artifact-integrity --manifest {output_dir / 'artifact_hashes.json'} --root {output_dir}",
        f"stable-asr paper-release-audit --repo-root {repo_root} --results {results_path} --artifacts-dir {output_dir}",
    ]
    return PaperProvenanceReport(
        generated_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        stable_asr_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cwd=str(Path.cwd()),
        output_dir=str(output_dir),
        git=_git_provenance(repo_root),
        results=_file_provenance(results_path, base=repo_root),
        configs=[_file_provenance(Path(path), base=repo_root) for path in config_paths],
        commands=commands,
    )


def write_paper_provenance(
    report: PaperProvenanceReport,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, str]:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _git_provenance(repo_root: Path) -> GitProvenance:
    root = _git(repo_root, "rev-parse", "--show-toplevel")
    if root is None:
        return GitProvenance(
            available=False,
            root=None,
            branch=None,
            commit=None,
            remote=None,
            dirty=False,
            status=[],
        )
    status = (_git(repo_root, "status", "--short") or "").splitlines()
    return GitProvenance(
        available=True,
        root=root,
        branch=_git(repo_root, "branch", "--show-current"),
        commit=_git(repo_root, "rev-parse", "HEAD"),
        remote=_git(repo_root, "remote", "get-url", "origin"),
        dirty=bool(status),
        status=status,
    )


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _file_provenance(path: Path, *, base: Path) -> ProvenanceFile:
    resolved = path if path.is_absolute() else base / path
    display = _display_path(path, base)
    if not resolved.exists() or not resolved.is_file():
        return ProvenanceFile(path=display, exists=False, size_bytes=None, sha256=None)
    return ProvenanceFile(
        path=display,
        exists=True,
        size_bytes=resolved.stat().st_size,
        sha256=_sha256_file(resolved),
    )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _display_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _file_row(item: ProvenanceFile) -> str:
    size = "" if item.size_bytes is None else str(item.size_bytes)
    sha = "" if item.sha256 is None else item.sha256
    return f"| `{item.path}` | `{item.exists}` | {size} | `{sha}` |"
