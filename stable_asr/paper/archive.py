"""Create publishable archives for paper artifact bundles."""

from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

from stable_asr.paper.integrity import verify_artifact_integrity
from stable_asr.paper.suites import audit_benchmark_required_artifacts, load_benchmark_suite


@dataclass(frozen=True)
class PaperArchiveReport:
    ok: bool
    artifacts_dir: str
    archive_path: str
    sha256_path: str
    root_name: str
    size_bytes: int
    sha256: str
    files: list[str]
    integrity_ok: bool
    required_artifacts_ok: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "artifacts_dir": self.artifacts_dir,
            "archive_path": self.archive_path,
            "sha256_path": self.sha256_path,
            "root_name": self.root_name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "files": self.files,
            "integrity_ok": self.integrity_ok,
            "required_artifacts_ok": self.required_artifacts_ok,
        }

    def to_text(self) -> str:
        lines = [
            f"paper_archive: {'OK' if self.ok else 'FAILED'}",
            f"archive: {self.archive_path}",
            f"sha256: {self.sha256}",
            f"sha256_file: {self.sha256_path}",
            f"files: {len(self.files)}",
            f"size_bytes: {self.size_bytes}",
            f"integrity_ok: {self.integrity_ok}",
            f"required_artifacts_ok: {self.required_artifacts_ok}",
        ]
        return "\n".join(lines)


def paper_artifact_archive(
    artifacts_dir: str | Path,
    output_path: str | Path,
    *,
    sha256_path: str | Path | None = None,
    root_name: str = "stable-asr-artifacts",
    require_valid: bool = True,
) -> PaperArchiveReport:
    """Archive an audited paper artifact directory as deterministic tar.gz."""

    artifacts_dir = Path(artifacts_dir)
    output_path = Path(output_path)
    sha256_path = Path(sha256_path) if sha256_path is not None else output_path.with_suffix(output_path.suffix + ".sha256")
    root_name = root_name.strip().strip("/")
    if not root_name:
        raise ValueError("root_name must be non-empty")

    integrity = verify_artifact_integrity(artifacts_dir / "artifact_hashes.json", root=artifacts_dir)
    suite = load_benchmark_suite(artifacts_dir / "benchmark_suite.json")
    required = audit_benchmark_required_artifacts(artifacts_dir, suite=suite)
    if require_valid and not integrity.ok:
        raise ValueError("artifact integrity verification failed")
    if require_valid and not required.ok:
        raise ValueError("benchmark required artifact audit failed: " + ", ".join(required.missing[:5]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sha256_path.parent.mkdir(parents=True, exist_ok=True)
    excluded = {output_path.resolve(), sha256_path.resolve()}
    files = _artifact_files(artifacts_dir, excluded=excluded)

    with output_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gzip_file:
            with tarfile.open(fileobj=gzip_file, mode="w") as archive:
                for file_path in files:
                    relative = file_path.relative_to(artifacts_dir)
                    info = archive.gettarinfo(str(file_path), arcname=f"{root_name}/{relative.as_posix()}")
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with file_path.open("rb") as handle:
                        archive.addfile(info, handle)

    sha256 = _sha256_file(output_path)
    sha256_path.write_text(f"{sha256}  {output_path.name}\n", encoding="utf-8")
    relative_files = [str(path.relative_to(artifacts_dir)) for path in files]
    return PaperArchiveReport(
        ok=integrity.ok and required.ok,
        artifacts_dir=str(artifacts_dir),
        archive_path=str(output_path),
        sha256_path=str(sha256_path),
        root_name=root_name,
        size_bytes=output_path.stat().st_size,
        sha256=sha256,
        files=relative_files,
        integrity_ok=integrity.ok,
        required_artifacts_ok=required.ok,
    )


def load_paper_archive_report(path: str | Path) -> PaperArchiveReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PaperArchiveReport(
        ok=bool(payload["ok"]),
        artifacts_dir=str(payload["artifacts_dir"]),
        archive_path=str(payload["archive_path"]),
        sha256_path=str(payload["sha256_path"]),
        root_name=str(payload["root_name"]),
        size_bytes=int(payload["size_bytes"]),
        sha256=str(payload["sha256"]),
        files=[str(item) for item in payload.get("files", [])],
        integrity_ok=bool(payload["integrity_ok"]),
        required_artifacts_ok=bool(payload["required_artifacts_ok"]),
    )


def write_paper_archive_report(report: PaperArchiveReport, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(output_path)


def _artifact_files(artifacts_dir: Path, *, excluded: set[Path]) -> list[Path]:
    files = []
    for path in artifacts_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() in excluded:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(artifacts_dir).as_posix())


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
