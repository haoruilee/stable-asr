"""Create publishable archives for paper artifact bundles."""

from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import PurePosixPath
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


@dataclass(frozen=True)
class PaperArchiveVerificationReport:
    ok: bool
    archive_path: str
    sha256_path: str
    root_name: str | None
    size_bytes: int
    sha256: str | None
    files: list[str]
    sha256_ok: bool
    safe_paths_ok: bool
    integrity_ok: bool
    required_artifacts_ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "archive_path": self.archive_path,
            "sha256_path": self.sha256_path,
            "root_name": self.root_name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "files": self.files,
            "sha256_ok": self.sha256_ok,
            "safe_paths_ok": self.safe_paths_ok,
            "integrity_ok": self.integrity_ok,
            "required_artifacts_ok": self.required_artifacts_ok,
            "errors": self.errors,
        }

    def to_text(self) -> str:
        lines = [
            f"paper_archive_verify: {'OK' if self.ok else 'FAILED'}",
            f"archive: {self.archive_path}",
            f"sha256_file: {self.sha256_path}",
            f"root_name: {self.root_name or ''}",
            f"files: {len(self.files)}",
            f"size_bytes: {self.size_bytes}",
            f"sha256_ok: {self.sha256_ok}",
            f"safe_paths_ok: {self.safe_paths_ok}",
            f"integrity_ok: {self.integrity_ok}",
            f"required_artifacts_ok: {self.required_artifacts_ok}",
        ]
        lines.extend(f"- {error}" for error in self.errors)
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


def verify_paper_artifact_archive(
    archive_path: str | Path,
    *,
    sha256_path: str | Path | None = None,
    expected_root: str | None = "stable-asr-artifacts",
) -> PaperArchiveVerificationReport:
    """Verify a paper artifact archive after transfer or publication."""

    archive_path = Path(archive_path)
    sha256_path = (
        Path(sha256_path)
        if sha256_path is not None
        else archive_path.with_suffix(archive_path.suffix + ".sha256")
    )
    errors: list[str] = []
    sha256 = _existing_sha256(archive_path, errors)
    expected_sha = _read_sha256_sidecar(sha256_path, errors)
    sha256_ok = sha256 is not None and expected_sha is not None and sha256 == expected_sha
    if expected_sha is not None and sha256 != expected_sha:
        errors.append(f"sha256 mismatch: expected {expected_sha}, got {sha256 or 'missing'}")

    members: list[tarfile.TarInfo] = []
    files: list[str] = []
    root_name: str | None = None
    safe_paths_ok = False
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            root_name, files, path_errors = _audit_tar_members(members, expected_root=expected_root)
            errors.extend(path_errors)
            safe_paths_ok = not path_errors
            if safe_paths_ok:
                with tempfile.TemporaryDirectory(prefix="stable-asr-archive-") as temp_dir:
                    root_dir = _extract_archive_files(archive, members, Path(temp_dir))
                    integrity = verify_artifact_integrity(root_dir / "artifact_hashes.json", root=root_dir)
                    suite = load_benchmark_suite(root_dir / "benchmark_suite.json")
                    required = audit_benchmark_required_artifacts(root_dir, suite=suite)
                    integrity_ok = integrity.ok
                    required_artifacts_ok = required.ok
                    if not integrity.ok:
                        errors.append("artifact integrity failed")
                    if not required.ok:
                        errors.append("required artifacts missing: " + ", ".join(required.missing[:5]))
            else:
                integrity_ok = False
                required_artifacts_ok = False
    except (OSError, tarfile.TarError, json.JSONDecodeError, KeyError, ValueError) as exc:
        errors.append(str(exc))
        integrity_ok = False
        required_artifacts_ok = False

    return PaperArchiveVerificationReport(
        ok=sha256_ok and safe_paths_ok and integrity_ok and required_artifacts_ok and not errors,
        archive_path=str(archive_path),
        sha256_path=str(sha256_path),
        root_name=root_name,
        size_bytes=archive_path.stat().st_size if archive_path.exists() else 0,
        sha256=sha256,
        files=files,
        sha256_ok=sha256_ok,
        safe_paths_ok=safe_paths_ok,
        integrity_ok=integrity_ok,
        required_artifacts_ok=required_artifacts_ok,
        errors=errors,
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


def _existing_sha256(path: Path, errors: list[str]) -> str | None:
    if not path.exists():
        errors.append(f"missing archive: {path}")
        return None
    try:
        return _sha256_file(path)
    except OSError as exc:
        errors.append(str(exc))
        return None


def _read_sha256_sidecar(path: Path, errors: list[str]) -> str | None:
    if not path.exists():
        errors.append(f"missing sha256 sidecar: {path}")
        return None
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except OSError as exc:
        errors.append(str(exc))
        return None
    line = lines[0] if lines else ""
    digest = line.split()[0] if line.split() else ""
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        errors.append(f"invalid sha256 sidecar: {path}")
        return None
    return digest.lower()


def _audit_tar_members(
    members: list[tarfile.TarInfo],
    *,
    expected_root: str | None,
) -> tuple[str | None, list[str], list[str]]:
    errors: list[str] = []
    files: list[str] = []
    roots: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        parts = path.parts
        if not parts or member.name.startswith("/") or ".." in parts:
            errors.append(f"unsafe archive path: {member.name}")
            continue
        roots.add(parts[0])
        if member.issym() or member.islnk():
            errors.append(f"archive links are not allowed: {member.name}")
        if not member.isfile() and not member.isdir():
            errors.append(f"unsupported archive member: {member.name}")
        if member.isfile():
            files.append(member.name)
    root_name = sorted(roots)[0] if len(roots) == 1 else None
    if not roots:
        errors.append("archive is empty")
    if len(roots) > 1:
        errors.append("archive must contain a single top-level root")
    if expected_root is not None and root_name != expected_root:
        errors.append(f"archive root mismatch: expected {expected_root}, got {root_name or 'none'}")
    return root_name, files, errors


def _extract_archive_files(
    archive: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    output_dir: Path,
) -> Path:
    roots: set[str] = set()
    for member in members:
        if not member.isfile():
            continue
        path = PurePosixPath(member.name)
        roots.add(path.parts[0])
        target = output_dir.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"cannot extract archive member: {member.name}")
        with source, target.open("wb") as handle:
            handle.write(source.read())
    if len(roots) != 1:
        raise ValueError("archive must contain a single top-level root")
    return output_dir / next(iter(roots))


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
