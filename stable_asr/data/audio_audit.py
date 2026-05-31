"""Audio file audits for turn and ASR manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from stable_asr.data.audio import inspect_wav


class AudioManifestRecord(Protocol):
    id: str
    audio: str
    sample_rate: int


@dataclass(frozen=True)
class AudioAuditCheck:
    record_id: str
    audio: str
    resolved_path: str
    exists: bool
    inspectable: bool
    ok: bool
    expected_sample_rate: int
    actual_sample_rate: int | None
    expected_duration_sec: float | None
    actual_duration_sec: float | None
    issues: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "audio": self.audio,
            "resolved_path": self.resolved_path,
            "exists": self.exists,
            "inspectable": self.inspectable,
            "ok": self.ok,
            "expected_sample_rate": self.expected_sample_rate,
            "actual_sample_rate": self.actual_sample_rate,
            "expected_duration_sec": self.expected_duration_sec,
            "actual_duration_sec": self.actual_duration_sec,
            "issues": self.issues,
        }


@dataclass(frozen=True)
class AudioAuditReport:
    kind: str
    records: int
    checked_files: int
    missing_files: int
    uninspectable_files: int
    sample_rate_mismatches: int
    duration_mismatches: int
    checks: list[AudioAuditCheck]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "ok": self.ok,
            "records": self.records,
            "checked_files": self.checked_files,
            "missing_files": self.missing_files,
            "uninspectable_files": self.uninspectable_files,
            "sample_rate_mismatches": self.sample_rate_mismatches,
            "duration_mismatches": self.duration_mismatches,
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_text(self) -> str:
        lines = [
            f"audio_audit: {'OK' if self.ok else 'FAILED'}",
            f"kind: {self.kind}",
            f"records: {self.records}",
            f"checked_files: {self.checked_files}",
            f"missing_files: {self.missing_files}",
            f"uninspectable_files: {self.uninspectable_files}",
            f"sample_rate_mismatches: {self.sample_rate_mismatches}",
            f"duration_mismatches: {self.duration_mismatches}",
        ]
        failed = [check for check in self.checks if not check.ok]
        if failed:
            lines.append("issues:")
            for check in failed[:25]:
                lines.append(f"- {check.record_id}: {check.resolved_path} ({'; '.join(check.issues)})")
            if len(failed) > 25:
                lines.append(f"- ... {len(failed) - 25} more issue(s)")
        return "\n".join(lines)


def audit_audio_records(
    records: Iterable[AudioManifestRecord],
    *,
    kind: str,
    audio_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    expected_duration_getter: object | None = None,
    duration_tolerance_sec: float = 0.05,
    require_inspectable: bool = False,
) -> AudioAuditReport:
    checks: list[AudioAuditCheck] = []
    manifest_dir = Path(manifest_path).parent if manifest_path is not None else None
    root = Path(audio_root) if audio_root is not None else None

    for record in records:
        expected_duration = _expected_duration(record, expected_duration_getter=expected_duration_getter)
        check = _audit_one(
            record,
            root=root,
            manifest_dir=manifest_dir,
            expected_duration=expected_duration,
            duration_tolerance_sec=duration_tolerance_sec,
            require_inspectable=require_inspectable,
        )
        checks.append(check)

    return AudioAuditReport(
        kind=kind,
        records=len(checks),
        checked_files=sum(1 for check in checks if check.exists),
        missing_files=sum(1 for check in checks if not check.exists),
        uninspectable_files=sum(1 for check in checks if check.exists and not check.inspectable),
        sample_rate_mismatches=sum(1 for check in checks if "sample_rate_mismatch" in check.issues),
        duration_mismatches=sum(1 for check in checks if "duration_mismatch" in check.issues),
        checks=checks,
    )


def _audit_one(
    record: AudioManifestRecord,
    *,
    root: Path | None,
    manifest_dir: Path | None,
    expected_duration: float | None,
    duration_tolerance_sec: float,
    require_inspectable: bool,
) -> AudioAuditCheck:
    resolved_path = _resolve_audio(record.audio, root=root, manifest_dir=manifest_dir)
    issues: list[str] = []
    if not resolved_path.exists():
        return AudioAuditCheck(
            record_id=record.id,
            audio=record.audio,
            resolved_path=str(resolved_path),
            exists=False,
            inspectable=False,
            ok=False,
            expected_sample_rate=record.sample_rate,
            actual_sample_rate=None,
            expected_duration_sec=expected_duration,
            actual_duration_sec=None,
            issues=["missing_file"],
        )

    if resolved_path.suffix.lower() != ".wav":
        if require_inspectable:
            issues.append("uninspectable_audio_suffix")
        return AudioAuditCheck(
            record_id=record.id,
            audio=record.audio,
            resolved_path=str(resolved_path),
            exists=True,
            inspectable=False,
            ok=not issues,
            expected_sample_rate=record.sample_rate,
            actual_sample_rate=None,
            expected_duration_sec=expected_duration,
            actual_duration_sec=None,
            issues=issues,
        )

    try:
        info = inspect_wav(resolved_path)
    except (OSError, ValueError, EOFError) as exc:
        return AudioAuditCheck(
            record_id=record.id,
            audio=record.audio,
            resolved_path=str(resolved_path),
            exists=True,
            inspectable=False,
            ok=False,
            expected_sample_rate=record.sample_rate,
            actual_sample_rate=None,
            expected_duration_sec=expected_duration,
            actual_duration_sec=None,
            issues=[f"inspect_failed:{exc}"],
        )

    if info.sample_rate != record.sample_rate:
        issues.append("sample_rate_mismatch")
    if expected_duration is not None and abs(info.duration_sec - expected_duration) > duration_tolerance_sec:
        issues.append("duration_mismatch")

    return AudioAuditCheck(
        record_id=record.id,
        audio=record.audio,
        resolved_path=str(resolved_path),
        exists=True,
        inspectable=True,
        ok=not issues,
        expected_sample_rate=record.sample_rate,
        actual_sample_rate=info.sample_rate,
        expected_duration_sec=expected_duration,
        actual_duration_sec=round(info.duration_sec, 6),
        issues=issues,
    )


def _resolve_audio(audio: str, *, root: Path | None, manifest_dir: Path | None) -> Path:
    path = Path(audio)
    if path.is_absolute():
        return path
    if root is not None:
        return root / path
    if path.exists():
        return path
    if manifest_dir is not None:
        candidate = manifest_dir / path
        if candidate.exists():
            return candidate
    return path


def _expected_duration(record: AudioManifestRecord, *, expected_duration_getter: object | None) -> float | None:
    if callable(expected_duration_getter):
        value = expected_duration_getter(record)
    elif hasattr(record, "duration"):
        value = getattr(record, "duration")
    else:
        value = None
    if value is None:
        return None
    return float(value)
