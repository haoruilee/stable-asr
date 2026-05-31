import hashlib
import json
from pathlib import Path

from stable_asr.paper.handoff import audit_final_handoff, final_handoff_template


def test_final_handoff_template_covers_input_collections() -> None:
    payload = final_handoff_template()

    assert payload["version"] == "stable_asr_final_handoff_v0"
    assert payload["entries"]
    assert any(entry["collection_id"] == "librispeech_dev_clean" for entry in payload["entries"])
    assert any(entry["checksums"] for entry in payload["entries"])


def test_final_handoff_audit_accepts_complete_handoff(tmp_path: Path) -> None:
    staged = tmp_path / "data.txt"
    staged.write_text("stable-asr\n", encoding="utf-8")
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    handoff = {
        "version": "stable_asr_final_handoff_v0",
        "entries": [
            {
                "collection_id": "unit_collection",
                "owner": "tester",
                "staged_paths": ["data.txt"],
                "source_urls": ["https://example.com/source"],
                "license_or_consent_notes": "local fixture with project permission",
                "commands_run": ["echo build"],
                "verification_outputs": ["pytest"],
                "checksums": [{"path": "data.txt", "sha256": digest, "bytes": staged.stat().st_size}],
                "known_gaps": [],
            }
        ],
    }
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(handoff), encoding="utf-8")

    report = audit_final_handoff(path, repo_root=tmp_path, require_checksums=True)

    assert report.ok
    assert report.entries == 1
    assert "data.txt" in report.checked_paths
    assert "final_handoff_audit: OK" in report.to_text()


def test_final_handoff_audit_can_require_checksums(tmp_path: Path) -> None:
    staged = tmp_path / "data.txt"
    staged.write_text("stable-asr\n", encoding="utf-8")
    handoff = {
        "version": "stable_asr_final_handoff_v0",
        "entries": [
            {
                "collection_id": "unit_collection",
                "owner": "tester",
                "staged_paths": ["data.txt"],
                "source_urls": ["https://example.com/source"],
                "license_or_consent_notes": "local fixture with project permission",
                "commands_run": ["echo build"],
                "verification_outputs": ["pytest"],
                "checksums": [],
                "known_gaps": [],
            }
        ],
    }
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(handoff), encoding="utf-8")

    loose = audit_final_handoff(path, repo_root=tmp_path)
    strict = audit_final_handoff(path, repo_root=tmp_path, require_checksums=True)

    assert loose.ok
    assert "unit_collection:checksums:missing" in loose.warnings
    assert not strict.ok
    assert "unit_collection:checksums:missing" in strict.errors


def test_final_handoff_audit_rejects_missing_metadata_and_paths(tmp_path: Path) -> None:
    handoff = {
        "version": "stable_asr_final_handoff_v0",
        "entries": [
            {
                "collection_id": "bad",
                "owner": "",
                "staged_paths": ["missing.wav"],
                "source_urls": [],
                "license_or_consent_notes": "",
                "commands_run": [],
                "verification_outputs": [],
                "checksums": [],
            }
        ],
    }
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(handoff), encoding="utf-8")

    report = audit_final_handoff(path, repo_root=tmp_path)

    assert not report.ok
    assert "bad:owner:missing" in report.errors
    assert "bad:staged_path_missing:missing.wav" in report.errors
    assert "bad:checksums:missing" in report.warnings
