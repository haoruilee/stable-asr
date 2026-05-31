from pathlib import Path

from stable_asr.paper.integrity import (
    artifact_integrity_manifest,
    verify_artifact_integrity,
    write_artifact_integrity,
)


def test_artifact_integrity_verifies_and_detects_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "nested" / "second.txt"
    first.write_text("stable-asr\n", encoding="utf-8")
    second.parent.mkdir()
    second.write_text("artifact\n", encoding="utf-8")

    report = artifact_integrity_manifest([first, second], root=tmp_path)
    outputs = write_artifact_integrity(report, tmp_path / "artifact_hashes.json", tmp_path / "ARTIFACT_HASHES.md")

    verified = verify_artifact_integrity(outputs["json"], root=tmp_path)
    assert verified.ok
    assert [digest.path for digest in verified.files] == ["first.txt", "nested/second.txt"]

    first.write_text("tampered\n", encoding="utf-8")
    tampered = verify_artifact_integrity(outputs["json"], root=tmp_path)

    assert not tampered.ok
    assert tampered.mismatched == ["first.txt"]


def test_artifact_integrity_accepts_relative_paths_that_include_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("artifacts")
    root.mkdir()
    artifact = root / "index.md"
    artifact.write_text("stable-asr\n", encoding="utf-8")

    report = artifact_integrity_manifest([artifact], root=root)

    assert report.ok
    assert [digest.path for digest in report.files] == ["index.md"]
