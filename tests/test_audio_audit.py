from pathlib import Path

from stable_asr.data.asr_manifest import ASRManifestRecord, write_asr_manifest
from stable_asr.data.audio import synth_tone, write_wav_mono
from stable_asr.data.audio_audit import audit_audio_records
from stable_asr.scenarios.synthetic_turn import write_synthetic_turn_manifest


def test_audio_audit_accepts_synthetic_turn_wavs(tmp_path: Path) -> None:
    manifest = tmp_path / "turn.jsonl"
    records = write_synthetic_turn_manifest(manifest, episodes=3, seed=4, write_audio=True)

    report = audit_audio_records(records, kind="turn", manifest_path=manifest)

    assert report.ok
    assert report.records == 3
    assert report.checked_files == 3
    assert report.sample_rate_mismatches == 0
    assert report.duration_mismatches == 0


def test_audio_audit_reports_missing_audio(tmp_path: Path) -> None:
    manifest = tmp_path / "turn.jsonl"
    records = write_synthetic_turn_manifest(manifest, episodes=2, seed=4, write_audio=False)

    report = audit_audio_records(records, kind="turn", manifest_path=manifest)

    assert not report.ok
    assert report.missing_files == 2
    assert "missing_file" in report.checks[0].issues


def test_audio_audit_reports_asr_sample_rate_mismatch(tmp_path: Path) -> None:
    audio = tmp_path / "audio" / "utt.wav"
    samples = synth_tone(0.1, sample_rate=8000, seed=1)
    write_wav_mono(audio, samples, sample_rate=8000)
    records = [
        ASRManifestRecord.from_dict(
            {
                "id": "utt",
                "audio": "audio/utt.wav",
                "sample_rate": 16000,
                "text": "hello",
                "language": "en",
                "source": "unit",
                "duration": 0.1,
            }
        )
    ]
    write_asr_manifest(tmp_path / "asr.jsonl", records)

    report = audit_audio_records(records, kind="asr", manifest_path=tmp_path / "asr.jsonl")

    assert not report.ok
    assert report.sample_rate_mismatches == 1
    assert "sample_rate_mismatch" in report.checks[0].issues
