"""Public ASR corpus recipes for canonical Stable-ASR manifests."""

from __future__ import annotations

from pathlib import Path

from stable_asr.data.asr_manifest import ASRManifestRecord, write_asr_manifest

PUBLIC_ASR_CORPORA = ("librispeech", "aishell1")


def prepare_public_asr_manifest(
    *,
    corpus: str,
    input_dir: str | Path,
    output_path: str | Path,
    split: str | None = None,
    sample_rate: int = 16000,
) -> list[ASRManifestRecord]:
    """Prepare an ASR manifest from a supported public corpus directory.

    The recipe expects a locally downloaded/extracted corpus. It does not
    download data, mutate the corpus, or infer labels beyond utterance-level ASR
    metadata.
    """

    if corpus not in PUBLIC_ASR_CORPORA:
        raise ValueError(f"unknown public ASR corpus {corpus!r}; expected one of {PUBLIC_ASR_CORPORA}")
    root = Path(input_dir)
    if not root.exists():
        raise ValueError(f"input directory does not exist: {root}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    if corpus == "librispeech":
        records = _prepare_librispeech(root, split=split, sample_rate=sample_rate)
    else:
        records = _prepare_aishell1(root, split=split, sample_rate=sample_rate)

    if not records:
        detail = f" for split {split!r}" if split else ""
        raise ValueError(f"no {corpus} utterances found in {root}{detail}")
    write_asr_manifest(output_path, records)
    return records


def _prepare_librispeech(root: Path, *, split: str | None, sample_rate: int) -> list[ASRManifestRecord]:
    search_root = root / split if split and (root / split).exists() else root
    records: list[ASRManifestRecord] = []
    for transcript_path in sorted(search_root.rglob("*.trans.txt")):
        inferred_split = split or _infer_librispeech_split(root, transcript_path)
        with transcript_path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    utterance_id, text = line.split(maxsplit=1)
                except ValueError as exc:
                    raise ValueError(f"{transcript_path}:{line_index + 1}: expected '<utt_id> <text>'") from exc
                speaker_id, chapter_id = _librispeech_ids(utterance_id)
                audio_path = transcript_path.parent / f"{utterance_id}.flac"
                records.append(
                    ASRManifestRecord.from_dict(
                        {
                            "id": utterance_id,
                            "audio": audio_path.as_posix(),
                            "sample_rate": sample_rate,
                            "text": text,
                            "language": "en",
                            "source": "librispeech",
                            "split": inferred_split,
                            "speaker_id": speaker_id,
                            "metadata": {
                                "corpus_recipe": "librispeech",
                                "chapter_id": chapter_id,
                                "transcript_path": transcript_path.as_posix(),
                                "line_index": line_index,
                            },
                        }
                    )
                )
    return records


def _prepare_aishell1(root: Path, *, split: str | None, sample_rate: int) -> list[ASRManifestRecord]:
    transcript_path = _find_aishell_transcript(root)
    wav_root = _find_aishell_wav_root(root)
    transcripts = _read_aishell_transcripts(transcript_path)

    records: list[ASRManifestRecord] = []
    for wav_path in sorted(wav_root.rglob("*.wav")):
        relative = wav_path.relative_to(wav_root)
        inferred_split = relative.parts[0] if len(relative.parts) >= 3 else split
        if split and inferred_split != split:
            continue
        utterance_id = wav_path.stem
        text = transcripts.get(utterance_id)
        if not text:
            continue
        records.append(
            ASRManifestRecord.from_dict(
                {
                    "id": utterance_id,
                    "audio": wav_path.as_posix(),
                    "sample_rate": sample_rate,
                    "text": text,
                    "language": "zh",
                    "source": "aishell1",
                    "split": inferred_split,
                    "speaker_id": wav_path.parent.name,
                    "metadata": {
                        "corpus_recipe": "aishell1",
                        "transcript_path": transcript_path.as_posix(),
                    },
                }
            )
        )
    return records


def _infer_librispeech_split(root: Path, transcript_path: Path) -> str | None:
    try:
        relative = transcript_path.relative_to(root)
    except ValueError:
        return root.name
    if len(relative.parts) >= 4:
        return relative.parts[0]
    return root.name if root.name else None


def _librispeech_ids(utterance_id: str) -> tuple[str | None, str | None]:
    parts = utterance_id.split("-")
    speaker_id = parts[0] if len(parts) >= 1 and parts[0] else None
    chapter_id = parts[1] if len(parts) >= 2 and parts[1] else None
    return speaker_id, chapter_id


def _find_aishell_transcript(root: Path) -> Path:
    candidates = [
        root / "transcript" / "aishell_transcript_v0.8.txt",
        root / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob("aishell_transcript*.txt"))
    if matches:
        return matches[0]
    raise ValueError(f"could not find AISHELL transcript file under {root}")


def _find_aishell_wav_root(root: Path) -> Path:
    candidates = [root / "wav", root / "data_aishell" / "wav"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = [path for path in root.rglob("wav") if path.is_dir()]
    if matches:
        return sorted(matches)[0]
    raise ValueError(f"could not find AISHELL wav directory under {root}")


def _read_aishell_transcripts(path: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                utterance_id, text = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_index + 1}: expected '<utt_id> <text>'") from exc
            transcripts[utterance_id] = text
    return transcripts
