"""Public ASR corpus recipes for canonical Stable-ASR manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from stable_asr.data.formats.jsonl import iter_jsonl
from stable_asr.data.asr_manifest import ASRManifestRecord, write_asr_manifest

PUBLIC_ASR_CORPORA = ("librispeech", "aishell1", "common_voice", "wenetspeech")
COMMON_VOICE_DEFAULT_SPLITS = ("train", "dev", "test", "validated", "other")
COMMON_VOICE_SPLIT_FILES = {f"{split}.tsv" for split in (*COMMON_VOICE_DEFAULT_SPLITS, "invalidated", "reported")}


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
    elif corpus == "aishell1":
        records = _prepare_aishell1(root, split=split, sample_rate=sample_rate)
    elif corpus == "common_voice":
        records = _prepare_common_voice(root, split=split, sample_rate=sample_rate)
    else:
        records = _prepare_wenetspeech(root, split=split, sample_rate=sample_rate)

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


def _prepare_common_voice(root: Path, *, split: str | None, sample_rate: int) -> list[ASRManifestRecord]:
    tsv_root = _find_common_voice_tsv_root(root)
    clips_root = _find_common_voice_clips_root(tsv_root)
    tsv_paths = _common_voice_tsv_paths(tsv_root, split=split)
    default_language = _infer_common_voice_language(tsv_root)

    records: list[ASRManifestRecord] = []
    seen_ids: set[str] = set()
    for tsv_path in tsv_paths:
        inferred_split = tsv_path.stem
        with tsv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = set(reader.fieldnames or [])
            if "path" not in fieldnames or "sentence" not in fieldnames:
                raise ValueError(f"{tsv_path} is not a Common Voice clip TSV; expected path and sentence columns")
            for row_index, row in enumerate(reader):
                audio_rel = str(row.get("path") or "").strip()
                text = str(row.get("sentence") or "").strip()
                if not audio_rel or not text:
                    continue
                audio_path = _common_voice_audio_path(tsv_root, clips_root, audio_rel)
                language = str(row.get("locale") or row.get("language") or default_language).strip() or "unknown"
                record_id = _unique_common_voice_id(Path(audio_rel).stem, inferred_split, seen_ids)
                records.append(
                    ASRManifestRecord.from_dict(
                        {
                            "id": record_id,
                            "audio": audio_path.as_posix(),
                            "sample_rate": sample_rate,
                            "text": text,
                            "language": language,
                            "source": "common_voice",
                            "duration": _optional_float(row.get("duration") or row.get("duration_sec")),
                            "split": inferred_split,
                            "speaker_id": _optional_row_str(row, "client_id"),
                            "metadata": _common_voice_metadata(row, tsv_path=tsv_path, row_index=row_index),
                        }
                    )
                )
    return records


def _prepare_wenetspeech(root: Path, *, split: str | None, sample_rate: int) -> list[ASRManifestRecord]:
    metadata_path = _find_wenetspeech_metadata(root)
    audio_base = root if root.is_dir() else root.parent
    if metadata_path.suffix.lower() == ".jsonl":
        rows = [row for _, row in iter_jsonl(metadata_path)]
        records = _prepare_wenetspeech_flat_rows(
            rows,
            metadata_path=metadata_path,
            audio_base=audio_base,
            split=split,
            sample_rate=sample_rate,
        )
    else:
        records = _prepare_wenetspeech_audios(
            _iter_wenetspeech_audios(metadata_path),
            metadata_path=metadata_path,
            audio_base=audio_base,
            split=split,
            sample_rate=sample_rate,
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


def _find_common_voice_tsv_root(root: Path) -> Path:
    if any((root / f"{split}.tsv").exists() for split in COMMON_VOICE_DEFAULT_SPLITS):
        return root
    candidates = sorted({path.parent for path in root.rglob("*.tsv") if path.name in COMMON_VOICE_SPLIT_FILES})
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"found multiple Common Voice TSV directories under {root}; pass one locale directory as input")
    raise ValueError(f"could not find Common Voice split TSV files under {root}")


def _find_common_voice_clips_root(tsv_root: Path) -> Path:
    candidate = tsv_root / "clips"
    if candidate.exists():
        return candidate
    matches = sorted(path for path in tsv_root.rglob("clips") if path.is_dir())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"found multiple Common Voice clips directories under {tsv_root}")
    return candidate


def _common_voice_tsv_paths(tsv_root: Path, *, split: str | None) -> list[Path]:
    if split:
        split_name = split[:-4] if split.endswith(".tsv") else split
        path = tsv_root / f"{split_name}.tsv"
        if not path.exists():
            raise ValueError(f"Common Voice split TSV does not exist: {path}")
        return [path]

    paths = [tsv_root / f"{split_name}.tsv" for split_name in COMMON_VOICE_DEFAULT_SPLITS]
    paths = [path for path in paths if path.exists()]
    if paths:
        return paths
    ignored = {
        "clip_durations.tsv",
        "invalidated.tsv",
        "reported.tsv",
        "unvalidated_sentences.tsv",
        "validated_sentences.tsv",
    }
    return [path for path in sorted(tsv_root.glob("*.tsv")) if path.name not in ignored]


def _infer_common_voice_language(tsv_root: Path) -> str:
    name = tsv_root.name
    if name and not name.startswith("cv-corpus") and name not in {"common_voice", "CommonVoice"}:
        return name
    return "unknown"


def _common_voice_audio_path(tsv_root: Path, clips_root: Path, audio_rel: str) -> Path:
    path = Path(audio_rel)
    if path.is_absolute():
        return path
    direct = tsv_root / path
    if path.parts[:1] == ("clips",) or direct.exists():
        return direct
    return clips_root / path


def _unique_common_voice_id(base_id: str, split: str, seen_ids: set[str]) -> str:
    candidate = base_id or f"{split}_row"
    if candidate not in seen_ids:
        seen_ids.add(candidate)
        return candidate
    prefix = f"{split}_{candidate}"
    candidate = prefix
    suffix = 2
    while candidate in seen_ids:
        candidate = f"{prefix}_{suffix}"
        suffix += 1
    seen_ids.add(candidate)
    return candidate


def _common_voice_metadata(row: dict[str, Any], *, tsv_path: Path, row_index: int) -> dict[str, Any]:
    keys = (
        "sentence_id",
        "up_votes",
        "down_votes",
        "age",
        "gender",
        "accent",
        "accents",
        "variant",
        "locale",
        "segment",
    )
    metadata = {
        key: row[key]
        for key in keys
        if key in row and row[key] is not None and str(row[key]).strip() != ""
    }
    metadata.update(
        {
            "corpus_recipe": "common_voice",
            "path": str(row.get("path") or ""),
            "tsv_path": tsv_path.as_posix(),
            "source_row_index": row_index,
        }
    )
    return metadata


def _optional_row_str(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"duration must be numeric when present, got {value!r}") from exc


def _find_wenetspeech_metadata(root: Path) -> Path:
    if root.is_file():
        return root
    candidates = [
        root / "WenetSpeech.json",
        root / "wenetspeech.json",
        root / "metadata" / "WenetSpeech.json",
        root / "metadata" / "wenetspeech.json",
        root / "WenetSpeech.jsonl",
        root / "wenetspeech.jsonl",
        root / "metadata" / "WenetSpeech.jsonl",
        root / "metadata" / "wenetspeech.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(path for path in root.rglob("WenetSpeech.json") if path.is_file())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"found multiple WenetSpeech metadata files under {root}; pass one metadata root")
    raise ValueError(f"could not find WenetSpeech.json or WenetSpeech.jsonl under {root}")


def _iter_wenetspeech_audios(metadata_path: Path) -> Iterable[dict[str, Any]]:
    try:
        import ijson  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        audios = payload.get("audios") if isinstance(payload, dict) else None
        if not isinstance(audios, list):
            raise ValueError(f"{metadata_path} must contain a top-level audios list")
        for audio in audios:
            if isinstance(audio, dict):
                yield audio
        return

    with metadata_path.open("rb") as handle:
        for audio in ijson.items(handle, "audios.item"):
            if isinstance(audio, dict):
                yield audio


def _prepare_wenetspeech_audios(
    audios: Iterable[dict[str, Any]],
    *,
    metadata_path: Path,
    audio_base: Path,
    split: str | None,
    sample_rate: int,
) -> list[ASRManifestRecord]:
    records: list[ASRManifestRecord] = []
    for audio_index, audio in enumerate(audios):
        audio_rel = _pick_wenetspeech_audio_path(audio)
        segments = audio.get("segments", [])
        if not isinstance(segments, list):
            raise ValueError(f"{metadata_path}: audio {audio_index} segments must be a list")
        for segment_index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            record = _wenetspeech_record_from_segment(
                audio,
                segment,
                audio_rel=audio_rel,
                metadata_path=metadata_path,
                audio_base=audio_base,
                split=split,
                sample_rate=sample_rate,
                audio_index=audio_index,
                segment_index=segment_index,
            )
            if record is not None:
                records.append(record)
    return records


def _prepare_wenetspeech_flat_rows(
    rows: list[dict[str, Any]],
    *,
    metadata_path: Path,
    audio_base: Path,
    split: str | None,
    sample_rate: int,
) -> list[ASRManifestRecord]:
    records: list[ASRManifestRecord] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        audio_payload = row.get("audio") if isinstance(row.get("audio"), dict) else {}
        audio_rel = _first_present(
            row.get("audio_path"),
            row.get("WavPath"),
            row.get("wav"),
            row.get("path"),
            audio_payload.get("path"),
        )
        segment = {
            "sid": row.get("utt_id") or row.get("sid") or row.get("id"),
            "begin_time": _first_present(row.get("begin_time"), audio_payload.get("start_time")),
            "end_time": _first_present(row.get("end_time"), audio_payload.get("end_time")),
            "text": row.get("text") or row.get("sentence"),
            "confidence": row.get("confidence"),
            "subsets": row.get("subsets") or row.get("split"),
        }
        audio = {
            "aid": row.get("aid"),
            "path": audio_rel,
            "source": row.get("source"),
            "category": row.get("category"),
        }
        record = _wenetspeech_record_from_segment(
            audio,
            segment,
            audio_rel=audio_rel,
            metadata_path=metadata_path,
            audio_base=audio_base,
            split=split,
            sample_rate=sample_rate,
            audio_index=row_index,
            segment_index=0,
        )
        if record is not None:
            records.append(record)
    return records


def _wenetspeech_record_from_segment(
    audio: dict[str, Any],
    segment: dict[str, Any],
    *,
    audio_rel: Any,
    metadata_path: Path,
    audio_base: Path,
    split: str | None,
    sample_rate: int,
    audio_index: int,
    segment_index: int,
) -> ASRManifestRecord | None:
    text = str(segment.get("text") or "").strip()
    if not audio_rel or not text:
        return None
    inferred_split = _infer_wenetspeech_split(audio, segment, audio_rel=audio_rel)
    if split and inferred_split != split:
        return None
    begin_time = _optional_float(segment.get("begin_time"))
    end_time = _optional_float(segment.get("end_time"))
    duration = _positive_duration(begin_time, end_time) or _optional_float(segment.get("duration"))
    sid = str(segment.get("sid") or "").strip()
    aid = _optional_row_str(audio, "aid")
    record_id = sid or f"{aid or 'wenetspeech'}_{segment_index:05d}"

    return ASRManifestRecord.from_dict(
        {
            "id": record_id,
            "audio": _normalize_wenetspeech_audio_path(audio_base, str(audio_rel)).as_posix(),
            "sample_rate": sample_rate,
            "text": text,
            "language": "zh",
            "source": "wenetspeech",
            "duration": duration,
            "split": inferred_split,
            "speaker_id": _optional_row_str(segment, "speaker_id") or _optional_row_str(audio, "speaker_id"),
            "metadata": _wenetspeech_metadata(
                audio,
                segment,
                metadata_path=metadata_path,
                audio_index=audio_index,
                segment_index=segment_index,
                begin_time=begin_time,
                end_time=end_time,
            ),
        }
    )


def _pick_wenetspeech_audio_path(audio: dict[str, Any]) -> Any:
    return audio.get("path") or audio.get("audio_path") or audio.get("wav") or audio.get("WavPath")


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _normalize_wenetspeech_audio_path(audio_base: Path, audio_rel: str) -> Path:
    path = Path(audio_rel)
    if path.is_absolute():
        return path
    return audio_base / path


def _infer_wenetspeech_split(audio: dict[str, Any], segment: dict[str, Any], *, audio_rel: Any) -> str | None:
    subset = segment.get("subsets") or segment.get("subset") or audio.get("subsets") or audio.get("subset")
    if isinstance(subset, list) and subset:
        return str(subset[0]).lower()
    if isinstance(subset, str) and subset:
        return subset.lower()
    path_parts = Path(str(audio_rel or "")).parts
    for index, part in enumerate(path_parts):
        if part == "audio" and index + 1 < len(path_parts):
            return path_parts[index + 1]
        if part in {"train", "dev", "test_net", "test_meeting", "test_meeting_16k"}:
            return part
    return None


def _positive_duration(begin_time: float | None, end_time: float | None) -> float | None:
    if begin_time is None or end_time is None:
        return None
    duration = round(end_time - begin_time, 6)
    return duration if duration > 0 else None


def _wenetspeech_metadata(
    audio: dict[str, Any],
    segment: dict[str, Any],
    *,
    metadata_path: Path,
    audio_index: int,
    segment_index: int,
    begin_time: float | None,
    end_time: float | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "corpus_recipe": "wenetspeech",
        "metadata_path": metadata_path.as_posix(),
        "audio_index": audio_index,
        "segment_index": segment_index,
    }
    for key in ("aid", "duration", "url", "source", "category"):
        if key in audio and audio[key] not in (None, ""):
            metadata[f"audio_{key}"] = audio[key]
    for key in ("confidence", "subsets"):
        if key in segment and segment[key] not in (None, ""):
            metadata[key] = segment[key]
    if begin_time is not None:
        metadata["segment_start_sec"] = begin_time
    if end_time is not None:
        metadata["segment_end_sec"] = end_time
    return metadata
