#!/usr/bin/env python3
"""LDC corpus adapters for Stable-ASR.

Converts Fisher English (LDC2004S13/LDC2005S13), Switchboard-1 (LDC97S62),
and CallHome (LDC97S42) to Stable-ASR ASR manifests and turn manifests.

Usage:
    # After extracting LDC archives to /data/ldc/:
    python3 scripts/prepare_ldc.py fisher   --src /data/ldc/fisher   --out data/dialogue/fisher
    python3 scripts/prepare_ldc.py swbd     --src /data/ldc/swbd     --out data/dialogue/swbd
    python3 scripts/prepare_ldc.py callhome --src /data/ldc/callhome --out data/dialogue/callhome

    # Then bootstrap turn manifests:
    stable-asr bootstrap-turn-data --input data/dialogue/fisher/fisher_train.jsonl \
        --output-dir data/dialogue/fisher/turns --include-incomplete

Output per corpus:
    {out}/{corpus}_train.jsonl   ASR manifest (train split)
    {out}/{corpus}_dev.jsonl     ASR manifest (dev/validation split)
    {out}/{corpus}_test.jsonl    ASR manifest (test split, if available)
    {out}/audio/                 symlinks to original .sph files (or converted .flac)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                    encoding="utf-8")
    print(f"  Wrote {len(records):,} records → {path}")


def sph_to_flac(sph_path: Path, out_dir: Path, channel: int = 0) -> Path | None:
    """Convert one channel of a .sph (NIST Sphere) file to 16kHz mono flac.

    Returns the output path, or None if conversion fails.
    channel: 0=A (first speaker), 1=B (second speaker)
    """
    stem = sph_path.stem
    flac_path = out_dir / f"{stem}_ch{channel}.flac"
    if flac_path.exists():
        return flac_path
    try:
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error",
            "-i", str(sph_path),
            "-map_channel", f"0.0.{channel}",
            "-ar", "8000",   # telephony: keep native 8kHz
            "-c:a", "flac",
            str(flac_path),
        ], check=True, capture_output=True)
        return flac_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def strip_transcript_noise(text: str) -> str:
    """Remove LDC annotation markers from transcript text."""
    # [laughter], [noise], [vocalized-noise], {B_TRANS}, {E_TRANS}, <b_aside>, etc.
    text = re.sub(r'\[[\w\-]+\]', '', text)
    text = re.sub(r'\{[^}]+\}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Partial words: word- or -word
    text = re.sub(r'\b\w+-\s', ' ', text)
    text = re.sub(r'\s-\w+\b', ' ', text)
    # Multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Fisher English (LDC2004S13, LDC2005S13)
# ---------------------------------------------------------------------------
# Directory layout after extraction:
#   fisher_eng_tr_sp_LDC2004S13/
#     fe_03_p1_sph1/   .sph audio (2-channel, 8kHz, µ-law)
#     fe_03_p1_tran1/  .txt transcripts
#       data/trans/  fe_03_NNNNN.txt  (one per call)
#
# Transcript format (per call):
#   #  comment lines
#   <time> <speaker:A|B> <text>
#   e.g.: 0.97 A: hi um   what do you think about

FISHER_TRANS_LINE = re.compile(
    r'^(\d+\.\d+)\s+([AB]):\s*(.*)')


def parse_fisher_trans(trans_path: Path) -> list[dict]:
    """Parse a Fisher .txt transcript into utterance-level dicts."""
    segments = []
    lines = trans_path.read_text(errors="replace").splitlines()
    call_id = trans_path.stem  # e.g. fe_03_00001

    pending: dict[str, dict] = {}  # speaker → current open segment

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = FISHER_TRANS_LINE.match(line)
        if not m:
            continue
        t, spk, text = float(m.group(1)), m.group(2), m.group(3)
        text = strip_transcript_noise(text)

        # Close previous segment for this speaker at this timestamp
        if spk in pending:
            seg = pending.pop(spk)
            seg["end"] = round(t, 6)
            if seg["end"] > seg["start"] + 0.1 and seg["text"]:
                segments.append(seg)

        if text:
            pending[spk] = {
                "id": f"{call_id}__{spk}_{len(segments):06d}",
                "call_id": call_id,
                "speaker": spk,
                "start": round(t, 6),
                "end": None,
                "text": text,
            }

    # Close any still-open segments at end-of-file (use last timestamp + 2s)
    last_t = max((s["end"] for s in segments if s["end"]), default=0.0)
    for spk, seg in pending.items():
        seg["end"] = round(last_t + 2.0, 6)
        if seg["text"]:
            segments.append(seg)

    return segments


def prepare_fisher(src: Path, out: Path, convert_audio: bool = False) -> None:
    """Build Fisher ASR manifest from extracted LDC2004S13/LDC2005S13 tree."""
    print(f"\nFisher: scanning {src}")

    # Find all transcript files
    trans_files = sorted(src.rglob("fe_03_*.txt"))
    if not trans_files:
        # Try alternate layout
        trans_files = sorted(src.rglob("*.txt"))
    print(f"  Found {len(trans_files):,} transcript files")

    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    for trans_path in trans_files:
        call_id = trans_path.stem
        segments = parse_fisher_trans(trans_path)

        # Find matching .sph (audio may be in a separate subtree)
        sph_candidates = list(src.rglob(f"{call_id}.sph"))
        sph_path = sph_candidates[0] if sph_candidates else None

        for seg in segments:
            spk_chan = 0 if seg["speaker"] == "A" else 1

            if sph_path and convert_audio:
                flac = sph_to_flac(sph_path, audio_dir, channel=spk_chan)
                audio_field = str(flac) if flac else str(sph_path)
            elif sph_path:
                # Keep .sph reference; caller can convert later
                audio_field = str(sph_path)
            else:
                audio_field = ""

            all_records.append({
                "id": seg["id"],
                "text": seg["text"],
                "audio": audio_field,
                "sample_rate": 8000,
                "start": seg["start"],
                "end": seg["end"],
                "duration": round(seg["end"] - seg["start"], 6),
                "language": "en",
                "source": "fisher",
                "metadata": {
                    "speaker_id": f"{call_id}_{seg['speaker']}",
                    "call_id": call_id,
                    "channel": spk_chan,
                    "corpus": "LDC2004S13",
                },
            })

    # Split: use call_id for speaker-independent split
    call_ids = sorted({r["metadata"]["call_id"] for r in all_records})
    n = len(call_ids)
    train_calls = set(call_ids[:int(n * 0.90)])
    dev_calls   = set(call_ids[int(n * 0.90):int(n * 0.95)])
    test_calls  = set(call_ids[int(n * 0.95):])

    train = [r for r in all_records if r["metadata"]["call_id"] in train_calls]
    dev   = [r for r in all_records if r["metadata"]["call_id"] in dev_calls]
    test  = [r for r in all_records if r["metadata"]["call_id"] in test_calls]

    write_jsonl(out / "fisher_train.jsonl", train)
    write_jsonl(out / "fisher_dev.jsonl",   dev)
    write_jsonl(out / "fisher_test.jsonl",  test)
    print(f"  Total: {len(all_records):,} segments from {n:,} calls")


# ---------------------------------------------------------------------------
# Switchboard-1 (LDC97S62)
# ---------------------------------------------------------------------------
# Directory layout after extraction:
#   swb1_release2/
#     audio/  sw0NNNN.sph  (2-channel, 8kHz)
#     transcriptions/   (word-level, ms98 style)
#       word/  sw0NNNN-ms98-a-word.text
#              sw0NNNN-ms98-b-word.text
#
# Transcript format (ms98 word-level):
#   sw0NNNN-ms98-a-word.text
#   Lines: <utt_id> <start> <end> <word>
#   Grouped by utterance: sw0NNNN-A-NNNNNN-NNNNNN
#   or sentence-level in:  sw0NNNN-ms98-a-trans.text
#   Lines: <utt_id> <text>

SWBD_WORD_LINE = re.compile(r'^(\S+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(.+)')
SWBD_UTT_ID    = re.compile(r'^(sw\d+)-([AB])-(\d+\.\d+)-(\d+\.\d+)')


def parse_swbd_trans(trans_dir: Path) -> list[dict]:
    """Parse Switchboard ms98 sentence-level transcripts."""
    segments = []

    # Prefer sentence-level -trans.text files if available
    trans_files = sorted(trans_dir.rglob("*-trans.text"))
    if not trans_files:
        # Fall back to word-level, group by utterance
        return _parse_swbd_word_level(trans_dir)

    for f in trans_files:
        # Determine call_id and speaker from filename
        # e.g. sw0NNNN-ms98-a-trans.text
        m = re.search(r'(sw\d+)-ms98-([ab])-trans', f.stem, re.I)
        if not m:
            continue
        call_id, spk = m.group(1), m.group(2).upper()

        for line in f.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Format: <utt_id> <text>  where utt_id = sw0NNNN-A-start-end
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            utt_id, text = parts[0], parts[1]
            text = strip_transcript_noise(text)
            if not text:
                continue

            uid_m = SWBD_UTT_ID.match(utt_id)
            if uid_m:
                start = float(uid_m.group(3))
                end   = float(uid_m.group(4))
            else:
                continue

            segments.append({
                "call_id": call_id,
                "speaker": spk,
                "start": round(start, 6),
                "end":   round(end,   6),
                "text":  text,
            })

    return segments


def _parse_swbd_word_level(trans_dir: Path) -> list[dict]:
    """Group word-level Switchboard annotations into utterances."""
    segments = []
    for f in sorted(trans_dir.rglob("*-word.text")):
        m = re.search(r'(sw\d+)-ms98-([ab])-word', f.stem, re.I)
        if not m:
            continue
        call_id, spk = m.group(1), m.group(2).upper()

        current_utt: str | None = None
        words: list[tuple[float, float, str]] = []

        for line in f.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            wm = SWBD_WORD_LINE.match(line)
            if not wm:
                continue
            utt_id = wm.group(1)
            t0, t1, word = float(wm.group(2)), float(wm.group(3)), wm.group(4)

            if utt_id != current_utt:
                if current_utt and words:
                    text = strip_transcript_noise(" ".join(w for _, _, w in words))
                    if text:
                        segments.append({
                            "call_id": call_id, "speaker": spk,
                            "start": round(words[0][0], 6),
                            "end":   round(words[-1][1], 6),
                            "text":  text,
                        })
                current_utt = utt_id
                words = []

            if word not in ("[silence]", "[noise]", "B_TRANS", "E_TRANS"):
                words.append((t0, t1, word))

        # flush last
        if words:
            text = strip_transcript_noise(" ".join(w for _, _, w in words))
            if text:
                segments.append({
                    "call_id": call_id, "speaker": spk,
                    "start": round(words[0][0], 6),
                    "end":   round(words[-1][1], 6),
                    "text":  text,
                })

    return segments


def prepare_swbd(src: Path, out: Path, convert_audio: bool = False) -> None:
    """Build Switchboard ASR manifest."""
    print(f"\nSwitchboard: scanning {src}")

    trans_dir = src / "transcriptions"
    if not trans_dir.exists():
        trans_dir = src  # try root

    segments = parse_swbd_trans(trans_dir)
    print(f"  Parsed {len(segments):,} utterances")

    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for i, seg in enumerate(segments):
        call_id = seg["call_id"]
        spk     = seg["speaker"]
        spk_chan = 0 if spk == "A" else 1

        sph_candidates = list(src.rglob(f"{call_id}.sph"))
        sph_path = sph_candidates[0] if sph_candidates else None

        if sph_path and convert_audio:
            flac = sph_to_flac(sph_path, audio_dir, channel=spk_chan)
            audio_field = str(flac) if flac else str(sph_path)
        elif sph_path:
            audio_field = str(sph_path)
        else:
            audio_field = ""

        all_records.append({
            "id": f"{call_id}__{spk}_{i:06d}",
            "text": seg["text"],
            "audio": audio_field,
            "sample_rate": 8000,
            "start": seg["start"],
            "end":   seg["end"],
            "duration": round(seg["end"] - seg["start"], 6),
            "language": "en",
            "source": "switchboard",
            "metadata": {
                "speaker_id": f"{call_id}_{spk}",
                "call_id": call_id,
                "channel": spk_chan,
                "corpus": "LDC97S62",
            },
        })

    # Speaker-independent split by call_id
    call_ids = sorted({r["metadata"]["call_id"] for r in all_records})
    n = len(call_ids)
    train_calls = set(call_ids[:int(n * 0.90)])
    dev_calls   = set(call_ids[int(n * 0.90):int(n * 0.95)])
    test_calls  = set(call_ids[int(n * 0.95):])

    write_jsonl(out / "swbd_train.jsonl", [r for r in all_records if r["metadata"]["call_id"] in train_calls])
    write_jsonl(out / "swbd_dev.jsonl",   [r for r in all_records if r["metadata"]["call_id"] in dev_calls])
    write_jsonl(out / "swbd_test.jsonl",  [r for r in all_records if r["metadata"]["call_id"] in test_calls])
    print(f"  Total: {len(all_records):,} segments from {n:,} calls")


# ---------------------------------------------------------------------------
# CallHome (LDC97S42)
# ---------------------------------------------------------------------------
# Directory layout:
#   callhome_english/
#     audio/  en_NNNN.sph  (2-channel, 8kHz)
#     transcripts/ en_NNNN.cha  (CHAT format) or en_NNNN.stm
#
# CHAT format (.cha):
#   @Begin ... @End
#   *SPK: text <start_ms>_<end_ms>
#   e.g.  *A: hello how are you <0_1234>

CALLHOME_CHAT_UTT  = re.compile(r'^\*(\w+):\s*(.*)')
CALLHOME_CHAT_TIME = re.compile(r'<(\d+)_(\d+)>\s*$')
CALLHOME_STM_LINE  = re.compile(r'^(\S+)\s+(\S+)\s+(\S+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*(.*)')


def parse_callhome_cha(cha_path: Path) -> list[dict]:
    """Parse CHAT-format CallHome transcript."""
    segments = []
    call_id = cha_path.stem
    for line in cha_path.read_text(errors="replace").splitlines():
        line = line.strip()
        m = CALLHOME_CHAT_UTT.match(line)
        if not m:
            continue
        spk, rest = m.group(1), m.group(2)
        tm = CALLHOME_CHAT_TIME.search(rest)
        if not tm:
            continue
        start_ms, end_ms = int(tm.group(1)), int(tm.group(2))
        text = CALLHOME_CHAT_TIME.sub('', rest).strip()
        text = strip_transcript_noise(text)
        if not text:
            continue
        segments.append({
            "call_id": call_id, "speaker": spk,
            "start": round(start_ms / 1000, 6),
            "end":   round(end_ms   / 1000, 6),
            "text":  text,
        })
    return segments


def parse_callhome_stm(stm_path: Path) -> list[dict]:
    """Parse STM-format CallHome transcript."""
    segments = []
    for line in stm_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(';;'):
            continue
        m = CALLHOME_STM_LINE.match(line)
        if not m:
            continue
        fname, chan, spk = m.group(1), m.group(2), m.group(3)
        start, end = float(m.group(4)), float(m.group(5))
        text = strip_transcript_noise(m.group(6))
        if not text or text == "ignore_time_segment_in_scoring":
            continue
        segments.append({
            "call_id": fname, "speaker": spk,
            "start": round(start, 6), "end": round(end, 6),
            "text": text,
        })
    return segments


def prepare_callhome(src: Path, out: Path, lang: str = "en",
                     convert_audio: bool = False) -> None:
    """Build CallHome ASR manifest."""
    print(f"\nCallHome ({lang}): scanning {src}")

    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    # Try .cha first, then .stm
    trans_files = sorted(src.rglob("*.cha")) or sorted(src.rglob("*.stm"))
    print(f"  Found {len(trans_files):,} transcript files")

    for tf in trans_files:
        segs = (parse_callhome_cha(tf) if tf.suffix == ".cha"
                else parse_callhome_stm(tf))

        for i, seg in enumerate(segs):
            call_id = seg["call_id"]
            spk     = seg["speaker"]
            spk_chan = 1 if spk in ("B", "b", "1") else 0

            sph_candidates = (list(src.rglob(f"{call_id}.sph")) or
                               list(src.rglob(f"{call_id}.wav")))
            sph_path = sph_candidates[0] if sph_candidates else None

            if sph_path and convert_audio:
                flac = sph_to_flac(sph_path, audio_dir, channel=spk_chan)
                audio_field = str(flac) if flac else str(sph_path)
            elif sph_path:
                audio_field = str(sph_path)
            else:
                audio_field = ""

            all_records.append({
                "id": f"{call_id}__{spk}_{i:06d}",
                "text": seg["text"],
                "audio": audio_field,
                "sample_rate": 8000,
                "start": seg["start"],
                "end":   seg["end"],
                "duration": round(seg["end"] - seg["start"], 6),
                "language": lang,
                "source": "callhome",
                "metadata": {
                    "speaker_id": f"{call_id}_{spk}",
                    "call_id": call_id,
                    "channel": spk_chan,
                    "corpus": "LDC97S42",
                },
            })

    call_ids = sorted({r["metadata"]["call_id"] for r in all_records})
    n = len(call_ids)
    train_calls = set(call_ids[:int(n * 0.80)])
    dev_calls   = set(call_ids[int(n * 0.80):int(n * 0.90)])
    test_calls  = set(call_ids[int(n * 0.90):])

    prefix = f"callhome_{lang}"
    write_jsonl(out / f"{prefix}_train.jsonl", [r for r in all_records if r["metadata"]["call_id"] in train_calls])
    write_jsonl(out / f"{prefix}_dev.jsonl",   [r for r in all_records if r["metadata"]["call_id"] in dev_calls])
    write_jsonl(out / f"{prefix}_test.jsonl",  [r for r in all_records if r["metadata"]["call_id"] in test_calls])
    print(f"  Total: {len(all_records):,} segments from {n:,} calls")


# ---------------------------------------------------------------------------
# Smoke test with partial data
# ---------------------------------------------------------------------------

def smoke_test(src: Path, corpus: str) -> None:
    """Quick sanity check on a partial/incomplete corpus download."""
    print(f"\nSmoke test: {corpus} @ {src}")
    if corpus == "fisher":
        trans_files = list(src.rglob("fe_03_*.txt"))[:5]
        for f in trans_files:
            segs = parse_fisher_trans(f)
            print(f"  {f.name}: {len(segs)} segments, "
                  f"first='{segs[0]['text'][:50] if segs else '(empty)'}'")
    elif corpus == "swbd":
        trans_files = list(src.rglob("*-trans.text"))[:5]
        for f in trans_files:
            segs = parse_swbd_trans(f.parent)
            print(f"  {f.name}: parsed (word-level fallback)")
    elif corpus == "callhome":
        for f in list(src.rglob("*.cha"))[:3] + list(src.rglob("*.stm"))[:3]:
            segs = (parse_callhome_cha(f) if f.suffix == ".cha"
                    else parse_callhome_stm(f))
            print(f"  {f.name}: {len(segs)} segments")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare LDC corpora for Stable-ASR")
    parser.add_argument("corpus", choices=["fisher", "swbd", "callhome", "smoke"],
                        help="Corpus to prepare")
    parser.add_argument("--src", required=True, type=Path,
                        help="Root of extracted LDC archive")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for JSONL manifests")
    parser.add_argument("--lang", default="en",
                        help="Language code for CallHome (default: en)")
    parser.add_argument("--convert-audio", action="store_true",
                        help="Convert .sph to .flac via ffmpeg (slow, needs ffmpeg)")
    args = parser.parse_args()

    if args.corpus == "fisher":
        prepare_fisher(args.src, args.out, args.convert_audio)
    elif args.corpus == "swbd":
        prepare_swbd(args.src, args.out, args.convert_audio)
    elif args.corpus == "callhome":
        prepare_callhome(args.src, args.out, args.lang, args.convert_audio)
    elif args.corpus == "smoke":
        smoke_test(args.src, args.lang)  # lang re-used as corpus hint


if __name__ == "__main__":
    main()
