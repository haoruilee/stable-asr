#!/usr/bin/env python3
"""Pilot experiment skeleton — Week-4 Go/No-Go gate proxy.

This script exercises the v1 paper's evaluation protocol end-to-end on
**AMI dev** (a stand-in for LDC corpora that have not yet arrived). The
goal is to surface findings-shape risks early — see ROADMAP.md
§"Paper Direction (locked 2026-06-03) → Findings risk and the Week-4
Go/No-Go Gate".

What it does:

  1. Load N records from a turn manifest (default: AMI dev bootstrap).
  2. For each requested factor (F3 speech_rate / F4 SNR / F5 overlap /
     F2' channel_simulate), apply each level to each record, producing
     a fan-out of ScenarioRecords with perturbed audio on disk.
  3. Run the configured ASR backend on every (record, factor, level)
     combination and the unperturbed baseline.
  4. Compute WER per (factor, level) and write a JSON results bundle +
     a Markdown summary.

Backends:

  --asr null            no ASR call; returns the ground-truth text. Used
                        for plumbing verification and timing the
                        perturbation step in isolation.
  --asr whisper         openai-whisper (`pip install -U openai-whisper`)
  --asr faster_whisper  CTranslate2 build (`pip install faster-whisper`)

LDC swap-in: when LDC manifests are ready, point ``--manifest`` at a
Switchboard / Fisher / CallHome turn manifest. Everything else stays.

Honest limitations of this proxy:

* AMI is read-style meeting speech; it will under-report
  conversational-stress factor effects relative to LDC Switchboard /
  Fisher. The pilot's job is to confirm the *shape* of findings, not
  to publish numbers on AMI.
* Turn-taking baselines are not run here — they need VAP installed
  (see scripts/run_vap_inference.py) and are a separate axis. The
  pilot is ASR-side only by design, to keep the gate small.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_asr.data.manifest import TurnManifestRecord, load_manifest
from stable_asr.eval.factors import (
    ChannelSimulateConfig,
    OverlapConfig,
    SNRConfig,
    SpeechRateConfig,
    apply_channel_simulate,
    apply_overlap,
    apply_snr,
    apply_speech_rate,
)
from stable_asr.eval.scenario_record import ScenarioRecord, write_scenario_jsonl


# ---------------------------------------------------------------------------
# Turn-baseline backends
# ---------------------------------------------------------------------------


def make_turn_baseline(name: str):
    """Construct a turn-taking baseline for the pilot.

    ``vap`` is the only audio-input baseline shipped today; it lazy-imports
    ``vap-turn-taking`` (``pip install vap-turn-taking``) and is the only
    one whose predictions actually change when audio is perturbed. The
    metadata-only baselines (rule_endpoint / vad_pause / text_turn) are
    useful for plumbing checks: their predictions stay identical across
    factor levels (they ignore audio), so a non-flat WER × turn_acc heatmap
    against them is itself a sanity signal that perturbation reached the
    audio path.
    """
    if name == "none":
        return None
    if name == "rule_endpoint":
        from stable_asr.models.baselines.rule_endpoint import RuleEndpointBaseline
        return RuleEndpointBaseline(complete_pause_ms=700)
    if name == "vad_pause":
        from stable_asr.models.baselines.vad_pause import VADPauseBaseline
        return VADPauseBaseline(complete_pause_ms=700)
    if name == "text_turn":
        from stable_asr.models.baselines.text_turn import TextTurnBaseline
        return TextTurnBaseline()
    if name == "vap":
        try:
            from stable_asr.models.baselines.vap import VAPPredictor
        except Exception as e:
            raise RuntimeError(
                "VAP baseline requires the vap-turn-taking package.\n"
                "Install with: pip install vap-turn-taking"
            ) from e
        return VAPPredictor()
    raise ValueError(f"unknown turn-baseline: {name!r}")


def _binary_turn_label(record: TurnManifestRecord) -> str | None:
    """Project the manifest's turn label to a binary {complete, incomplete}."""
    if record.turn_label in ("complete", "incomplete"):
        return record.turn_label
    return None


# ---------------------------------------------------------------------------
# WER (small, dependency-free)
# ---------------------------------------------------------------------------


def _normalize_text(s: str) -> list[str]:
    """Whisper-ish normalization: lowercase, strip punctuation, split on ws."""
    import re

    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    return [w for w in s.split() if w]


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = curr
    return prev[-1]


def wer(ref: str, hyp: str) -> tuple[float, int, int]:
    """Return (wer, ref_word_count, edits)."""
    rwords = _normalize_text(ref)
    hwords = _normalize_text(hyp)
    if not rwords:
        return 0.0 if not hwords else 1.0, 0, len(hwords)
    edits = _edit_distance(rwords, hwords)
    return edits / len(rwords), len(rwords), edits


# ---------------------------------------------------------------------------
# ASR backends
# ---------------------------------------------------------------------------


@dataclass
class ASRBackend:
    name: str

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        raise NotImplementedError


class NullBackend(ASRBackend):
    """Returns the ground-truth text. For plumbing verification only."""

    def __init__(self) -> None:
        super().__init__("null")
        self._gt: dict[str, str] = {}

    def set_groundtruth(self, mapping: dict[str, str]) -> None:
        self._gt = mapping

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        return self._gt.get(audio_path, "")


class WhisperBackend(ASRBackend):
    def __init__(self, model: str = "small", device: str | None = None):
        super().__init__(f"whisper:{model}")
        try:
            import whisper  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai-whisper not installed. Try: pip install -U openai-whisper"
            ) from e
        self._whisper = whisper.load_model(model, device=device)

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        result = self._whisper.transcribe(
            audio_path, language=language if language else None, fp16=False, verbose=False
        )
        return str(result.get("text", "")).strip()


class FasterWhisperBackend(ASRBackend):
    def __init__(self, model: str = "small", device: str = "auto", compute_type: str = "auto"):
        super().__init__(f"faster_whisper:{model}")
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper not installed. Try: pip install faster-whisper"
            ) from e
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        segs, _info = self._model.transcribe(audio_path, language=language or None)
        return " ".join(s.text for s in segs).strip()


def make_backend(name: str, *, model: str | None = None, device: str | None = None) -> ASRBackend:
    if name == "null":
        return NullBackend()
    if name == "whisper":
        return WhisperBackend(model=model or "small", device=device)
    if name == "faster_whisper":
        return FasterWhisperBackend(model=model or "small", device=device or "auto")
    raise ValueError(f"unknown backend {name!r}")


# ---------------------------------------------------------------------------
# Factor matrix
# ---------------------------------------------------------------------------


def _factor_levels(factor: str, output_root: Path, *, competitor_pool: Path | None) -> list[tuple[str, callable]]:
    """Return [(level_label, apply_fn(record) -> ScenarioRecord), ...]."""
    if factor == "speech_rate":
        return [
            (f"rate_{r:.2f}x", lambda rec, _r=r: apply_speech_rate(
                rec, SpeechRateConfig(rate=_r, output_dir=output_root / "speech_rate")))
            for r in (0.7, 1.0, 1.3)
        ]
    if factor == "snr":
        return [
            (f"snr_{db}", lambda rec, _db=db: apply_snr(
                rec, SNRConfig(snr_db=_db, output_dir=output_root / "snr")))
            for db in (20, 10, 0)
        ]
    if factor == "overlap":
        if competitor_pool is None:
            raise ValueError("overlap factor requires --competitor-pool")
        return [
            (f"overlap_{ov:.2f}", lambda rec, _ov=ov: apply_overlap(
                rec, OverlapConfig(
                    overlap_ratio=_ov, output_dir=output_root / "overlap",
                    competitor_pool=competitor_pool, n_overlap_windows=2)))
            for ov in (0.0, 0.15, 0.30)
        ]
    if factor == "channel_simulate":
        return [
            (f"channel_{kind}", lambda rec, _kind=kind: apply_channel_simulate(
                rec, ChannelSimulateConfig(kind=_kind, output_dir=output_root / "channel")))
            for kind in ("clean", "narrowband", "telephone", "cellular")
        ]
    raise ValueError(f"unknown factor {factor!r}")


# ---------------------------------------------------------------------------
# Pilot run
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--manifest", type=Path, required=True,
                   help="turn manifest JSONL (e.g. runs/bench_accel_ami/turn_data/turn_manifest.jsonl)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=200,
                   help="number of records to sample (default 200; set 0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--factors", nargs="+",
                   default=["speech_rate", "snr", "channel_simulate"],
                   choices=["speech_rate", "snr", "overlap", "channel_simulate"])
    p.add_argument("--asr", default="null", choices=["null", "whisper", "faster_whisper"])
    p.add_argument("--asr-model", default="small")
    p.add_argument("--asr-device", default=None)
    p.add_argument("--language", default="en")
    p.add_argument("--turn-baseline", default="none",
                   choices=["none", "rule_endpoint", "vad_pause", "text_turn", "vap"],
                   help="Optional turn-taking baseline to evaluate per scenario "
                        "(VAP is the only one whose predictions change with audio).")
    p.add_argument("--competitor-pool", type=Path, default=None,
                   help="dir of audio files for the F5 overlap competitor (required if --factors overlap)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = args.output_dir / "perturbed_audio"
    scenario_jsonl = args.output_dir / "scenario_records.jsonl"
    results_json = args.output_dir / "pilot_results.json"
    summary_md = args.output_dir / "pilot_summary.md"

    # ---- 1. sample records --------------------------------------------------
    print(f"[pilot] loading manifest {args.manifest}")
    records = load_manifest(args.manifest)
    if args.n > 0 and len(records) > args.n:
        import random
        random.Random(args.seed).shuffle(records)
        records = records[: args.n]
    print(f"[pilot] using {len(records)} records (seed={args.seed})")

    # Filter to records whose audio file actually exists.
    keep: list[TurnManifestRecord] = []
    for r in records:
        if Path(r.audio).exists():
            keep.append(r)
    if len(keep) < len(records):
        print(f"[pilot] dropped {len(records) - len(keep)} records with missing audio")
    records = keep
    if not records:
        print("[pilot] no usable records — abort.")
        return 2

    # ---- 2. apply factors ---------------------------------------------------
    fan_out: list[ScenarioRecord] = []
    t0 = time.time()
    for factor in args.factors:
        levels = _factor_levels(factor, audio_root, competitor_pool=args.competitor_pool)
        print(f"[pilot] factor={factor}  levels={[lab for lab, _ in levels]}")
        for level_label, apply_fn in levels:
            for rec in records:
                try:
                    out = apply_fn(rec)
                    fan_out.append(out)
                except Exception as e:
                    print(f"  ! {factor}/{level_label}/{rec.id} failed: {e}")
        elapsed = time.time() - t0
        print(f"[pilot]   factor={factor} cumulative={elapsed:.1f}s scenarios={len(fan_out)}")

    write_scenario_jsonl(scenario_jsonl, fan_out)
    print(f"[pilot] wrote {len(fan_out)} ScenarioRecords → {scenario_jsonl}")

    # ---- 3. ASR transcription ----------------------------------------------
    print(f"[pilot] ASR backend = {args.asr}")
    backend = make_backend(args.asr, model=args.asr_model, device=args.asr_device)
    if isinstance(backend, NullBackend):
        # Map every audio path back to the source record's reference text.
        backend.set_groundtruth({s.audio: (s.text or "") for s in fan_out})

    # ---- 3b. Turn baseline (optional) --------------------------------------
    print(f"[pilot] turn baseline = {args.turn_baseline}")
    turn_predictor = None
    if args.turn_baseline != "none":
        try:
            turn_predictor = make_turn_baseline(args.turn_baseline)
        except RuntimeError as e:
            print(f"[pilot] WARN: turn baseline unavailable: {e}", file=sys.stderr)
            turn_predictor = None

    rows: list[dict] = []
    t0 = time.time()
    for i, sc in enumerate(fan_out):
        ref = sc.text or ""
        try:
            hyp = backend.transcribe(sc.audio, language=sc.language or args.language)
        except Exception as e:
            hyp = ""
            err = str(e)
        else:
            err = None
        wer_value, n_ref, n_edits = wer(ref, hyp)

        # Turn-baseline prediction (optional)
        turn_pred_label: str | None = None
        turn_correct: bool | None = None
        turn_err: str | None = None
        if turn_predictor is not None:
            gold = _binary_turn_label(sc.to_turn_record())
            try:
                pred = turn_predictor.predict(sc.to_turn_record())
                turn_pred_label = pred.label
                if gold is not None and turn_pred_label in ("complete", "incomplete"):
                    turn_correct = (turn_pred_label == gold)
            except Exception as e:
                turn_err = str(e)

        rows.append({
            "id": sc.id,
            "base_id": sc.base_id,
            "factor": sc.factor,
            "factor_level": sc.factor_level,
            "wer": wer_value,
            "ref_words": n_ref,
            "edits": n_edits,
            "ref": ref,
            "hyp": hyp,
            "error": err,
            "turn_gold": _binary_turn_label(sc.to_turn_record()),
            "turn_pred": turn_pred_label,
            "turn_correct": turn_correct,
            "turn_error": turn_err,
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(fan_out):
            elapsed = time.time() - t0
            print(f"[pilot]   asr {i + 1}/{len(fan_out)}  {(i+1)/max(elapsed,1e-3):.1f} rec/s")

    # ---- 4. aggregate -------------------------------------------------------
    from statistics import mean

    by_factor: dict[tuple[str, str], list[float]] = {}
    by_factor_turn: dict[tuple[str, str], list[bool]] = {}
    for r in rows:
        key = (r["factor"], r["factor_level"])
        by_factor.setdefault(key, []).append(r["wer"])
        if r["turn_correct"] is not None:
            by_factor_turn.setdefault(key, []).append(bool(r["turn_correct"]))

    factor_summary = []
    for (factor, level), vals in sorted(by_factor.items()):
        turn_vals = by_factor_turn.get((factor, level), [])
        turn_acc: float | None = None
        if turn_vals:
            turn_acc = sum(1 for v in turn_vals if v) / len(turn_vals)
        factor_summary.append({
            "factor": factor,
            "level": level,
            "n": len(vals),
            "wer_mean": float(mean(vals)) if vals else float("nan"),
            "wer_min": float(min(vals)) if vals else float("nan"),
            "wer_max": float(max(vals)) if vals else float("nan"),
            "turn_n": len(turn_vals),
            "turn_accuracy": turn_acc,
        })

    bundle = {
        "manifest": str(args.manifest),
        "n_records": len(records),
        "n_scenarios": len(fan_out),
        "factors": list(args.factors),
        "asr_backend": backend.name,
        "asr_model": args.asr_model,
        "turn_baseline": args.turn_baseline,
        "factor_summary": factor_summary,
        "rows": rows,
    }
    results_json.write_text(json.dumps(bundle, ensure_ascii=False, indent=2))
    print(f"[pilot] wrote {results_json}")

    # ---- 5. markdown summary -----------------------------------------------
    lines = [
        "# Pilot run summary",
        "",
        f"- manifest: `{args.manifest}`",
        f"- records: {len(records)}",
        f"- scenarios (factor levels × records): {len(fan_out)}",
        f"- factors: {', '.join(args.factors)}",
        f"- ASR: {backend.name}",
        f"- turn baseline: {args.turn_baseline}",
        "",
        "## WER and turn accuracy by factor × level",
        "",
        "| Factor | Level | N | WER mean | WER min | WER max | Turn N | Turn acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in factor_summary:
        turn_acc_str = (
            f"{row['turn_accuracy']:.3f}" if row["turn_accuracy"] is not None else "—"
        )
        lines.append(
            f"| {row['factor']} | {row['level']} | {row['n']} | "
            f"{row['wer_mean']:.3f} | {row['wer_min']:.3f} | {row['wer_max']:.3f} | "
            f"{row['turn_n']} | {turn_acc_str} |"
        )
    lines.append("")
    if isinstance(backend, NullBackend):
        lines.append(
            "_NOTE_: backend = `null`; WER is 0 by construction. Re-run with "
            "`--asr whisper` or `--asr faster_whisper` to get real numbers."
        )
    summary_md.write_text("\n".join(lines))
    print(f"[pilot] wrote {summary_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
