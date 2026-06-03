#!/usr/bin/env python3
"""Run real VAP inference on a Stable-ASR turn manifest.

Replaces the bootstrap placeholder predictions previously produced by
``stable_asr.data.bootstrap``. Outputs a raw VAP-style JSONL that
``scripts/export_turn_predictions.py --schema vap`` can normalize into
the canonical TurnPrediction format used by ``stable-asr compare-turn``.

The bootstrap predictions are recognisable by
``"source": "stable_asr_bootstrap_prediction"`` lines and ``probs`` that
match the manifest's own labels. Real VAP predictions will lack that
source tag and have continuous probabilities derived from the model.

Usage:
    pip install vap-turn-taking
    python scripts/run_vap_inference.py \\
        --manifest runs/final/turn_test.jsonl \\
        --output   runs/final/external/vap_raw.jsonl \\
        --checkpoint ErikEkstedt/VAP \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_asr.data.manifest import load_manifest
from stable_asr.models.baselines.vap import VAPPredictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--manifest", type=Path, required=True,
                        help="turn manifest JSONL")
    parser.add_argument("--output", type=Path, required=True,
                        help="raw VAP output JSONL (feed to export_turn_predictions.py --schema vap)")
    parser.add_argument("--checkpoint", default="ErikEkstedt/VAP",
                        help="HF model id or local path to a VAP checkpoint")
    parser.add_argument("--device", default=None,
                        help="torch device; defaults to cuda if available")
    parser.add_argument("--context-sec", type=float, default=10.0,
                        help="seconds of audio context fed to VAP per record")
    parser.add_argument("--limit", type=int, default=0,
                        help="if > 0, only run the first N records (for smoke testing)")
    parser.add_argument("--audio-root", type=Path, default=None,
                        help="root for relative audio paths in the manifest")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    records = load_manifest(args.manifest)
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[vap] loaded {len(records)} records from {args.manifest}")

    predictor = VAPPredictor(
        checkpoint=args.checkpoint,
        device=args.device,
        context_sec=args.context_sec,
    )
    predictor._load()
    print(f"[vap] model loaded: {args.checkpoint} on {predictor._device}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = 0
    n_err = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for i, record in enumerate(records):
            if args.audio_root and not Path(record.audio).is_absolute():
                from dataclasses import replace
                record = replace(record, audio=str(args.audio_root / record.audio))
            try:
                pred = predictor.predict(record)
                p_complete = pred.probs.get("complete", 0.5)
                row = {
                    "id": record.id,
                    "p_system_future": p_complete,
                    "p_user_future": 1.0 - p_complete,
                    "p_backchannel": pred.probs.get("backchannel", 0.0),
                    "p_wait": pred.probs.get("wait", 0.0),
                    "timestamp": pred.timestamp,
                    "source": "vap_real_inference",
                    "checkpoint": args.checkpoint,
                }
                n_ok += 1
            except Exception as e:
                row = {
                    "id": record.id,
                    "p_system_future": 0.5,
                    "p_user_future": 0.5,
                    "p_backchannel": 0.0,
                    "p_wait": 0.0,
                    "timestamp": float(record.end),
                    "source": "vap_real_inference",
                    "checkpoint": args.checkpoint,
                    "error": str(e),
                }
                n_err += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 100 == 0 or (i + 1) == len(records):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(records) - (i + 1)) / max(rate, 1e-6)
                print(f"[vap] {i + 1}/{len(records)}  ok={n_ok} err={n_err}  {rate:.1f} rec/s  eta={eta:.0f}s")

    print(f"[vap] wrote {len(records)} predictions to {args.output}")
    print(f"[vap] success: {n_ok}  errors: {n_err}")
    if n_err > 0:
        print("[vap] note: error rows have source=vap_real_inference + an `error` field; "
              "they are NOT bootstrap predictions, but the model could not run on those records.")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
