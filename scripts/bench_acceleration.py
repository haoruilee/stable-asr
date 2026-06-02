#!/usr/bin/env python3
"""Benchmark training and inference acceleration.

Two benchmark tracks:
  1. nano (metadata MLP) — lightweight model, shows batch inference speedup
  2. micro (audio_seq TCN) — compute-heavy model, shows AMP training speedup

Usage:
    python3 scripts/bench_acceleration.py [--output runs/bench_accel] [--epochs N]
    python3 scripts/bench_acceleration.py --inference-only   # skip training, just bench inference
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list:
    from stable_asr.data.manifest import TurnManifestRecord
    return [TurnManifestRecord.from_dict(json.loads(l))
            for l in Path(path).read_text().splitlines() if l.strip()]


def export_onnx(ckpt: Path) -> Path | None:
    """Export checkpoint to ONNX via CLI; return path or None on failure."""
    onnx_path = ckpt.parent / "model.onnx"
    if onnx_path.exists():
        return onnx_path
    result = subprocess.run(
        [sys.executable, "-m", "stable_asr.cli", "export-turn-onnx",
         "--checkpoint", str(ckpt), "--output", str(onnx_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and onnx_path.exists():
        return onnx_path
    print(f"    ONNX export failed: {result.stderr.strip()[:200]}")
    return None


def run_training(name: str, out: Path, train_records, dev_records, *,
                 model_type: str, feature_source: str, epochs: int,
                 lr: float = 1e-2, batch_size: int = 256, **kw) -> dict:
    from stable_asr.train.turn_trainer import train_nanoturn
    kw_full = dict(
        output_dir=str(out / name),
        model_type=model_type,
        epochs=epochs,
        lr=lr,
        seed=42,
        feature_source=feature_source,
        batch_size=batch_size,
        validation_split=0.0,
        optimizer="adam",
        checkpoint_interval=epochs,
        **kw,
    )
    t0 = time.perf_counter()
    result = train_nanoturn(train_records, val_records=dev_records, **kw_full)
    elapsed = time.perf_counter() - t0
    m = result.metrics
    return {
        "elapsed_sec": round(elapsed, 2),
        "wall_sec": round(m.get("wall_seconds", elapsed), 2),
        "epochs_completed": m.get("epochs", epochs),
        "sec_per_epoch": round(elapsed / max(m.get("epochs", epochs), 1), 3),
        "val_accuracy": m.get("final_val_accuracy", m.get("best_accuracy")),
        "val_loss": m.get("final_val_loss"),
        "checkpoint": str(out / name / "checkpoint.pt"),
    }


# ---------------------------------------------------------------------------
# Training benchmarks
# ---------------------------------------------------------------------------

def bench_nano_training(out: Path, train_records, dev_records, epochs: int) -> list[dict]:
    """nano metadata MLP — illustrates early-stopping and scheduler effects."""
    print("\n" + "="*60)
    print("TRACK 1: nano (metadata MLP) — scheduler & early-stopping")
    print(f"  {len(train_records)} train / {len(dev_records)} dev records, {epochs} max epochs")

    configs = [
        ("nano_baseline_cuda", dict(device="cuda", amp=False, num_workers=0, pin_memory=False, lr_schedule=None)),
        ("nano_amp",           dict(device="cuda", amp=True,  num_workers=0, pin_memory=False, lr_schedule=None)),
        ("nano_cosine",        dict(device="cuda", amp=True,  num_workers=0, pin_memory=False, lr_schedule="cosine", lr_min=1e-5)),
        ("nano_cosine_es",     dict(device="cuda", amp=True,  num_workers=0, pin_memory=False, lr_schedule="cosine", lr_min=1e-5,
                                    early_stopping_patience=5)),
    ]
    return _run_configs("nano", out, train_records, dev_records, epochs, configs,
                        model_type="nanoturn_nano", feature_source="metadata")


def bench_micro_training(out: Path, train_records, dev_records, epochs: int) -> list[dict]:
    """micro audio TCN — compute-heavy, AMP gives real speedup here."""
    print("\n" + "="*60)
    print("TRACK 2: micro (audio TCN) — AMP training speedup")
    print(f"  {len(train_records)} train / {len(dev_records)} dev records, {epochs} max epochs")

    configs = [
        ("micro_baseline",    dict(device="cuda", amp=False, num_workers=0, pin_memory=False, lr_schedule=None)),
        ("micro_amp",         dict(device="cuda", amp=True,  num_workers=2, pin_memory=True,  lr_schedule=None)),
        ("micro_amp_cosine",  dict(device="cuda", amp=True,  num_workers=2, pin_memory=True,  lr_schedule="cosine", lr_min=1e-5)),
        ("micro_amp_dw",      dict(device="cuda", amp=True,  num_workers=2, pin_memory=True,  lr_schedule="cosine", lr_min=1e-5,
                                    depthwise=True)),
    ]
    return _run_configs("micro", out, train_records, dev_records, epochs, configs,
                        model_type="nanoturn_micro", feature_source="audio_seq", batch_size=64)


def _run_configs(track: str, out: Path, train_records, dev_records, epochs: int,
                 configs: list, **common_kw) -> list[dict]:
    results = []
    baseline_time = None
    for name, kw in configs:
        print(f"\n  [{track}] {name}")
        r = run_training(name, out, train_records, dev_records, epochs=epochs, **common_kw, **kw)
        if baseline_time is None:
            baseline_time = r["elapsed_sec"]
        speedup = baseline_time / r["elapsed_sec"]
        print(f"    {r['elapsed_sec']:.1f}s | {r['epochs_completed']} ep | "
              f"{r['sec_per_epoch']:.3f}s/ep | acc={r['val_accuracy']:.4f} | "
              f"speedup={speedup:.2f}x")
        results.append({"name": name, "track": track, "speedup_vs_baseline": round(speedup, 3), **r, **kw})
    return results


# ---------------------------------------------------------------------------
# Inference benchmarks
# ---------------------------------------------------------------------------

def bench_inference(out: Path, dev_records) -> list[dict]:
    print("\n" + "="*60)
    print("TRACK 3: inference (batch vs single vs ONNX)")

    from stable_asr.train.turn_trainer import NanoTurnCheckpointPredictor

    results = []
    candidates = [
        ("nano_cosine",    "nano"),
        ("nano_baseline_cuda", "nano"),
        ("micro_amp_cosine", "micro"),
        ("micro_baseline",   "micro"),
    ]

    for folder, track in candidates:
        ckpt = out / folder / "checkpoint.pt"
        if not ckpt.exists():
            continue

        N = len(dev_records)
        ITERS = 5
        print(f"\n  [{track}] {folder}  (N={N})")

        # PyTorch batch
        pred = NanoTurnCheckpointPredictor(ckpt)
        pred.predict_batch(dev_records[:10])  # warmup
        times_batch = [_time_batch(pred, dev_records) for _ in range(ITERS)]
        avg_batch = sum(times_batch) / ITERS

        # PyTorch single
        times_single = [_time_single(pred, dev_records) for _ in range(3)]
        avg_single = sum(times_single) / 3

        batch_speedup = avg_single / avg_batch
        print(f"    batch:  {avg_batch*1000:.1f}ms  ({N/avg_batch:.0f} rec/s)")
        print(f"    single: {avg_single*1000:.1f}ms  ({N/avg_single:.0f} rec/s)")
        print(f"    batch speedup: {batch_speedup:.1f}x")

        row = {
            "name": folder, "track": track,
            "n_records": N,
            "pytorch_batch_ms": round(avg_batch * 1000, 2),
            "pytorch_single_ms": round(avg_single * 1000, 2),
            "batch_speedup": round(batch_speedup, 2),
        }

        # ONNX Runtime
        onnx_path = export_onnx(ckpt)
        if onnx_path:
            try:
                pred_ort = NanoTurnCheckpointPredictor(ckpt, onnx_path=onnx_path)
                pred_ort.predict_batch(dev_records[:10])  # warmup
                times_ort = [_time_batch(pred_ort, dev_records) for _ in range(ITERS)]
                avg_ort = sum(times_ort) / ITERS
                ort_speedup = avg_batch / avg_ort
                print(f"    ONNX:   {avg_ort*1000:.1f}ms  ({N/avg_ort:.0f} rec/s)  ({ort_speedup:.1f}x vs PT batch)")
                row["onnx_batch_ms"] = round(avg_ort * 1000, 2)
                row["onnx_speedup_vs_pt"] = round(ort_speedup, 2)
            except Exception as e:
                print(f"    ONNX runtime error: {e}")

        results.append(row)
        break  # one checkpoint is enough to demonstrate; remove to bench all

    return results


def _time_batch(pred, records) -> float:
    t0 = time.perf_counter()
    pred.predict_batch(records)
    return time.perf_counter() - t0


def _time_single(pred, records) -> float:
    t0 = time.perf_counter()
    for r in records:
        pred.predict(r)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark acceleration optimisations")
    parser.add_argument("--train", default="runs/final/turn_train.jsonl")
    parser.add_argument("--dev",   default="runs/final/turn_dev.jsonl")
    parser.add_argument("--output", default="runs/bench_accel")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max training epochs per config")
    parser.add_argument("--micro-epochs", type=int, default=10,
                        help="Max epochs for micro TCN configs (heavier)")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip training, only run inference benchmark on existing checkpoints")
    parser.add_argument("--no-micro", action="store_true",
                        help="Skip micro TCN training benchmark")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl(args.train)
    dev_records   = load_jsonl(args.dev)
    print(f"Train: {len(train_records)} | Dev: {len(dev_records)}")

    all_results: dict[str, list[dict]] = {}

    if not args.inference_only:
        all_results["nano"] = bench_nano_training(out, train_records, dev_records, args.epochs)

        if not args.no_micro:
            # Filter to records that have audio files (micro needs audio_seq)
            micro_train = [r for r in train_records if r.audio and Path(r.audio).exists()]
            micro_dev   = [r for r in dev_records   if r.audio and Path(r.audio).exists()]
            if micro_train:
                all_results["micro"] = bench_micro_training(
                    out, micro_train, micro_dev, args.micro_epochs)
            else:
                print("\nSkipping micro benchmark — no local audio files found.")

    all_results["inference"] = bench_inference(out, dev_records)

    # ---- Summary table ----
    print("\n" + "="*60)
    print("SUMMARY")
    print(f"{'name':<25} {'time(s)':>8} {'ep':>4} {'s/ep':>7} {'acc':>7} {'speedup':>9}")
    print("-" * 65)
    for track, rows in all_results.items():
        if track == "inference":
            continue
        for r in rows:
            print(f"{r['name']:<25} {r['elapsed_sec']:>8.1f} {r['epochs_completed']:>4} "
                  f"{r['sec_per_epoch']:>7.3f} {r.get('val_accuracy', '?'):>7.4f} "
                  f"{r.get('speedup_vs_baseline', 1.0):>8.2f}x")

    print("\nInference:")
    for r in all_results.get("inference", []):
        ort = f"  ONNX={r.get('onnx_batch_ms','?')}ms({r.get('onnx_speedup_vs_pt','?')}x)" if "onnx_batch_ms" in r else ""
        print(f"  {r['name']:<25} batch={r['pytorch_batch_ms']}ms  single={r['pytorch_single_ms']}ms  "
              f"batch_speedup={r['batch_speedup']}x{ort}")

    # Save
    report = out / "acceleration_results.json"
    report.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results → {report}")


if __name__ == "__main__":
    main()
