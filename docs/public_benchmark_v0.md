# Stable-ASR Public Benchmark v0

`configs/benchmarks/stable_asr_public_v0.json` is the first checked-in public
benchmark contract for Stable-ASR. It is designed to be reproducible without
committing public corpus audio or generated model artifacts.

## Scope

- ASR corpora: LibriSpeech `dev-clean` and AISHELL-1 `dev`.
- Turn/action splits: `turn_train`, `turn_dev`, and `turn_test`.
- Scenario suite: `voiceworld_real` over incomplete pause, backchannel,
  wait/stop, interruption, side conversation, ambient speech, noisy far-field,
  and code-switching slices.
- External ASR: precomputed transcript converters plus optional runtime runners
  for OpenAI Whisper, FunASR, and whisper.cpp.
- External turn: SmartTurn, EasyTurn, and VAP raw prediction exports normalized
  through a shared coverage-checked bridge.
- Data layer: JSONL/Parquet/Lance manifest benchmarks and cached log-mel
  training-window benchmarks.

## Observed Local Counts

These counts were observed in the local final workspace on 2026-06-01:

| artifact | records |
| --- | ---: |
| `runs/final/asr_eval_manifest.jsonl` | 3067 |
| `runs/final/turn_train.jsonl` | 4906 |
| `runs/final/turn_dev.jsonl` | 614 |
| `runs/final/turn_test.jsonl` | 614 |
| `runs/final/voiceworld_real.jsonl` | 180 |

The repository intentionally keeps `data/` and `runs/` ignored. The benchmark
config, schema, commands, and fixtures are committed; heavyweight corpora,
checkpoints, raw ASR outputs, and generated reports are regenerated locally.

## Reproduce

```bash
stable-asr prepare-public-asr \
  --corpus librispeech \
  --input-dir data/librispeech/LibriSpeech/dev-clean \
  --output runs/final/librispeech_dev_clean/asr_manifest.jsonl

stable-asr prepare-public-asr \
  --corpus aishell1 \
  --input-dir data/aishell1/data_aishell \
  --split dev \
  --output runs/final/aishell1_dev/asr_manifest.jsonl

stable-asr final-config --config configs/final/paper_final.json --prepare-asr-eval-manifest
stable-asr final-config --config configs/final/paper_final.json --bootstrap-turn-splits
stable-asr final-config --config configs/final/paper_final.json --prepare-voiceworld-real
```

Runtime ASR runners are optional because they pull large upstream dependencies
or model weights:

```bash
python3 scripts/run_whisper_streaming.py \
  --manifest runs/final/asr_eval_manifest.jsonl \
  --model tiny \
  --device cpu \
  --output runs/final/asr_commands/raw/whisper_tiny_raw.jsonl

python3 scripts/export_streaming_transcript.py \
  --schema whisper \
  --manifest runs/final/asr_eval_manifest.jsonl \
  --raw runs/final/asr_commands/raw/whisper_tiny_raw.jsonl \
  --output runs/final/asr_commands/whisper_tiny_streaming.jsonl
```

Turn prediction exports are normalized and coverage checked with:

```bash
python3 scripts/export_turn_predictions.py \
  --schema smart_turn \
  --dataset runs/final/turn_test.jsonl \
  --raw runs/final/external/smartturn_raw.jsonl \
  --output runs/final/external/smartturn_predictions.jsonl
```

## Stable-WorldModel Alignment

Stable-worldmodel exposes a unified collect/train/evaluate interface, a
training-oriented Lance data layer, baseline and solver implementations, and
controlled environments. Stable-ASR mirrors those roles in the speech domain:

| stable-worldmodel role | Stable-ASR v0 counterpart |
| --- | --- |
| dataset registry and Lance tables | ASR/turn manifests plus Parquet/Lance and log-mel training-window caches |
| world model baselines | Rule endpoint, VAD pause, text turn, NanoTurn, prediction-manifest baselines |
| planning solvers | threshold and cost-sensitive turn policy optimization |
| environments and variation factors | VoiceWorld scenarios and public-corpus slices |
| reproducible evaluation | benchmark configs, leaderboard rows, paper bundles, cards, audits |

The remaining paper-grade step is not more scaffolding; it is running the
optional upstream systems on the full public benchmark with recorded model
versions, weights, hardware, and license notes.
