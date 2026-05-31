# Quick Start

Stable-ASR starts with turn-taking and endpointing, then expands into streaming
ASR evaluation, data-layer benchmarking, scenario evaluation, and paper-ready
artifact generation.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -e ".[train]"   # optional NanoTurn training
python -m pip install -e ".[lance]"   # optional Parquet/Lance data backends
python -m pip install -e ".[lance,train]"  # paper-release-smoke READY path
```

## Three-Step Platform Flow

```bash
# 1. Validate and profile turn data
stable-asr validate-manifest examples/data/turn_demo.jsonl
stable-asr profile-turn-data --dataset examples/data/turn_demo.jsonl --report runs/turn_profile.md

# 2. Compare baselines and train NanoTurn
stable-asr compare-turn \
  --dataset examples/data/turn_demo.jsonl \
  --baseline vad_pause \
  --baseline text_turn \
  --report runs/turn_compare.md
stable-asr turn-submission \
  --dataset examples/data/turn_demo.jsonl \
  --predictions tests/fixtures/turn_predictions_sample.jsonl \
  --system oracle_fixture \
  --output-dir runs/submissions/oracle_fixture
stable-asr benchmark-pack --output-dir runs/benchmark_pack
stable-asr train-turn --dataset examples/data/turn_demo.jsonl --output-dir runs/nanoturn

# 3. Package a streaming ASR trace for leaderboard-style review
stable-asr streaming-submission \
  --input tests/fixtures/streaming_asr_sample.jsonl \
  --system streaming_fixture \
  --output-dir runs/submissions/streaming_fixture
stable-asr adapter-pack --output-dir runs/adapter_pack

# 4. Generate paper-facing evidence
stable-asr doctor --check-release-env
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke
```

## What The Smoke Run Produces

`paper-release-smoke` writes:

- `paper/paper_results.json`
- `artifacts/` with copied paper results, tables, figures, registries, case studies, claims, roadmap status, benchmark suite files, provenance, and integrity hashes
- `artifacts.tar.gz` and `artifacts.tar.gz.sha256`
- `PAPER_DRAFT.md`
- `paper.tex`
- `DATASET_CARD.md`
- `EXPERIMENT_CARD.md`
- `MODEL_CARD.md`
- `release_audit.json`
- `RELEASE_AUDIT.md`

Use `--skip-train` for a faster structural run. Use `doctor
--check-release-env` before `--strict`; strict smoke requires both the optional
Lance data backend and NanoTurn training dependencies.
