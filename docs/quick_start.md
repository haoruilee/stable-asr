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
# 0. Inspect the checked-in platform catalog
stable-asr catalog --output runs/PLATFORM_CATALOG.md

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
stable-asr train-turn --dataset examples/data/turn_demo.jsonl --output-dir runs/nanoturn_nano --config configs/nanoturn_nano.json --epochs 5

# 3. Package a streaming ASR trace for leaderboard-style review
stable-asr streaming-submission \
  --input tests/fixtures/streaming_asr_sample.jsonl \
  --system streaming_fixture \
  --output-dir runs/submissions/streaming_fixture
stable-asr submission-index \
  --root runs/submissions \
  --output-dir runs/submissions/leaderboard
stable-asr adapter-pack --output-dir runs/adapter_pack
stable-asr scenario-pack --output-dir runs/scenario_pack

# 4. Generate paper-facing evidence
stable-asr doctor --check-release-env
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke
stable-asr final-pack --output-dir runs/final_pack
stable-asr final-acquisition-pack --output-dir runs/final_acquisition_pack
stable-asr final-assignment-audit --input runs/final_acquisition_pack/acquisition/assignments.json
stable-asr reference-workqueue --output runs/REFERENCE_WORKQUEUE.md
stable-asr reference-workqueue --format assignments-json --output runs/reference_assignments.json
stable-asr reference-workqueue --format assignments-tsv --output runs/reference_assignments.tsv
stable-asr reference-assignment-audit --input runs/reference_assignments.json --output runs/REFERENCE_ASSIGNMENT_AUDIT.md
stable-asr contributor-pack --output-dir runs/contributor_pack
```

## What The Smoke Run Produces

`paper-release-smoke` writes:

- `paper/paper_results.json`
- `artifacts/` with copied paper results, tables, figures, registries, case studies, paper/platform parity audits, claims, roadmap status, benchmark suite files, provenance, and integrity hashes
- `artifacts/PLATFORM_CATALOG.md` and `artifacts/platform_catalog.json`
- `artifacts.tar.gz` and `artifacts.tar.gz.sha256`
- `PAPER_DRAFT.md`
- `paper.tex`
- `DATASET_CARD.md`
- `EXPERIMENT_CARD.md`
- `MODEL_CARD.md`
- `artifacts/PAPER_STATUS.md`
- `release_audit.json`
- `RELEASE_AUDIT.md`

Use `--skip-train` for a faster structural run. Use `doctor
--check-release-env` before `--strict`; strict smoke requires both the optional
Lance data backend and NanoTurn training dependencies.
Use `paper-release-smoke --require-final-ready` for a final gate that must fail
until the release audit is READY and real corpora, external predictions, and
final artifacts are present.
