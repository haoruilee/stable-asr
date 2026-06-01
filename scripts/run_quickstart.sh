#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PHASE="${1:-all}"
RUN_DIR="${RUN_DIR:-runs/quickstart}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
EPISODES="${EPISODES:-32}"

run() {
  printf '\n+ %s\n' "$*"
  "$@"
}

activate_venv_if_available() {
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  fi
}

setup_base() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    run "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  run python -m pip install -U pip
  run python -m pip install -e .
}

setup_all() {
  setup_base
  run python -m pip install -e ".[all]"
}

stable_asr() {
  run python -m stable_asr.cli "$@"
}

has_module() {
  python - "$1" <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)
PY
}

has_lance() {
  python - <<'PY'
import sys

try:
    import lance
except Exception:
    sys.exit(1)
sys.exit(0 if hasattr(lance, "dataset") and hasattr(lance, "write_dataset") else 1)
PY
}

ensure_run_dirs() {
  mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/splits" "${RUN_DIR}/reports" "${RUN_DIR}/submissions"
}

make_demo_data() {
  ensure_run_dirs
  stable_asr make-synthetic-turn-data \
    --output "${RUN_DIR}/data/synthetic_turn.jsonl" \
    --episodes "${EPISODES}" \
    --seed 0 \
    --write-audio
  stable_asr split-turn-data \
    --input "${RUN_DIR}/data/synthetic_turn.jsonl" \
    --output-dir "${RUN_DIR}/splits" \
    --seed 0
  stable_asr audit-turn-splits \
    --train "${RUN_DIR}/splits/turn_train.jsonl" \
    --dev "${RUN_DIR}/splits/turn_dev.jsonl" \
    --test "${RUN_DIR}/splits/turn_test.jsonl" \
    --report "${RUN_DIR}/reports/split_audit.md"
}

smoke() {
  ensure_run_dirs
  stable_asr doctor
  stable_asr catalog --output "${RUN_DIR}/PLATFORM_CATALOG.md"
  stable_asr validate-manifest examples/data/turn_demo.jsonl
  stable_asr profile-turn-data \
    --dataset examples/data/turn_demo.jsonl \
    --report "${RUN_DIR}/reports/turn_profile.md"
  stable_asr compare-turn \
    --dataset examples/data/turn_demo.jsonl \
    --baseline vad_pause \
    --baseline text_turn \
    --report "${RUN_DIR}/reports/turn_compare.md" \
    --json-output "${RUN_DIR}/reports/turn_compare.json"
  stable_asr eval-streaming-asr \
    --input tests/fixtures/streaming_asr_sample.jsonl \
    --json-output "${RUN_DIR}/reports/streaming_eval.json"
  stable_asr compare-streaming-asr \
    --input balanced=tests/fixtures/streaming_asr_sample.jsonl \
    --input fast_unstable=tests/fixtures/streaming_asr_fast_unstable_sample.jsonl \
    --report "${RUN_DIR}/reports/streaming_compare.md" \
    --json-output "${RUN_DIR}/reports/streaming_compare.json"
  stable_asr eval-scenario \
    --episodes 21 \
    --seed 0 \
    --baseline vad_pause \
    --report "${RUN_DIR}/reports/scenario.md" \
    --json-output "${RUN_DIR}/reports/scenario.json"
}

data_layer() {
  make_demo_data

  data_formats=(jsonl)
  window_formats=(source_wav)
  feature_formats=(source_audio source_audio_file_cache)
  if has_module pyarrow; then
    data_formats+=(parquet)
    window_formats+=(parquet)
    feature_formats+=(parquet)
  fi
  if has_lance; then
    data_formats+=(lance)
    window_formats+=(lance)
    feature_formats+=(lance)
  fi

  stable_asr benchmark-data \
    --dataset "${RUN_DIR}/data/synthetic_turn.jsonl" \
    --output-dir "${RUN_DIR}/data_bench" \
    --formats "${data_formats[@]}" \
    --sample-count 200 \
    --json-output "${RUN_DIR}/reports/data_benchmark.json"

  stable_asr benchmark-audio-windows \
    --dataset "${RUN_DIR}/data/synthetic_turn.jsonl" \
    --output-dir "${RUN_DIR}/audio_window_bench" \
    --formats "${window_formats[@]}" \
    --sample-count 200 \
    --correctness-sample-count 200 \
    --audio-root "${RUN_DIR}/data" \
    --json-output "${RUN_DIR}/reports/audio_window_benchmark.json"

  if has_module torch; then
    stable_asr benchmark-train-features \
      --dataset "${RUN_DIR}/data/synthetic_turn.jsonl" \
      --output-dir "${RUN_DIR}/train_feature_bench" \
      --formats "${feature_formats[@]}" \
      --sample-count 200 \
      --correctness-sample-count 200 \
      --audio-root "${RUN_DIR}/data" \
      --json-output "${RUN_DIR}/reports/train_feature_benchmark.json"
  else
    printf '\nSkipping train feature benchmark: install Torch with `bash scripts/run_quickstart.sh setup-all`.\n'
  fi
}

train_demo() {
  make_demo_data
  if ! has_module torch; then
    printf '\nSkipping NanoTurn training: install Torch with `bash scripts/run_quickstart.sh setup-all`.\n'
    return 0
  fi
  stable_asr train-turn \
    --dataset "${RUN_DIR}/splits/turn_train.jsonl" \
    --dev-dataset "${RUN_DIR}/splits/turn_dev.jsonl" \
    --output-dir "${RUN_DIR}/nanoturn" \
    --model nanoturn_pico \
    --epochs 2 \
    --batch-size 8 \
    --feature-source audio \
    --audio-root "${RUN_DIR}/data"
  stable_asr eval-turn \
    --dataset "${RUN_DIR}/splits/turn_test.jsonl" \
    --checkpoint "${RUN_DIR}/nanoturn/checkpoint.pt" \
    --audio-root "${RUN_DIR}/data" \
    --json-output "${RUN_DIR}/reports/nanoturn_eval.json"
  if has_module onnx; then
    stable_asr export-turn-onnx \
      --checkpoint "${RUN_DIR}/nanoturn/checkpoint.pt" \
      --output "${RUN_DIR}/nanoturn/nanoturn.onnx"
  else
    printf '\nSkipping ONNX export: install ONNX with `bash scripts/run_quickstart.sh setup-all`.\n'
  fi
}

adapter_demo() {
  ensure_run_dirs
  stable_asr compare-asr-commands \
    --config examples/configs/asr_command_compare_demo.json \
    --report "${RUN_DIR}/reports/asr_command_compare.md" \
    --json-output "${RUN_DIR}/reports/asr_command_compare.json"
  stable_asr compare-asr-commands \
    --config examples/configs/asr_vendor_adapter_demo.json \
    --report "${RUN_DIR}/reports/asr_vendor_adapter.md" \
    --json-output "${RUN_DIR}/reports/asr_vendor_adapter.json"
}

packs() {
  ensure_run_dirs
  stable_asr benchmark-pack --output-dir "${RUN_DIR}/benchmark_pack"
  stable_asr adapter-pack --output-dir "${RUN_DIR}/adapter_pack"
  stable_asr scenario-pack --output-dir "${RUN_DIR}/scenario_pack"
  stable_asr contributor-pack --output-dir "${RUN_DIR}/contributor_pack"
}

if [[ "${PHASE}" != "setup" && "${PHASE}" != "setup-all" ]]; then
  activate_venv_if_available
fi

case "${PHASE}" in
  setup) setup_base ;;
  setup-all) setup_all ;;
  smoke) smoke ;;
  data) data_layer ;;
  train) train_demo ;;
  adapters) adapter_demo ;;
  packs) packs ;;
  all)
    smoke
    data_layer
    train_demo
    adapter_demo
    packs
    ;;
  *)
    cat >&2 <<'EOF'
Usage: scripts/run_quickstart.sh <phase>

Phases:
  setup      create .venv and install the base package
  setup-all  create .venv and install all optional extras
  smoke      run zero-external-data platform, turn, streaming, and scenario checks
  data       generate synthetic audio and run data/cache benchmarks
  train      generate synthetic audio and train/evaluate NanoTurn when Torch is available
  adapters   run command-backed ASR adapter demos on fixtures
  packs      generate contributor starter packs
  all        run smoke, data, train, adapters, and packs

Environment:
  RUN_DIR=/path/to/output   default: runs/quickstart
  EPISODES=64              default: 32 synthetic turn/audio records
  PYTHON_BIN=python         default: python3
  VENV_DIR=.venv            default: .venv in the repository root
EOF
    exit 2
    ;;
esac
