#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-status}"
CONFIG="${CONFIG:-configs/final/paper_final.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

run() {
  printf '\n+ %s\n' "$*"
  "$@"
}

ensure_venv() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    run "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  run python -m pip install -U pip
}

activate_venv_if_available() {
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  fi
}

setup() {
  ensure_venv
  run python -m pip install -e ".[all]"
  run stable-asr doctor --repo-root . --check-release-env
}

status() {
  run stable-asr doctor --repo-root . --check-final-files
  run stable-asr final-config --config "${CONFIG}" --check-files
  run stable-asr final-config --config "${CONFIG}" --plan-missing --output runs/final/FINAL_RUN_ACTION_PLAN.md
}

prepare_inputs() {
  run stable-asr final-config --config "${CONFIG}" --scaffold
  run stable-asr final-config --config "${CONFIG}" --prepare-inputs
  run stable-asr final-config --config "${CONFIG}" --check-files
}

data_layer() {
  run stable-asr benchmark-data \
    --dataset runs/final/turn_train.jsonl \
    --output-dir runs/final/data_bench \
    --formats jsonl parquet lance \
    --sample-count 10000 \
    --json-output runs/final/reports/data_benchmark.json
  run stable-asr benchmark-audio-windows \
    --dataset runs/final/voiceworld_real.jsonl \
    --output-dir runs/final/audio_window_bench_correctness \
    --formats source_wav parquet lance \
    --sample-count 10000 \
    --correctness-sample-count 10000 \
    --json-output runs/final/reports/audio_window_benchmark_correctness.json
  run stable-asr benchmark-train-features \
    --dataset runs/final/voiceworld_real.jsonl \
    --output-dir runs/final/train_feature_bench_10k_correctness \
    --formats source_audio source_audio_file_cache parquet lance \
    --sample-count 10000 \
    --correctness-sample-count 10000 \
    --json-output runs/final/reports/train_feature_benchmark_10k_correctness.json
  run stable-asr benchmark-train-features \
    --dataset runs/final/voiceworld_real.jsonl \
    --output-dir runs/final/train_feature_bench_100k_cached_correctness \
    --formats parquet lance \
    --sample-count 100000 \
    --correctness-sample-count 10000 \
    --json-output runs/final/reports/train_feature_benchmark_100k_cached_correctness.json
}

train_nanoturn() {
  run stable-asr train-turn \
    --dataset runs/final/turn_train.jsonl \
    --output-dir runs/final/nanoturn \
    --model nanoturn_pico \
    --feature-source audio \
    --feature-cache runs/final/nanoturn/logmel_features.lance \
    --feature-cache-format lance
  run stable-asr export-turn-onnx \
    --checkpoint runs/final/nanoturn/checkpoint.pt \
    --output runs/final/nanoturn/nanoturn.onnx
}

evaluate_core() {
  run stable-asr compare-turn \
    --dataset runs/final/turn_test.jsonl \
    --baseline rule_endpoint \
    --baseline vad_pause \
    --baseline text_turn \
    --predictions smart_turn=runs/final/external/smartturn_predictions.jsonl \
    --predictions easy_turn=runs/final/external/easyturn_predictions.jsonl \
    --predictions vap=runs/final/external/vap_predictions.jsonl \
    --checkpoint nanoturn=runs/final/nanoturn/checkpoint.pt \
    --report runs/final/reports/baselines.md \
    --json-output runs/final/reports/baselines.json
  run stable-asr benchmark-turn \
    --dataset runs/final/turn_test.jsonl \
    --checkpoint runs/final/nanoturn/checkpoint.pt \
    --artifact runs/final/nanoturn/checkpoint.pt \
    --artifact runs/final/nanoturn/metrics.json \
    --report runs/final/reports/turn_benchmarks.md \
    --json-output runs/final/reports/turn_benchmarks.json
  run stable-asr eval-scenario \
    --dataset runs/final/voiceworld_real.jsonl \
    --checkpoint runs/final/nanoturn/checkpoint.pt \
    --seed 0 \
    --report runs/final/reports/scenarios.md \
    --json-output runs/final/reports/scenarios.json
}

asr_adapters() {
  run stable-asr final-config --config "${CONFIG}" --audit-asr-commands
  run stable-asr compare-asr-commands \
    --config configs/final/asr_command_compare.json \
    --report runs/final/reports/asr_command_compare.md \
    --json-output runs/final/reports/asr_command_compare.json
  run stable-asr final-config --config "${CONFIG}" --prepare-asr-transcript-conversions
}

bundle() {
  run stable-asr final-results --config "${CONFIG}" --output runs/final/paper_results.json
  run stable-asr make-card model \
    --input configs/models/stable_asr_models.json \
    --model-id nanoturn_pico \
    --metrics runs/final/nanoturn/metrics.json \
    --output runs/final/MODEL_CARD.md
  run stable-asr paper-bundle --results runs/final/paper_results.json --output-dir runs/final/artifacts
  run stable-asr paper-archive --artifacts-dir runs/final/artifacts --output runs/final/artifacts.tar.gz
  run stable-asr paper-archive-verify --archive runs/final/artifacts.tar.gz
  run stable-asr paper-release-audit \
    --repo-root . \
    --results runs/final/paper_results.json \
    --artifacts-dir runs/final/artifacts \
    --model-card runs/final/MODEL_CARD.md \
    --require-final-ready
}

if [[ "${PHASE}" != "setup" ]]; then
  activate_venv_if_available
fi

case "${PHASE}" in
  setup) setup ;;
  status) status ;;
  prepare-inputs) prepare_inputs ;;
  data-layer) data_layer ;;
  train) train_nanoturn ;;
  evaluate) evaluate_core ;;
  asr-adapters) asr_adapters ;;
  bundle) bundle ;;
  final)
    status
    prepare_inputs
    data_layer
    train_nanoturn
    evaluate_core
    asr_adapters
    bundle
    ;;
  *)
    cat >&2 <<'EOF'
Usage: scripts/run_final_machine.sh <phase>

Phases:
  setup           create .venv and install all optional Stable-ASR dependencies
  status          run doctor, check final files, and write missing-input plan
  prepare-inputs  scaffold and prepare configured final inputs that exist
  data-layer      run JSONL/Parquet/Lance and correctness-checked cache benchmarks
  train           train NanoTurn and export ONNX
  evaluate        run turn, latency, and VoiceWorld evaluations
  asr-adapters    run configured command-backed ASR adapter comparisons
  bundle          assemble final results, cards, archive, and release audit
  final           run the full sequence after inputs are staged
EOF
    exit 2
    ;;
esac
