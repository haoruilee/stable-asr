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
  # Train all 5 NanoTurn variants; export ONNX for each.
  local base="runs/final"

  # pico: MLP 1-layer, metadata 8-dim
  run stable-asr train-turn \
    --dataset "${base}/turn_train.jsonl" \
    --dev-dataset "${base}/turn_dev.jsonl" \
    --output-dir "${base}/nanoturn_pico" \
    --model nanoturn_pico \
    --feature-source metadata \
    --json
  run stable-asr export-turn-onnx \
    --checkpoint "${base}/nanoturn_pico/checkpoint.pt" \
    --output "${base}/nanoturn_pico/nanoturn_pico.onnx"

  # nano: MLP 2-layer, metadata 8-dim
  run stable-asr train-turn \
    --dataset "${base}/turn_train.jsonl" \
    --dev-dataset "${base}/turn_dev.jsonl" \
    --output-dir "${base}/nanoturn_nano" \
    --model nanoturn_nano \
    --feature-source metadata \
    --json
  run stable-asr export-turn-onnx \
    --checkpoint "${base}/nanoturn_nano/checkpoint.pt" \
    --output "${base}/nanoturn_nano/nanoturn_nano.onnx"

  # pico_v1: MLP 2-layer, logmel_v1 160-dim
  run stable-asr train-turn \
    --dataset "${base}/turn_train.jsonl" \
    --dev-dataset "${base}/turn_dev.jsonl" \
    --output-dir "${base}/nanoturn_pico_v1" \
    --model nanoturn_pico_v1 \
    --feature-source logmel_v1 \
    --json
  run stable-asr export-turn-onnx \
    --checkpoint "${base}/nanoturn_pico_v1/checkpoint.pt" \
    --output "${base}/nanoturn_pico_v1/nanoturn_pico_v1.onnx"

  # nano_v1: MLP 3-layer, logmel_v1 160-dim
  run stable-asr train-turn \
    --dataset "${base}/turn_train.jsonl" \
    --dev-dataset "${base}/turn_dev.jsonl" \
    --output-dir "${base}/nanoturn_nano_v1" \
    --model nanoturn_nano_v1 \
    --feature-source logmel_v1 \
    --json
  run stable-asr export-turn-onnx \
    --checkpoint "${base}/nanoturn_nano_v1/checkpoint.pt" \
    --output "${base}/nanoturn_nano_v1/nanoturn_nano_v1.onnx"

  # micro: TCN 4-block dilated causal, audio_seq (T, 80)
  run stable-asr train-turn \
    --dataset "${base}/turn_train.jsonl" \
    --dev-dataset "${base}/turn_dev.jsonl" \
    --output-dir "${base}/nanoturn_micro" \
    --model nanoturn_micro \
    --feature-source audio_seq \
    --json
  run stable-asr export-turn-onnx \
    --checkpoint "${base}/nanoturn_micro/checkpoint.pt" \
    --output "${base}/nanoturn_micro/nanoturn_micro.onnx"
}

evaluate_core() {
  local base="runs/final"

  # Build checkpoint args for all trained models
  local ckpt_args=()
  for model in nanoturn_pico nanoturn_nano nanoturn_pico_v1 nanoturn_nano_v1 nanoturn_micro; do
    local ckpt="${base}/${model}/checkpoint.pt"
    [[ -f "${ckpt}" ]] && ckpt_args+=(--checkpoint "${model}=${ckpt}")
  done

  run stable-asr compare-turn \
    --dataset "${base}/turn_test.jsonl" \
    --baseline rule_endpoint \
    --baseline vad_pause \
    --baseline text_turn \
    --predictions smart_turn="${base}/external/smartturn_predictions.jsonl" \
    --predictions easy_turn="${base}/external/easyturn_predictions.jsonl" \
    --predictions vap="${base}/external/vap_predictions.jsonl" \
    "${ckpt_args[@]}" \
    --report "${base}/reports/baselines.md" \
    --json-output "${base}/reports/baselines.json"

  # Benchmark and scenario eval for the primary model (nanoturn_nano)
  local primary_ckpt="${base}/nanoturn_nano/checkpoint.pt"
  if [[ -f "${primary_ckpt}" ]]; then
    run stable-asr benchmark-turn \
      --dataset "${base}/turn_test.jsonl" \
      --checkpoint "${primary_ckpt}" \
      --artifact "${primary_ckpt}" \
      --report "${base}/reports/turn_benchmarks.md" \
      --json-output "${base}/reports/turn_benchmarks.json"
    run stable-asr eval-scenario \
      --dataset "${base}/voiceworld_real.jsonl" \
      --checkpoint "${primary_ckpt}" \
      --seed 0 \
      --report "${base}/reports/scenarios.md" \
      --json-output "${base}/reports/scenarios.json"
  fi
}

ablations() {
  bash scripts/run_ablations.sh all \
    2>&1 | tee runs/ablations/run.log
}

full_eval() {
  bash scripts/run_eval.sh all \
    2>&1 | tee runs/eval/run.log
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
    --model-id nanoturn_nano \
    --metrics runs/final/nanoturn_nano/metrics.json \
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
  ablations) ablations ;;
  full-eval) full_eval ;;
  asr-adapters) asr_adapters ;;
  bundle) bundle ;;
  final)
    status
    prepare_inputs
    data_layer
    train_nanoturn
    evaluate_core
    ablations
    full_eval
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
  train           train all 5 NanoTurn variants and export ONNX
  evaluate        run turn, latency, and VoiceWorld evaluations
  ablations       run feature/arch/scale/seed ablation experiments
  full-eval       run consolidated eval across all trained checkpoints
  asr-adapters    run configured command-backed ASR adapter comparisons
  bundle          assemble final results, cards, archive, and release audit
  final           run the full sequence after inputs are staged
EOF
    exit 2
    ;;
esac
