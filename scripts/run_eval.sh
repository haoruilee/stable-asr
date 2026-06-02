#!/usr/bin/env bash
# Full evaluation runner for all trained NanoTurn checkpoints.
#
# Usage:
#   scripts/run_eval.sh <phase>
#
# Phases:
#   all        run compare-turn, benchmark-turn, eval-scenario for every checkpoint
#   compare    compare-turn only (baselines + all checkpoints on test split)
#   benchmark  benchmark-turn latency/accuracy table
#   scenario   VoiceWorld per-scenario evaluation
#   report     generate consolidated markdown report
#
# Environment variables:
#   TEST_MANIFEST    path to test JSONL           (default: runs/final/turn_test.jsonl)
#   VW_MANIFEST      VoiceWorld JSONL             (default: runs/final/voiceworld_real.jsonl)
#   TRAIN_OUT        base directory of trained models (default: runs/final)
#   ABLATION_OUT     base directory of ablation runs  (default: runs/ablations)
#   EVAL_OUT         evaluation report output dir     (default: runs/eval)
#   VENV_DIR         virtualenv path                  (default: .venv)
set -euo pipefail

PHASE="${1:-all}"
TEST_MANIFEST="${TEST_MANIFEST:-runs/final/turn_test.jsonl}"
VW_MANIFEST="${VW_MANIFEST:-runs/final/voiceworld_real.jsonl}"
TRAIN_OUT="${TRAIN_OUT:-runs/final}"
ABLATION_OUT="${ABLATION_OUT:-runs/ablations}"
EVAL_OUT="${EVAL_OUT:-runs/eval}"
VENV_DIR="${VENV_DIR:-.venv}"

run() {
  printf '\n\033[1;34m+ %s\033[0m\n' "$*"
  "$@"
}

activate_venv() {
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  fi
}

# ---------------------------------------------------------------------------
# Collect all checkpoint paths from trained models
# Returns: space-separated "label=path" pairs
# ---------------------------------------------------------------------------
collect_checkpoints() {
  local checkpoints=()

  # Core trained models under runs/final/
  local core_models=(
    "nanoturn_pico"
    "nanoturn_nano"
    "nanoturn_pico_v1"
    "nanoturn_nano_v1"
    "nanoturn_micro"
  )
  for model in "${core_models[@]}"; do
    local ckpt="${TRAIN_OUT}/${model}/checkpoint.pt"
    if [[ -f "${ckpt}" ]]; then
      checkpoints+=("${model}=${ckpt}")
    fi
  done

  # Ablation feature variants (seed 0 only for comparison table)
  if [[ -d "${ABLATION_OUT}/feature" ]]; then
    for variant_dir in "${ABLATION_OUT}/feature"/*/seed_0; do
      [[ -d "${variant_dir}" ]] || continue
      local ckpt="${variant_dir}/checkpoint.pt"
      [[ -f "${ckpt}" ]] || continue
      local label
      label="abl_$(basename "$(dirname "${variant_dir}")")"
      checkpoints+=("${label}=${ckpt}")
    done
  fi

  # Architecture ablation (seed 0)
  if [[ -d "${ABLATION_OUT}/arch" ]]; then
    for variant_dir in "${ABLATION_OUT}/arch"/*/seed_0; do
      [[ -d "${variant_dir}" ]] || continue
      local ckpt="${variant_dir}/checkpoint.pt"
      [[ -f "${ckpt}" ]] || continue
      local label
      label="arch_$(basename "$(dirname "${variant_dir}")")"
      checkpoints+=("${label}=${ckpt}")
    done
  fi

  echo "${checkpoints[@]}"
}

# ---------------------------------------------------------------------------
# COMPARE TURN
# ---------------------------------------------------------------------------
run_compare() {
  local report_dir="${EVAL_OUT}/compare_turn"
  mkdir -p "${report_dir}"
  echo "=== compare-turn ==="

  local -a checkpoint_args=()
  for ckpt_pair in $(collect_checkpoints); do
    checkpoint_args+=(--checkpoint "${ckpt_pair}")
  done

  if [[ ${#checkpoint_args[@]} -eq 0 ]]; then
    echo "No checkpoints found; skipping compare-turn."
    return
  fi

  run stable-asr compare-turn \
    --dataset "${TEST_MANIFEST}" \
    --baseline rule_endpoint \
    --baseline vad_pause \
    --baseline text_turn \
    "${checkpoint_args[@]}" \
    --report "${report_dir}/baselines.md" \
    --json-output "${report_dir}/baselines.json"

  echo "Report: ${report_dir}/baselines.md"
}

# ---------------------------------------------------------------------------
# BENCHMARK TURN (latency + accuracy table)
# ---------------------------------------------------------------------------
run_benchmark() {
  local report_dir="${EVAL_OUT}/benchmark"
  mkdir -p "${report_dir}"
  echo "=== benchmark-turn ==="

  local core_models=("nanoturn_pico" "nanoturn_nano" "nanoturn_pico_v1" "nanoturn_nano_v1" "nanoturn_micro")
  for model in "${core_models[@]}"; do
    local ckpt="${TRAIN_OUT}/${model}/checkpoint.pt"
    [[ -f "${ckpt}" ]] || continue

    run stable-asr benchmark-turn \
      --dataset "${TEST_MANIFEST}" \
      --checkpoint "${ckpt}" \
      --artifact "${ckpt}" \
      --report "${report_dir}/${model}.md" \
      --json-output "${report_dir}/${model}.json"
  done
}

# ---------------------------------------------------------------------------
# SCENARIO EVAL (VoiceWorld 9-scenario breakdown)
# ---------------------------------------------------------------------------
run_scenario() {
  local report_dir="${EVAL_OUT}/scenarios"
  mkdir -p "${report_dir}"
  echo "=== eval-scenario (VoiceWorld) ==="

  if [[ ! -f "${VW_MANIFEST}" ]]; then
    echo "VoiceWorld manifest not found: ${VW_MANIFEST} — skipping scenario eval."
    return
  fi

  local core_models=("nanoturn_pico" "nanoturn_nano" "nanoturn_pico_v1" "nanoturn_nano_v1" "nanoturn_micro")
  for model in "${core_models[@]}"; do
    local ckpt="${TRAIN_OUT}/${model}/checkpoint.pt"
    [[ -f "${ckpt}" ]] || continue

    run stable-asr eval-scenario \
      --dataset "${VW_MANIFEST}" \
      --checkpoint "${ckpt}" \
      --seed 0 \
      --report "${report_dir}/${model}_scenarios.md" \
      --json-output "${report_dir}/${model}_scenarios.json"
  done
}

# ---------------------------------------------------------------------------
# CONSOLIDATED REPORT
# ---------------------------------------------------------------------------
run_report() {
  local report_dir="${EVAL_OUT}"
  mkdir -p "${report_dir}"
  echo "=== Consolidated evaluation report ==="

  python3 - "${report_dir}" <<'PYEOF'
import sys, json, pathlib

root = pathlib.Path(sys.argv[1])
lines = ["# NanoTurn Evaluation Report\n"]

# compare_turn baselines
baseline_json = root / "compare_turn" / "baselines.json"
if baseline_json.exists():
    lines.append("## Baselines (compare-turn on test split)\n")
    try:
        data = json.loads(baseline_json.read_text())
        rows = data if isinstance(data, list) else data.get("rows", [])
        if rows:
            keys = list(rows[0].keys())
            lines.append("| " + " | ".join(keys) + " |")
            lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
            for row in rows:
                lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
        else:
            lines.append(f"```json\n{baseline_json.read_text()}\n```")
    except Exception as e:
        lines.append(f"_(parse error: {e})_")
    lines.append("")

# benchmark results
bench_dir = root / "benchmark"
if bench_dir.exists():
    lines.append("## Benchmark (per-model latency + accuracy)\n")
    for jf in sorted(bench_dir.glob("*.json")):
        lines.append(f"### {jf.stem}\n")
        try:
            data = json.loads(jf.read_text())
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                for k, v in row.items():
                    lines.append(f"- **{k}**: {v}")
        except Exception as e:
            lines.append(f"_(parse error: {e})_")
        lines.append("")

# scenario results
scenario_dir = root / "scenarios"
if scenario_dir.exists():
    lines.append("## VoiceWorld Scenarios\n")
    for jf in sorted(scenario_dir.glob("*.json")):
        lines.append(f"### {jf.stem}\n")
        try:
            data = json.loads(jf.read_text())
            rows = data if isinstance(data, list) else data.get("rows", [])
            if rows:
                keys = list(rows[0].keys())
                lines.append("| " + " | ".join(keys) + " |")
                lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
                for row in rows:
                    lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
            else:
                lines.append(f"```json\n{jf.read_text()[:500]}\n```")
        except Exception as e:
            lines.append(f"_(parse error: {e})_")
        lines.append("")

out = root / "EVAL_REPORT.md"
out.write_text("\n".join(lines))
print(f"Report written: {out}")
PYEOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
activate_venv

mkdir -p "${EVAL_OUT}"

case "${PHASE}" in
  all)
    run_compare
    run_benchmark
    run_scenario
    run_report
    ;;
  compare)    run_compare ;;
  benchmark)  run_benchmark ;;
  scenario)   run_scenario ;;
  report)     run_report ;;
  *)
    cat >&2 <<'EOF'
Usage: scripts/run_eval.sh <phase>

Phases:
  all        run all evaluation phases
  compare    compare-turn: all checkpoints vs baselines on test split
  benchmark  benchmark-turn: latency + accuracy per model
  scenario   eval-scenario: VoiceWorld 9-scenario breakdown
  report     generate consolidated EVAL_REPORT.md

Key env vars:
  TEST_MANIFEST  VW_MANIFEST  TRAIN_OUT  ABLATION_OUT  EVAL_OUT
EOF
    exit 2
    ;;
esac
