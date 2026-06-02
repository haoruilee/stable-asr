#!/usr/bin/env bash
# Ablation experiment runner for NanoTurn.
#
# Usage:
#   scripts/run_ablations.sh <phase> [options]
#
# Phases:
#   all           run every ablation group (feature + arch + scale + seed)
#   feature       feature ablation only
#   arch          architecture ablation only
#   scale         data-scale ablation only
#   seed          multi-seed reproducibility run
#   summarise     print a summary table from all ablation metrics JSONs
#
# Environment variables:
#   TRAIN_MANIFEST   path to training JSONL  (default: runs/final/turn_train.jsonl)
#   DEV_MANIFEST     path to dev JSONL        (default: runs/final/turn_dev.jsonl)
#   TEST_MANIFEST    path to test JSONL        (default: runs/final/turn_test.jsonl)
#   VW_MANIFEST      VoiceWorld JSONL          (default: runs/final/voiceworld_real.jsonl)
#   ABLATION_OUT     root output directory     (default: runs/ablations)
#   SEEDS            space-separated seeds     (default: "0 1 2")
#   EPOCHS           training epochs           (default: 30)
#   BATCH_SIZE       batch size                (default: 256)
#   DEVICE           torch device              (default: auto — uses GPU if available)
#   VENV_DIR         virtualenv path           (default: .venv)
set -euo pipefail

PHASE="${1:-all}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-runs/final/turn_train.jsonl}"
DEV_MANIFEST="${DEV_MANIFEST:-runs/final/turn_dev.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-runs/final/turn_test.jsonl}"
VW_MANIFEST="${VW_MANIFEST:-runs/final/voiceworld_real.jsonl}"
ABLATION_OUT="${ABLATION_OUT:-runs/ablations}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-auto}"
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
# train_model  <out_dir>  <model>  <feature_source>  <seed>  [extra args...]
# ---------------------------------------------------------------------------
train_model() {
  local out_dir="$1" model="$2" feature_source="$3" seed="$4"
  shift 4
  local extra=("$@")
  mkdir -p "${out_dir}"
  run stable-asr train-turn \
    --dataset "${TRAIN_MANIFEST}" \
    --dev-dataset "${DEV_MANIFEST}" \
    --output-dir "${out_dir}" \
    --model "${model}" \
    --feature-source "${feature_source}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --json \
    "${extra[@]}"
}

# ---------------------------------------------------------------------------
# eval_model  <out_dir>  <model_label>
#   Writes compare-turn and eval-scenario JSON reports into <out_dir>/reports/
# ---------------------------------------------------------------------------
eval_model() {
  local out_dir="$1" label="$2"
  local checkpoint="${out_dir}/checkpoint.pt"
  local report_dir="${out_dir}/reports"
  mkdir -p "${report_dir}"
  [[ -f "${checkpoint}" ]] || { echo "SKIP eval (no checkpoint): ${checkpoint}"; return; }

  run stable-asr compare-turn \
    --dataset "${TEST_MANIFEST}" \
    --baseline rule_endpoint \
    --baseline vad_pause \
    --checkpoint "${label}=${checkpoint}" \
    --report "${report_dir}/compare_turn.md" \
    --json-output "${report_dir}/compare_turn.json"

  if [[ -f "${VW_MANIFEST}" ]]; then
    run stable-asr eval-scenario \
      --dataset "${VW_MANIFEST}" \
      --checkpoint "${checkpoint}" \
      --seed 0 \
      --report "${report_dir}/scenarios.md" \
      --json-output "${report_dir}/scenarios.json"
  fi
}

# ---------------------------------------------------------------------------
# FEATURE ABLATION
# Goal: quantify how much each metadata signal contributes.
# Ablation variants defined in stable_asr/train/features.py _ABLATION_MASKS.
# ---------------------------------------------------------------------------
run_feature_ablation() {
  local base="${ABLATION_OUT}/feature"
  echo "=== Feature ablation ==="

  declare -A VARIANTS=(
    ["metadata_full"]="metadata"
    ["metadata_no_duration"]="metadata_no_duration"
    ["metadata_no_pause"]="metadata_no_pause"
    ["metadata_no_duration_no_pause"]="metadata_no_duration_no_pause"
    ["metadata_content_only"]="metadata_content_only"
    ["audio_v1"]="audio_v1"
    ["audio_seq_micro"]="audio_seq"
  )
  declare -A VARIANT_MODELS=(
    ["metadata_full"]="nanoturn_nano"
    ["metadata_no_duration"]="nanoturn_nano"
    ["metadata_no_pause"]="nanoturn_nano"
    ["metadata_no_duration_no_pause"]="nanoturn_nano"
    ["metadata_content_only"]="nanoturn_nano"
    ["audio_v1"]="nanoturn_nano_v1"
    ["audio_seq_micro"]="nanoturn_micro"
  )

  local seed=0
  for name in "${!VARIANTS[@]}"; do
    local fsrc="${VARIANTS[$name]}"
    local model="${VARIANT_MODELS[$name]}"
    local out="${base}/${name}/seed_${seed}"
    echo "--- Feature ablation: ${name} (model=${model}, feature=${fsrc}) ---"
    train_model "${out}" "${model}" "${fsrc}" "${seed}"
    eval_model "${out}" "${name}"
  done
}

# ---------------------------------------------------------------------------
# ARCHITECTURE ABLATION
# Goal: compare all 5 NanoTurn variants on their natural feature source.
# ---------------------------------------------------------------------------
run_arch_ablation() {
  local base="${ABLATION_OUT}/arch"
  echo "=== Architecture ablation ==="

  declare -A ARCH_CONFIGS=(
    ["nanoturn_pico"]="nanoturn_pico metadata"
    ["nanoturn_nano"]="nanoturn_nano metadata"
    ["nanoturn_pico_v1"]="nanoturn_pico_v1 audio_v1"
    ["nanoturn_nano_v1"]="nanoturn_nano_v1 audio_v1"
    ["nanoturn_micro"]="nanoturn_micro audio_seq"
  )

  local seed=0
  for name in "${!ARCH_CONFIGS[@]}"; do
    read -r model fsrc <<< "${ARCH_CONFIGS[$name]}"
    local out="${base}/${name}/seed_${seed}"
    echo "--- Arch ablation: ${name} ---"
    train_model "${out}" "${model}" "${fsrc}" "${seed}"
    eval_model "${out}" "${name}"
  done
}

# ---------------------------------------------------------------------------
# DATA-SCALE ABLATION
# Goal: learning curves — does more data help? Uses nanoturn_nano/metadata.
# ---------------------------------------------------------------------------
run_scale_ablation() {
  local base="${ABLATION_OUT}/scale"
  echo "=== Data-scale ablation ==="

  # Subset fractions: generate smaller manifests with stable-asr bootstrap-turn-data
  # or use --validation-split to proxy scale via limited train rows.
  # We use a simple Python snippet to downsample the manifest.
  local model="nanoturn_nano"
  local fsrc="metadata"
  local seed=0

  for fraction in 0.10 0.25 0.50 1.00; do
    local label="frac_$(echo "${fraction}" | tr '.' '_')"
    local out="${base}/${label}/seed_${seed}"
    mkdir -p "${out}"
    echo "--- Scale ablation: ${fraction} of train data ---"

    # Downsample manifest
    local subset_manifest="${out}/train_subset.jsonl"
    python3 - "${TRAIN_MANIFEST}" "${fraction}" "${seed}" "${subset_manifest}" <<'PYEOF'
import sys, json, random, pathlib
src, frac, seed, dst = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
records = pathlib.Path(src).read_text().splitlines()
random.seed(seed)
n = max(1, int(len(records) * frac))
subset = random.sample(records, n)
pathlib.Path(dst).write_text("\n".join(subset) + "\n")
print(f"Subset: {n}/{len(records)} records ({frac:.0%})")
PYEOF

    mkdir -p "${out}"
    run stable-asr train-turn \
      --dataset "${subset_manifest}" \
      --dev-dataset "${DEV_MANIFEST}" \
      --output-dir "${out}" \
      --model "${model}" \
      --feature-source "${fsrc}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --json
    eval_model "${out}" "scale_${label}"
  done
}

# ---------------------------------------------------------------------------
# MULTI-SEED REPRODUCIBILITY
# Goal: verify variance across seeds; report mean ± std ACC.
# ---------------------------------------------------------------------------
run_seed_ablation() {
  local base="${ABLATION_OUT}/seed"
  echo "=== Multi-seed reproducibility ==="

  declare -A SEED_CONFIGS=(
    ["nanoturn_nano_metadata"]="nanoturn_nano metadata"
    ["nanoturn_nano_v1_audio_v1"]="nanoturn_nano_v1 audio_v1"
    ["nanoturn_micro_audio_seq"]="nanoturn_micro audio_seq"
  )

  for name in "${!SEED_CONFIGS[@]}"; do
    read -r model fsrc <<< "${SEED_CONFIGS[$name]}"
    for seed in ${SEEDS}; do
      local out="${base}/${name}/seed_${seed}"
      echo "--- Seed ablation: ${name} seed=${seed} ---"
      train_model "${out}" "${model}" "${fsrc}" "${seed}"
      eval_model "${out}" "${name}_seed${seed}"
    done
  done
}

# ---------------------------------------------------------------------------
# SUMMARISE: collect all metrics.json and print a table
# ---------------------------------------------------------------------------
run_summarise() {
  echo "=== Ablation summary ==="
  python3 - "${ABLATION_OUT}" <<'PYEOF'
import sys, json, pathlib

root = pathlib.Path(sys.argv[1])
rows = []
for metrics_file in sorted(root.rglob("metrics.json")):
    try:
        m = json.loads(metrics_file.read_text())
    except Exception:
        continue
    rel = metrics_file.parent.relative_to(root)
    acc = m.get("test_acc", m.get("val_acc", m.get("accuracy", "?")))
    f1  = m.get("test_f1",  m.get("val_f1",  m.get("f1", "?")))
    rows.append((str(rel), acc, f1))

if not rows:
    print("No metrics.json files found under", root)
    sys.exit(0)

col_w = max(len(r[0]) for r in rows) + 2
header = f"{'path':<{col_w}}  {'acc':>8}  {'f1':>8}"
print(header)
print("-" * len(header))
for path, acc, f1 in rows:
    acc_s = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
    f1_s  = f"{f1:.4f}"  if isinstance(f1,  float) else str(f1)
    print(f"{path:<{col_w}}  {acc_s:>8}  {f1_s:>8}")
PYEOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
activate_venv

mkdir -p "${ABLATION_OUT}"

case "${PHASE}" in
  all)
    run_feature_ablation
    run_arch_ablation
    run_scale_ablation
    run_seed_ablation
    run_summarise
    ;;
  feature)    run_feature_ablation ;;
  arch)       run_arch_ablation ;;
  scale)      run_scale_ablation ;;
  seed)       run_seed_ablation ;;
  summarise)  run_summarise ;;
  *)
    cat >&2 <<'EOF'
Usage: scripts/run_ablations.sh <phase>

Phases:
  all        run every ablation group
  feature    feature masking ablation (duration/pause/audio-only)
  arch       architecture ablation (all 5 NanoTurn variants)
  scale      data-scale ablation (10% / 25% / 50% / 100% of train)
  seed       multi-seed reproducibility (seeds 0,1,2 for key models)
  summarise  print ACC/F1 table from all completed metrics.json files

Key env vars:
  TRAIN_MANIFEST  DEV_MANIFEST  TEST_MANIFEST  VW_MANIFEST
  ABLATION_OUT    SEEDS         EPOCHS         BATCH_SIZE  DEVICE
EOF
    exit 2
    ;;
esac
