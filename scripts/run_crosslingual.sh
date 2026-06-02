#!/usr/bin/env bash
# Cross-lingual zero-shot and few-shot turn-taking evaluation.
#
# Direction B: train on English (AMI / Fisher), test on Mandarin
# (MagicData-RAMC / AISHELL-4) without any Mandarin training data.
#
# Usage:
#   scripts/run_crosslingual.sh <phase>
#
# Phases:
#   prepare     build turn manifests from downloaded dialogue corpora
#   train-en    train NanoTurn on English (AMI IHM)
#   eval-zh     zero-shot eval on Mandarin (MagicData-RAMC)
#   finetune-zh few-shot fine-tune on 5% / 10% of Mandarin train data
#   all         run all phases in order
#   report      print cross-lingual results table
#
# Environment variables:
#   DATA_ROOT      base directory for datasets (default: /home/li/stable-asr/data/dialogue)
#   CROSSLINGUAL_OUT  output directory          (default: runs/crosslingual)
#   DEVICE         torch device                 (default: auto)
#   EPOCHS         training epochs              (default: 30)
#   SEEDS          space-separated seeds        (default: "0 1 2")
#   VENV_DIR       virtualenv path              (default: .venv)
set -euo pipefail

PHASE="${1:-all}"
DATA_ROOT="${DATA_ROOT:-/home/li/stable-asr/data/dialogue}"
OUT="${CROSSLINGUAL_OUT:-runs/crosslingual}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-0 1 2}"
VENV_DIR="${VENV_DIR:-.venv}"

run() { printf '\n\033[1;34m[%s] + %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; "$@"; }
warn() { printf '\033[1;33m⚠  %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m✓  %s\033[0m\n' "$*"; }

activate_venv() {
  [[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"
}

# ── Phase 1: build turn manifests from dialogue corpora ───────────────────
# The raw AMI / MagicData JSONL from download_datasets.sh contain ASR
# segments, not turn-taking labels.  We convert them with bootstrap-turn-data
# using a configurable incomplete_ratio range (avoid the fixed-65% shortcut).
prepare() {
  echo "=== Preparing cross-lingual turn manifests ==="

  local ami_jsonl="${DATA_ROOT}/ami/ami_train.jsonl"
  local ami_test_jsonl="${DATA_ROOT}/ami/ami_test.jsonl"
  local ramc_jsonl="${DATA_ROOT}/magicdata_ramc/magicdata_ramc_all.jsonl"

  if [[ ! -f "${ami_jsonl}" ]]; then
    warn "AMI manifest not found at ${ami_jsonl}. Run download_datasets.sh ami first."
  fi
  if [[ ! -f "${ramc_jsonl}" ]]; then
    warn "MagicData-RAMC manifest not found at ${ramc_jsonl}. Run download_datasets.sh magicdata first."
  fi

  mkdir -p "${OUT}/manifests"

  # Build English turn manifest from AMI
  # Uses randomised incomplete_ratio [0.4, 0.85] to avoid duration shortcut
  if [[ -f "${ami_jsonl}" ]]; then
    run python3 - "${ami_jsonl}" "${OUT}/manifests" <<'PYEOF'
import sys, json, random, pathlib

src_path = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
random.seed(42)

records = [json.loads(l) for l in src_path.read_text().splitlines() if l.strip()]
print(f"AMI source records: {len(records)}")

turn_records = []
for rec in records:
    duration = rec.get("end", 0.0) - rec.get("start", 0.0)
    if duration < 0.5:
        continue  # too short

    # Complete sample: full utterance
    cr = dict(rec)
    cr["id"] = rec["id"] + "__complete"
    cr["turn_label"] = "complete"
    cr["action_label"] = "take_turn"
    cr["scenario"] = "ami_complete"
    cr.setdefault("metadata", {})
    cr["metadata"]["pause_ms"] = 900
    cr["metadata"]["vad_pause_ms"] = 900
    cr["metadata"]["duration_ms"] = round(duration * 1000.0, 1)
    cr["metadata"]["strategy"] = "ami_complete"
    turn_records.append(cr)

    # Incomplete sample: RANDOM ratio [0.4, 0.85] — avoids fixed-ratio shortcut
    ratio = random.uniform(0.40, 0.85)
    trunc_end = rec["start"] + duration * ratio
    ir = dict(rec)
    ir["id"] = rec["id"] + "__incomplete"
    ir["end"] = round(trunc_end, 6)
    ir["turn_label"] = "incomplete"
    ir["action_label"] = "keep_listening"
    ir["scenario"] = "ami_incomplete"
    ir["text"] = None
    ir.setdefault("metadata", {})
    ir["metadata"]["pause_ms"] = 150
    ir["metadata"]["vad_pause_ms"] = 150
    ir["metadata"]["duration_ms"] = round(duration * ratio * 1000.0, 1)
    ir["metadata"]["truncation_ratio"] = round(ratio, 4)
    ir["metadata"]["strategy"] = "ami_incomplete_random_ratio"
    turn_records.append(ir)

# Shuffle and split 80/10/10 by meeting_id (speaker-independent)
meeting_ids = list({r.get("metadata", {}).get("meeting_id", r["id"][:8]) for r in turn_records})
random.shuffle(meeting_ids)
n = len(meeting_ids)
train_ids = set(meeting_ids[:int(n * 0.8)])
dev_ids   = set(meeting_ids[int(n * 0.8):int(n * 0.9)])

train = [r for r in turn_records if r.get("metadata", {}).get("meeting_id", r["id"][:8]) in train_ids]
dev   = [r for r in turn_records if r.get("metadata", {}).get("meeting_id", r["id"][:8]) in dev_ids]
test  = [r for r in turn_records if r.get("metadata", {}).get("meeting_id", r["id"][:8]) not in (train_ids | dev_ids)]

for split, rows in [("train", train), ("dev", dev), ("test", test)]:
    out = out_dir / f"ami_turn_{split}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"  {split}: {len(rows)} records → {out}")
PYEOF
    ok "AMI turn manifests ready"
  fi

  # Build Mandarin turn manifest from MagicData-RAMC
  if [[ -f "${ramc_jsonl}" ]]; then
    run python3 - "${ramc_jsonl}" "${OUT}/manifests" <<'PYEOF'
import sys, json, random, pathlib

src_path = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
random.seed(42)

records = [json.loads(l) for l in src_path.read_text().splitlines() if l.strip()]
print(f"MagicData-RAMC source records: {len(records)}")

turn_records = []
for rec in records:
    duration = rec.get("end", 0.0) - rec.get("start", 0.0)
    if duration < 0.5:
        continue

    cr = dict(rec)
    cr["id"] = rec["id"] + "__complete"
    cr["turn_label"] = "complete"
    cr["action_label"] = "take_turn"
    cr["scenario"] = "ramc_complete"
    cr.setdefault("metadata", {})
    cr["metadata"]["pause_ms"] = 900
    cr["metadata"]["vad_pause_ms"] = 900
    cr["metadata"]["duration_ms"] = round(duration * 1000.0, 1)
    turn_records.append(cr)

    ratio = random.uniform(0.40, 0.85)
    trunc_end = rec["start"] + duration * ratio
    ir = dict(rec)
    ir["id"] = rec["id"] + "__incomplete"
    ir["end"] = round(trunc_end, 6)
    ir["turn_label"] = "incomplete"
    ir["action_label"] = "keep_listening"
    ir["scenario"] = "ramc_incomplete"
    ir["text"] = None
    ir.setdefault("metadata", {})
    ir["metadata"]["pause_ms"] = 150
    ir["metadata"]["vad_pause_ms"] = 150
    ir["metadata"]["duration_ms"] = round(duration * ratio * 1000.0, 1)
    ir["metadata"]["truncation_ratio"] = round(ratio, 4)
    turn_records.append(ir)

# Split by session_id
session_ids = list({r.get("metadata", {}).get("session_id", r["id"][:12]) for r in turn_records})
random.shuffle(session_ids)
n = len(session_ids)
train_ids = set(session_ids[:int(n * 0.7)])
dev_ids   = set(session_ids[int(n * 0.7):int(n * 0.85)])

train = [r for r in turn_records if r.get("metadata", {}).get("session_id", r["id"][:12]) in train_ids]
dev   = [r for r in turn_records if r.get("metadata", {}).get("session_id", r["id"][:12]) in dev_ids]
test  = [r for r in turn_records if r.get("metadata", {}).get("session_id", r["id"][:12]) not in (train_ids | dev_ids)]

for split, rows in [("train", train), ("dev", dev), ("test", test)]:
    out = out_dir / f"ramc_turn_{split}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"  {split}: {len(rows)} records → {out}")
PYEOF
    ok "MagicData-RAMC turn manifests ready"
  fi
}

# ── Phase 2: train on English (AMI) ───────────────────────────────────────
train_en() {
  echo "=== Training on English (AMI) ==="
  local train="${OUT}/manifests/ami_turn_train.jsonl"
  local dev="${OUT}/manifests/ami_turn_dev.jsonl"

  if [[ ! -f "${train}" ]]; then
    warn "AMI train manifest not found. Run: scripts/run_crosslingual.sh prepare"
    return 1
  fi

  for seed in ${SEEDS}; do
    for model_fsrc in "nanoturn_nano:metadata" "nanoturn_nano_v1:logmel_v1" "nanoturn_micro:audio_seq"; do
      local model="${model_fsrc%%:*}"
      local fsrc="${model_fsrc##*:}"
      local out_dir="${OUT}/models/en_ami/${model}/seed_${seed}"
      echo "--- Train: ${model} feature=${fsrc} seed=${seed} ---"
      run stable-asr train-turn \
        --dataset "${train}" \
        --dev-dataset "${dev}" \
        --output-dir "${out_dir}" \
        --model "${model}" \
        --feature-source "${fsrc}" \
        --epochs "${EPOCHS}" \
        --seed "${seed}" \
        --device "${DEVICE}" \
        --json
    done
  done
  ok "English training complete"
}

# ── Phase 3: zero-shot eval on Mandarin ────────────────────────────────────
eval_zh() {
  echo "=== Zero-shot Mandarin evaluation ==="
  local zh_test="${OUT}/manifests/ramc_turn_test.jsonl"

  if [[ ! -f "${zh_test}" ]]; then
    warn "RAMC test manifest not found. Run: scripts/run_crosslingual.sh prepare"
    return 1
  fi

  mkdir -p "${OUT}/results/zero_shot"

  for model_dir in "${OUT}/models/en_ami"/*/seed_0; do
    [[ -d "${model_dir}" ]] || continue
    local ckpt="${model_dir}/checkpoint.pt"
    [[ -f "${ckpt}" ]] || continue
    local model_name
    model_name=$(basename "$(dirname "${model_dir}")")

    echo "--- Zero-shot eval: ${model_name} on RAMC ---"
    local report_dir="${OUT}/results/zero_shot/${model_name}"
    mkdir -p "${report_dir}"

    run stable-asr compare-turn \
      --dataset "${zh_test}" \
      --baseline rule_endpoint \
      --baseline vad_pause \
      --checkpoint "${model_name}_zeroshot=${ckpt}" \
      --report "${report_dir}/compare_turn.md" \
      --json-output "${report_dir}/compare_turn.json"
  done
  ok "Zero-shot Mandarin eval complete"
}

# ── Phase 4: few-shot fine-tuning on Mandarin ─────────────────────────────
finetune_zh() {
  echo "=== Few-shot Mandarin fine-tuning ==="
  local zh_train="${OUT}/manifests/ramc_turn_train.jsonl"
  local zh_dev="${OUT}/manifests/ramc_turn_dev.jsonl"
  local zh_test="${OUT}/manifests/ramc_turn_test.jsonl"

  if [[ ! -f "${zh_train}" ]]; then
    warn "RAMC train manifest not found. Run prepare first."
    return 1
  fi

  for fraction in 0.05 0.10 0.25; do
    local label="frac_$(printf '%.0f' "$(echo "${fraction} * 100" | bc)")pct"
    for model_name in nanoturn_nano nanoturn_nano_v1; do
      local en_ckpt="${OUT}/models/en_ami/${model_name}/seed_0/checkpoint.pt"
      [[ -f "${en_ckpt}" ]] || { warn "English checkpoint missing: ${en_ckpt}"; continue; }

      local out_dir="${OUT}/models/zh_finetune/${model_name}/${label}"
      mkdir -p "${out_dir}"

      # Subsample Mandarin train data
      local subset_manifest="${out_dir}/ramc_train_subset.jsonl"
      python3 - "${zh_train}" "${fraction}" 42 "${subset_manifest}" <<'PYEOF'
import sys, json, random, pathlib
src, frac, seed, dst = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
records = pathlib.Path(src).read_text().splitlines()
random.seed(seed)
n = max(2, int(len(records) * frac))
subset = random.sample(records, n)
pathlib.Path(dst).write_text("\n".join(subset) + "\n")
print(f"Subset: {n}/{len(records)} records ({frac:.0%})")
PYEOF

      local fsrc="metadata"
      [[ "${model_name}" == *"v1"* ]] && fsrc="logmel_v1"

      echo "--- Fine-tune: ${model_name} ${label} zh data ---"
      run stable-asr train-turn \
        --dataset "${subset_manifest}" \
        --dev-dataset "${zh_dev}" \
        --output-dir "${out_dir}" \
        --model "${model_name}" \
        --feature-source "${fsrc}" \
        --resume-from "${en_ckpt}" \
        --epochs 10 \
        --lr 1e-4 \
        --seed 0 \
        --device "${DEVICE}" \
        --json

      # Evaluate fine-tuned model on Mandarin test
      local report_dir="${OUT}/results/finetune/${model_name}/${label}"
      mkdir -p "${report_dir}"
      run stable-asr compare-turn \
        --dataset "${zh_test}" \
        --baseline vad_pause \
        --checkpoint "${model_name}_ft_${label}=${out_dir}/checkpoint.pt" \
        --report "${report_dir}/compare_turn.md" \
        --json-output "${report_dir}/compare_turn.json"
    done
  done
  ok "Few-shot fine-tuning complete"
}

# ── Phase 5: report table ──────────────────────────────────────────────────
report() {
  echo "=== Cross-lingual results table ==="
  python3 - "${OUT}" <<'PYEOF'
import sys, json, pathlib

root = pathlib.Path(sys.argv[1])
rows = []

for jf in sorted(root.rglob("compare_turn.json")):
    try:
        data = json.loads(jf.read_text())
    except Exception:
        continue
    rel = jf.parent.relative_to(root / "results")
    # Extract per-system metrics
    systems = data if isinstance(data, list) else data.get("rows", [])
    for sys_row in systems:
        name = sys_row.get("system") or sys_row.get("name") or "?"
        acc = sys_row.get("accuracy") or (sys_row.get("classification") or {}).get("accuracy", "?")
        f1  = sys_row.get("macro_f1") or (sys_row.get("classification") or {}).get("macro_f1", "?")
        rows.append((str(rel), name, acc, f1))

if not rows:
    print("No results found yet. Run eval phases first.")
    sys.exit(0)

col = max(len(r[0]) for r in rows)
nc  = max(len(r[1]) for r in rows)
print(f"\n{'condition':<{col}}  {'system':<{nc}}  {'acc':>8}  {'macro_f1':>8}")
print("-" * (col + nc + 24))
for cond, sname, acc, f1 in rows:
    acc_s = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
    f1_s  = f"{f1:.4f}"  if isinstance(f1, float)  else str(f1)
    print(f"{cond:<{col}}  {sname:<{nc}}  {acc_s:>8}  {f1_s:>8}")
PYEOF
}

# ── Main ───────────────────────────────────────────────────────────────────
activate_venv
mkdir -p "${OUT}"

case "${PHASE}" in
  prepare)     prepare ;;
  train-en)    train_en ;;
  eval-zh)     eval_zh ;;
  finetune-zh) finetune_zh ;;
  report)      report ;;
  all)
    prepare
    train_en
    eval_zh
    finetune_zh
    report
    ;;
  *)
    cat >&2 <<'EOF'
Usage: scripts/run_crosslingual.sh <phase>

Phases:
  prepare      build turn manifests from AMI + MagicData-RAMC
               (uses randomised truncation ratio to avoid duration shortcut)
  train-en     train nanoturn_nano / nano_v1 / micro on AMI English
  eval-zh      zero-shot eval on MagicData-RAMC Mandarin (no Mandarin training)
  finetune-zh  few-shot fine-tune on 5% / 10% / 25% of Mandarin data
  report       print results table

  all          run all phases in order

Key env vars:
  DATA_ROOT  CROSSLINGUAL_OUT  DEVICE  EPOCHS  SEEDS
EOF
    exit 2
    ;;
esac
