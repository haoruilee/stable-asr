#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PHASE="${1:-status}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/large_data}"
DATA_ROOT="${DATA_ROOT:-data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"
SEED="${SEED:-0}"
WINDOW_SEC="${WINDOW_SEC:-2.0}"
INCOMPLETE_RATIO="${INCOMPLETE_RATIO:-0.65}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
DEV_RATIO="${DEV_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
TURN_GROUP_BY="${TURN_GROUP_BY:-metadata.asr_record_id}"
SAMPLE_COUNT="${SAMPLE_COUNT:-10000}"
CORRECTNESS_SAMPLE_COUNT="${CORRECTNESS_SAMPLE_COUNT:-10000}"
MAX_RECORDS="${MAX_RECORDS:-}"
REQUIRE_CORPORA="${REQUIRE_CORPORA:-0}"
REQUIRE_EXTERNAL="${REQUIRE_EXTERNAL:-0}"

LIBRISPEECH_DIR="${LIBRISPEECH_DIR:-${DATA_ROOT}/librispeech/LibriSpeech}"
LIBRISPEECH_SPLIT="${LIBRISPEECH_SPLIT:-}"
AISHELL1_DIR="${AISHELL1_DIR:-${DATA_ROOT}/aishell1/data_aishell}"
AISHELL1_SPLIT="${AISHELL1_SPLIT:-}"
WENETSPEECH_DIR="${WENETSPEECH_DIR:-${DATA_ROOT}/wenetspeech/WenetSpeech}"
WENETSPEECH_SPLIT="${WENETSPEECH_SPLIT:-}"
COMMON_VOICE_DIR="${COMMON_VOICE_DIR:-${DATA_ROOT}/common_voice/en}"
COMMON_VOICE_SPLIT="${COMMON_VOICE_SPLIT:-}"

ASR_METADATA="${ASR_METADATA:-}"
ASR_AUDIO_ROOT="${ASR_AUDIO_ROOT:-}"
ASR_LANGUAGE="${ASR_LANGUAGE:-unknown}"
ASR_SOURCE="${ASR_SOURCE:-custom_asr}"
ASR_SPLIT="${ASR_SPLIT:-}"
ASR_ID_FIELD="${ASR_ID_FIELD:-}"
ASR_AUDIO_FIELD="${ASR_AUDIO_FIELD:-}"
ASR_TEXT_FIELD="${ASR_TEXT_FIELD:-}"
ASR_DURATION_FIELD="${ASR_DURATION_FIELD:-}"
ASR_SPEAKER_FIELD="${ASR_SPEAKER_FIELD:-}"

VOICEWORLD_METADATA="${VOICEWORLD_METADATA:-${DATA_ROOT}/voiceworld/metadata.tsv}"
VOICEWORLD_AUDIO_ROOT="${VOICEWORLD_AUDIO_ROOT:-${DATA_ROOT}/voiceworld/audio}"
VOICEWORLD_LANGUAGE="${VOICEWORLD_LANGUAGE:-zh}"
VOICEWORLD_SOURCE="${VOICEWORLD_SOURCE:-voiceworld_real}"

EASYTURN_INPUT="${EASYTURN_INPUT:-}"
FULL_DUPLEX_BENCH_INPUT="${FULL_DUPLEX_BENCH_INPUT:-}"
SMART_TURN_INPUT="${SMART_TURN_INPUT:-}"
EXTERNAL_LANGUAGE="${EXTERNAL_LANGUAGE:-unknown}"

ASR_MANIFEST_LIST="${OUTPUT_DIR}/asr_manifest_paths.txt"
COMBINED_ASR_MANIFEST="${OUTPUT_DIR}/asr/asr_manifest.jsonl"
WEAK_TURN_MANIFEST="${OUTPUT_DIR}/turn/turn_manifest.jsonl"
TURN_SPLIT_DIR="${OUTPUT_DIR}/turn/splits"
REPORT_DIR="${OUTPUT_DIR}/reports"

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

setup_data() {
  setup_base
  run python -m pip install -e ".[lance]"
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

ensure_dirs() {
  mkdir -p "${OUTPUT_DIR}/asr" "${OUTPUT_DIR}/turn" "${OUTPUT_DIR}/external" "${OUTPUT_DIR}/voiceworld" "${REPORT_DIR}"
}

write_status() {
  ensure_dirs
  {
    echo "# Stable-ASR Large-Scale Data Status"
    echo
    echo "- output_dir: \`${OUTPUT_DIR}\`"
    echo "- data_root: \`${DATA_ROOT}\`"
    echo "- sample_rate: \`${SAMPLE_RATE}\`"
    echo "- require_corpora: \`${REQUIRE_CORPORA}\`"
    echo "- require_external: \`${REQUIRE_EXTERNAL}\`"
    echo
    echo "## Corpus Paths"
    printf -- "- LibriSpeech: \`%s\` %s\n" "${LIBRISPEECH_DIR}" "$(exists_label "${LIBRISPEECH_DIR}")"
    printf -- "- AISHELL-1: \`%s\` %s\n" "${AISHELL1_DIR}" "$(exists_label "${AISHELL1_DIR}")"
    printf -- "- WenetSpeech: \`%s\` %s\n" "${WENETSPEECH_DIR}" "$(exists_label "${WENETSPEECH_DIR}")"
    printf -- "- Common Voice: \`%s\` %s\n" "${COMMON_VOICE_DIR}" "$(exists_label "${COMMON_VOICE_DIR}")"
    if [[ -n "${ASR_METADATA}" ]]; then
      printf -- "- Custom ASR metadata: \`%s\` %s\n" "${ASR_METADATA}" "$(exists_label "${ASR_METADATA}")"
    fi
    echo
    echo "## Optional Turn/Scenario Inputs"
    printf -- "- VoiceWorld metadata: \`%s\` %s\n" "${VOICEWORLD_METADATA}" "$(exists_label "${VOICEWORLD_METADATA}")"
    print_optional_file "EasyTurn" "${EASYTURN_INPUT}"
    print_optional_file "Full-Duplex-Bench" "${FULL_DUPLEX_BENCH_INPUT}"
    print_optional_file "Smart Turn" "${SMART_TURN_INPUT}"
    echo
    echo "## Main Outputs"
    print_optional_file "combined ASR manifest" "${COMBINED_ASR_MANIFEST}"
    print_optional_file "weak turn manifest" "${WEAK_TURN_MANIFEST}"
    print_optional_file "turn train split" "${TURN_SPLIT_DIR}/turn_train.jsonl"
    print_optional_file "turn dev split" "${TURN_SPLIT_DIR}/turn_dev.jsonl"
    print_optional_file "turn test split" "${TURN_SPLIT_DIR}/turn_test.jsonl"
  } > "${REPORT_DIR}/LARGE_DATA_STATUS.md"
  cat "${REPORT_DIR}/LARGE_DATA_STATUS.md"
}

exists_label() {
  if [[ -e "$1" ]]; then
    echo "present"
  else
    echo "missing"
  fi
}

print_optional_file() {
  local label="$1"
  local path="$2"
  if [[ -n "${path}" ]]; then
    printf -- "- %s: \`%s\` %s\n" "${label}" "${path}" "$(exists_label "${path}")"
  else
    printf -- "- %s: not configured\n" "${label}"
  fi
}

prepare_acquisition() {
  ensure_dirs
  stable_asr data-sources --output "${REPORT_DIR}/DATA_SOURCES.md"
  stable_asr asr-collections --format acquisition-markdown --output "${REPORT_DIR}/ASR_COLLECTION_ACQUISITION.md"
  stable_asr turn-collections --format acquisition-markdown --output "${REPORT_DIR}/TURN_COLLECTION_ACQUISITION.md"
  stable_asr asr-collections --format source-manifest --output "${OUTPUT_DIR}/ASR_COLLECTION_SOURCE_MANIFEST.json"
  stable_asr turn-collections --format source-manifest --output "${OUTPUT_DIR}/TURN_COLLECTION_SOURCE_MANIFEST.json"
  stable_asr final-acquisition-pack --output-dir "${OUTPUT_DIR}/final_acquisition_pack"
}

prepare_manifests() {
  ensure_dirs
  : > "${ASR_MANIFEST_LIST}"
  prepare_public_corpus "librispeech" "${LIBRISPEECH_DIR}" "${OUTPUT_DIR}/asr/librispeech/asr_manifest.jsonl" "${LIBRISPEECH_SPLIT}"
  prepare_public_corpus "aishell1" "${AISHELL1_DIR}" "${OUTPUT_DIR}/asr/aishell1/asr_manifest.jsonl" "${AISHELL1_SPLIT}"
  prepare_public_corpus "wenetspeech" "${WENETSPEECH_DIR}" "${OUTPUT_DIR}/asr/wenetspeech/asr_manifest.jsonl" "${WENETSPEECH_SPLIT}"
  prepare_public_corpus "common_voice" "${COMMON_VOICE_DIR}" "${OUTPUT_DIR}/asr/common_voice/asr_manifest.jsonl" "${COMMON_VOICE_SPLIT}"
  prepare_custom_asr_metadata
  combine_asr_manifests
}

prepare_public_corpus() {
  local corpus="$1"
  local input_dir="$2"
  local output="$3"
  local split="$4"

  if [[ ! -d "${input_dir}" ]]; then
    if [[ "${REQUIRE_CORPORA}" == "1" ]]; then
      echo "ERROR: required corpus directory is missing: ${input_dir}" >&2
      exit 1
    fi
    printf '\nSkipping %s: missing directory %s\n' "${corpus}" "${input_dir}"
    return 0
  fi

  mkdir -p "$(dirname "${output}")"
  local args=(prepare-public-asr --corpus "${corpus}" --input-dir "${input_dir}" --output "${output}" --sample-rate "${SAMPLE_RATE}")
  if [[ -n "${split}" ]]; then
    args+=(--split "${split}")
  fi
  stable_asr "${args[@]}"
  stable_asr validate-asr-manifest "${output}"
  stable_asr inspect-asr-manifest "${output}" > "${output}.summary.txt"
  printf '%s\n' "${output}" >> "${ASR_MANIFEST_LIST}"
}

prepare_custom_asr_metadata() {
  if [[ -z "${ASR_METADATA}" ]]; then
    return 0
  fi
  if [[ ! -f "${ASR_METADATA}" ]]; then
    echo "ERROR: ASR_METADATA does not exist: ${ASR_METADATA}" >&2
    exit 1
  fi

  local output="${OUTPUT_DIR}/asr/custom/asr_manifest.jsonl"
  mkdir -p "$(dirname "${output}")"
  local args=(
    prepare-asr-manifest
    --input "${ASR_METADATA}"
    --output "${output}"
    --sample-rate "${SAMPLE_RATE}"
    --language "${ASR_LANGUAGE}"
    --source "${ASR_SOURCE}"
  )
  append_arg_if_set args --audio-root "${ASR_AUDIO_ROOT}"
  append_arg_if_set args --split "${ASR_SPLIT}"
  append_arg_if_set args --id-field "${ASR_ID_FIELD}"
  append_arg_if_set args --audio-field "${ASR_AUDIO_FIELD}"
  append_arg_if_set args --text-field "${ASR_TEXT_FIELD}"
  append_arg_if_set args --duration-field "${ASR_DURATION_FIELD}"
  append_arg_if_set args --speaker-field "${ASR_SPEAKER_FIELD}"
  stable_asr "${args[@]}"
  stable_asr validate-asr-manifest "${output}"
  stable_asr inspect-asr-manifest "${output}" > "${output}.summary.txt"
  printf '%s\n' "${output}" >> "${ASR_MANIFEST_LIST}"
}

append_arg_if_set() {
  local -n target_array="$1"
  local flag="$2"
  local value="$3"
  if [[ -n "${value}" ]]; then
    target_array+=("${flag}" "${value}")
  fi
}

combine_asr_manifests() {
  if [[ ! -s "${ASR_MANIFEST_LIST}" ]]; then
    echo "ERROR: no ASR manifests were prepared. Set corpus paths or ASR_METADATA." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${COMBINED_ASR_MANIFEST}")"
  : > "${COMBINED_ASR_MANIFEST}"
  while IFS= read -r manifest; do
    [[ -z "${manifest}" ]] && continue
    cat "${manifest}" >> "${COMBINED_ASR_MANIFEST}"
  done < "${ASR_MANIFEST_LIST}"
  stable_asr validate-asr-manifest "${COMBINED_ASR_MANIFEST}"
  stable_asr inspect-asr-manifest "${COMBINED_ASR_MANIFEST}" > "${REPORT_DIR}/ASR_MANIFEST_SUMMARY.txt"
  printf '\ncombined_asr_manifest: %s\n' "${COMBINED_ASR_MANIFEST}"
}

prepare_turn_splits() {
  ensure_dirs
  if [[ ! -f "${COMBINED_ASR_MANIFEST}" ]]; then
    prepare_manifests
  fi
  stable_asr asr-to-turn \
    --input "${COMBINED_ASR_MANIFEST}" \
    --output "${WEAK_TURN_MANIFEST}" \
    --include-incomplete \
    --window-sec "${WINDOW_SEC}" \
    --incomplete-ratio "${INCOMPLETE_RATIO}" \
    --source "large_asr_weak_turn_v0"
  stable_asr validate-manifest "${WEAK_TURN_MANIFEST}"
  stable_asr inspect-manifest "${WEAK_TURN_MANIFEST}" > "${REPORT_DIR}/WEAK_TURN_SUMMARY.txt"
  stable_asr profile-turn-data \
    --dataset "${WEAK_TURN_MANIFEST}" \
    --report "${REPORT_DIR}/WEAK_TURN_PROFILE.md"
  stable_asr split-turn-data \
    --input "${WEAK_TURN_MANIFEST}" \
    --output-dir "${TURN_SPLIT_DIR}" \
    --train-ratio "${TRAIN_RATIO}" \
    --dev-ratio "${DEV_RATIO}" \
    --test-ratio "${TEST_RATIO}" \
    --group-by "${TURN_GROUP_BY}" \
    --seed "${SEED}"
  stable_asr audit-turn-splits \
    --train "${TURN_SPLIT_DIR}/turn_train.jsonl" \
    --dev "${TURN_SPLIT_DIR}/turn_dev.jsonl" \
    --test "${TURN_SPLIT_DIR}/turn_test.jsonl" \
    --report "${REPORT_DIR}/TURN_SPLIT_AUDIT.md"
}

prepare_voiceworld() {
  ensure_dirs
  if [[ ! -f "${VOICEWORLD_METADATA}" ]]; then
    printf '\nSkipping VoiceWorld: missing metadata %s\n' "${VOICEWORLD_METADATA}"
    return 0
  fi
  local args=(
    prepare-voiceworld
    --input "${VOICEWORLD_METADATA}"
    --output "${OUTPUT_DIR}/voiceworld/voiceworld_real.jsonl"
    --sample-rate "${SAMPLE_RATE}"
    --language "${VOICEWORLD_LANGUAGE}"
    --source "${VOICEWORLD_SOURCE}"
  )
  if [[ -d "${VOICEWORLD_AUDIO_ROOT}" ]]; then
    args+=(--audio-root "${VOICEWORLD_AUDIO_ROOT}")
  fi
  stable_asr "${args[@]}"
  stable_asr validate-manifest "${OUTPUT_DIR}/voiceworld/voiceworld_real.jsonl"
  stable_asr profile-turn-data \
    --dataset "${OUTPUT_DIR}/voiceworld/voiceworld_real.jsonl" \
    --require-all-turn-labels \
    --report "${REPORT_DIR}/VOICEWORLD_PROFILE.md"
}

prepare_external_turn_data() {
  ensure_dirs
  convert_external_if_configured "easyturn" "${EASYTURN_INPUT}" "${OUTPUT_DIR}/external/easyturn_turn.jsonl"
  convert_external_if_configured "full_duplex_bench" "${FULL_DUPLEX_BENCH_INPUT}" "${OUTPUT_DIR}/external/full_duplex_bench_turn.jsonl"
  convert_external_if_configured "smart_turn" "${SMART_TURN_INPUT}" "${OUTPUT_DIR}/external/smart_turn_turn.jsonl"
}

convert_external_if_configured() {
  local schema="$1"
  local input="$2"
  local output="$3"
  if [[ -z "${input}" ]]; then
    return 0
  fi
  if [[ ! -f "${input}" ]]; then
    if [[ "${REQUIRE_EXTERNAL}" == "1" ]]; then
      echo "ERROR: required external ${schema} input is missing: ${input}" >&2
      exit 1
    fi
    printf '\nSkipping external %s: missing file %s\n' "${schema}" "${input}"
    return 0
  fi
  stable_asr convert-external \
    --schema "${schema}" \
    --input "${input}" \
    --output "${output}" \
    --sample-rate "${SAMPLE_RATE}" \
    --language "${EXTERNAL_LANGUAGE}"
  stable_asr validate-manifest "${output}"
  stable_asr profile-turn-data \
    --dataset "${output}" \
    --report "${output}.profile.md"
}

data_layer() {
  ensure_dirs
  if [[ ! -f "${WEAK_TURN_MANIFEST}" ]]; then
    prepare_turn_splits
  fi
  local formats=(jsonl)
  if has_module pyarrow; then
    formats+=(parquet)
  fi
  if has_lance; then
    formats+=(lance)
  fi
  stable_asr benchmark-data \
    --dataset "${WEAK_TURN_MANIFEST}" \
    --output-dir "${OUTPUT_DIR}/data_bench" \
    --formats "${formats[@]}" \
    --sample-count "${SAMPLE_COUNT}" \
    --json-output "${REPORT_DIR}/data_benchmark.json"
}

cache_features() {
  ensure_dirs
  local default_cache_dataset="${OUTPUT_DIR}/voiceworld/voiceworld_real.jsonl"
  local dataset="${CACHE_DATASET:-${default_cache_dataset}}"
  local audio_root="${CACHE_AUDIO_ROOT:-${VOICEWORLD_AUDIO_ROOT}}"
  if [[ ! -f "${dataset}" ]]; then
    if [[ -z "${CACHE_DATASET:-}" && -f "${VOICEWORLD_METADATA}" ]]; then
      prepare_voiceworld
    fi
  fi
  if [[ ! -f "${dataset}" ]]; then
    echo "ERROR: cache dataset does not exist: ${dataset}" >&2
    echo "Set CACHE_DATASET to a turn manifest with local audio paths." >&2
    exit 1
  fi
  if ! has_module torch; then
    echo "ERROR: Torch is required for log-mel feature cache. Run setup-all or install stable-asr[train]." >&2
    exit 1
  fi
  local formats=(source_audio source_audio_file_cache)
  if has_module pyarrow; then
    formats+=(parquet)
  fi
  if has_lance; then
    formats+=(lance)
  fi
  local args=(
    benchmark-train-features
    --dataset "${dataset}"
    --output-dir "${OUTPUT_DIR}/feature_cache_bench"
    --formats "${formats[@]}"
    --sample-count "${SAMPLE_COUNT}"
    --correctness-sample-count "${CORRECTNESS_SAMPLE_COUNT}"
    --audio-root "${audio_root}"
    --json-output "${REPORT_DIR}/train_feature_benchmark.json"
  )
  if [[ -n "${MAX_RECORDS}" ]]; then
    args+=(--max-records "${MAX_RECORDS}")
  fi
  stable_asr "${args[@]}"
}

audit_outputs() {
  ensure_dirs
  stable_asr data-sources --output "${REPORT_DIR}/DATA_SOURCES.md"
  if [[ -f "${COMBINED_ASR_MANIFEST}" ]]; then
    stable_asr validate-asr-manifest "${COMBINED_ASR_MANIFEST}"
    stable_asr inspect-asr-manifest "${COMBINED_ASR_MANIFEST}"
    stable_asr audit-audio \
      --kind asr \
      --manifest "${COMBINED_ASR_MANIFEST}" \
      --report "${REPORT_DIR}/ASR_AUDIO_AUDIT.md"
  fi
  if [[ -f "${WEAK_TURN_MANIFEST}" ]]; then
    stable_asr validate-manifest "${WEAK_TURN_MANIFEST}"
    stable_asr profile-turn-data \
      --dataset "${WEAK_TURN_MANIFEST}" \
      --report "${REPORT_DIR}/WEAK_TURN_PROFILE.md"
  fi
  if [[ -f "${TURN_SPLIT_DIR}/turn_train.jsonl" ]]; then
    stable_asr audit-turn-splits \
      --train "${TURN_SPLIT_DIR}/turn_train.jsonl" \
      --dev "${TURN_SPLIT_DIR}/turn_dev.jsonl" \
      --test "${TURN_SPLIT_DIR}/turn_test.jsonl" \
      --report "${REPORT_DIR}/TURN_SPLIT_AUDIT.md"
  fi
  write_status
}

if [[ "${PHASE}" != "setup" && "${PHASE}" != "setup-data" && "${PHASE}" != "setup-all" ]]; then
  activate_venv_if_available
fi

case "${PHASE}" in
  setup) setup_base ;;
  setup-data) setup_data ;;
  setup-all) setup_all ;;
  status) write_status ;;
  acquisition) prepare_acquisition ;;
  manifests) prepare_manifests ;;
  turn) prepare_turn_splits ;;
  voiceworld) prepare_voiceworld ;;
  external) prepare_external_turn_data ;;
  data-layer) data_layer ;;
  cache) cache_features ;;
  audit) audit_outputs ;;
  all)
    prepare_acquisition
    prepare_manifests
    prepare_turn_splits
    prepare_voiceworld
    prepare_external_turn_data
    data_layer
    audit_outputs
    ;;
  full)
    prepare_acquisition
    prepare_manifests
    prepare_turn_splits
    prepare_voiceworld
    prepare_external_turn_data
    data_layer
    cache_features
    audit_outputs
    ;;
  *)
    cat >&2 <<'EOF'
Usage: scripts/prepare_large_scale_data.sh <phase>

Phases:
  setup       create .venv and install the base package
  setup-data  install data backends: Parquet and Lance
  setup-all   install all optional deps, including Torch and ONNX
  status      write a local data readiness report
  acquisition generate source, license, and collection work items
  manifests   normalize configured ASR corpora into one ASR manifest
  turn        derive weak complete/incomplete turn windows and audited splits
  voiceworld  normalize configured VoiceWorld turn/action scenario metadata
  external    convert EasyTurn, Full-Duplex-Bench, and Smart Turn JSONL inputs
  data-layer  run JSONL/Parquet/Lance manifest data-layer benchmarks
  cache       build and correctness-check log-mel feature cache benchmarks
  audit       validate manifests, audio paths, splits, and status
  all         acquisition + manifests + turn + voiceworld + external + data-layer + audit
  full        all + feature cache benchmark

Common environment variables:
  OUTPUT_DIR=runs/large_data
  DATA_ROOT=data
  LIBRISPEECH_DIR=/path/to/LibriSpeech
  AISHELL1_DIR=/path/to/data_aishell
  WENETSPEECH_DIR=/path/to/WenetSpeech
  COMMON_VOICE_DIR=/path/to/common_voice/locale
  ASR_METADATA=/path/to/custom.tsv
  ASR_AUDIO_ROOT=/path/to/audio/root
  EASYTURN_INPUT=/path/to/easyturn.jsonl
  FULL_DUPLEX_BENCH_INPUT=/path/to/full_duplex_bench.jsonl
  SMART_TURN_INPUT=/path/to/smart_turn.jsonl
  VOICEWORLD_METADATA=/path/to/metadata.tsv
  VOICEWORLD_AUDIO_ROOT=/path/to/audio
  REQUIRE_CORPORA=1
  REQUIRE_EXTERNAL=1
  SAMPLE_COUNT=10000
  CORRECTNESS_SAMPLE_COUNT=10000
  TURN_GROUP_BY=metadata.asr_record_id
EOF
    exit 2
    ;;
esac
