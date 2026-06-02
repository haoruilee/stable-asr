#!/usr/bin/env bash
# Download publicly available dialogue/turn-taking datasets.
#
# Usage:
#   scripts/download_datasets.sh <dataset> [<dataset> ...]
#   scripts/download_datasets.sh all
#   scripts/download_datasets.sh status
#
# Datasets (free, no application required):
#   ami          AMI Meeting Corpus (audio + NXT annotations) — ~17GB download, ~100GB extracted
#   icsi         ICSI Meeting Corpus (audio + annotations)    — ~8GB download,  ~40GB extracted
#   magicdata    MagicData-RAMC (Mandarin conversational)     — ~43GB download, ~60GB extracted
#   aishell4     AISHELL-4 (Mandarin meeting, 8-ch array)     — ~20GB download, ~30GB extracted
#
# NOT included here (require LDC license — apply at https://www.ldc.upenn.edu/):
#   Switchboard  LDC97S62  — standard corpus for VAP/TurnGPT/SmartTurn
#   Fisher       LDC2004S13 / LDC2005S13
#   CallHome     LDC97S42
#
# Environment variables:
#   DATA_ROOT    base directory for datasets  (default: /data/data)
#   JOBS         parallel download threads    (default: 4)
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/data}"
JOBS="${JOBS:-4}"

# ── helpers ────────────────────────────────────────────────────────────────

log()  { printf '\n\033[1;34m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

need() {
  for cmd in "$@"; do
    command -v "${cmd}" &>/dev/null || die "Required command not found: ${cmd}. Install with: sudo apt-get install ${cmd}"
  done
}

# Download with resume support and progress
fetch() {
  local url="$1" dest="$2"
  wget --continue --progress=bar:force --timeout=60 --tries=5 \
    --no-check-certificate -O "${dest}" "${url}"
}

# Parallel download using wget (for multi-part corpora)
fetch_parallel() {
  # $@: "url dest" pairs, newline separated via stdin or positional
  # Each arg is "URL DESTFILE" space-separated
  printf '%s\n' "$@" | xargs -P "${JOBS}" -I{} bash -c '
    pair="{}"
    url="${pair%% *}"
    dest="${pair#* }"
    wget --continue --progress=dot:giga --timeout=60 --tries=5 \
         --no-check-certificate -O "${dest}" "${url}" 2>&1 | tail -1
  '
}

verify_md5() {
  local file="$1" expected="$2"
  local actual
  actual=$(md5sum "${file}" | awk '{print $1}')
  if [[ "${actual}" == "${expected}" ]]; then
    ok "MD5 OK: $(basename "${file}")"
  else
    warn "MD5 MISMATCH: $(basename "${file}") expected=${expected} got=${actual}"
  fi
}

# ── AMI Meeting Corpus ─────────────────────────────────────────────────────
# Official page: https://groups.inf.ed.ac.uk/ami/corpus/
# Audio: Edinburgh mirror via amicorpus.org (multiple headset + array channels)
# Annotations: NXT-format XML (IPU, overlap, DA, backchannel, etc.)
# License: CC BY 4.0
# Size: ~17GB compressed audio (headset mix), ~100GB if all channels
# We download headset mix only (the channel used in most NLP research)
download_ami() {
  local dest="${DATA_ROOT}/ami"
  mkdir -p "${dest}"
  log "Downloading AMI Meeting Corpus → ${dest}"

  # AMI is most easily downloaded via HuggingFace datasets (audio + transcripts)
  # or directly from the Edinburgh OpenSLR/amicorpus mirrors.
  # HuggingFace path is the most reliable and includes auto-segmented splits.
  need python3

  log "AMI: downloading via HuggingFace datasets (IHM headset mix)"
  python3 - "${dest}" <<'PYEOF'
import sys, pathlib

dest = pathlib.Path(sys.argv[1])
dest.mkdir(parents=True, exist_ok=True)

try:
    from datasets import load_dataset, DownloadConfig
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "soundfile", "librosa"])
    from datasets import load_dataset

print("Loading AMI (IHM) train split — this will download ~14GB ...")
ds_train = load_dataset(
    "edinburghcstr/ami",
    "ihm",            # Individual HeadSet Microphone mix
    split="train",
    trust_remote_code=True,
)
print(f"  train: {len(ds_train)} segments")

print("Loading AMI (IHM) validation split ...")
ds_val = load_dataset("edinburghcstr/ami", "ihm", split="validation", trust_remote_code=True)
print(f"  validation: {len(ds_val)} segments")

print("Loading AMI (IHM) test split ...")
ds_test = load_dataset("edinburghcstr/ami", "ihm", split="test", trust_remote_code=True)
print(f"  test: {len(ds_test)} segments")

# Save as JSONL manifests compatible with stable-asr
import json

def save_manifest(ds, path):
    records = []
    for i, ex in enumerate(ds):
        records.append({
            "id": ex.get("meeting_id", f"ami_{i:06d}") + f"__{i:06d}",
            "text": ex.get("text", ""),
            "audio": ex["audio"]["path"] if isinstance(ex["audio"], dict) else "",
            "sample_rate": ex["audio"]["sampling_rate"] if isinstance(ex["audio"], dict) else 16000,
            "start": 0.0,
            "end": ex["audio"]["array"].shape[0] / ex["audio"]["sampling_rate"]
                   if isinstance(ex["audio"], dict) else 0.0,
            "language": "en",
            "source": "ami_ihm",
            "metadata": {
                "meeting_id": ex.get("meeting_id", ""),
                "speaker_id": ex.get("speaker_id", ""),
                "microphone_id": ex.get("microphone_id", ""),
                "segment_id": str(i),
            },
        })
    path = pathlib.Path(path)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    print(f"  Wrote {len(records)} records → {path}")

save_manifest(ds_train, dest / "ami_train.jsonl")
save_manifest(ds_val,   dest / "ami_dev.jsonl")
save_manifest(ds_test,  dest / "ami_test.jsonl")

# Also save HuggingFace dataset to disk for fast reloading
ds_train.save_to_disk(str(dest / "hf_cache" / "train"))
ds_val.save_to_disk(str(dest  / "hf_cache" / "validation"))
ds_test.save_to_disk(str(dest / "hf_cache" / "test"))
print("Done. AMI dataset saved.")
PYEOF

  ok "AMI download complete → ${dest}"
  echo "  Manifests: ${dest}/ami_{train,dev,test}.jsonl"
  echo "  HF cache:  ${dest}/hf_cache/"
}

# ── ICSI Meeting Corpus ────────────────────────────────────────────────────
# Official page: https://icsi.berkeley.edu/icsi/projects/2002/09/icsi-meeting-corpus/
# Free for research use; direct download from ICSI Berkeley
# Audio: .sph (Sphere) format, multiple channels
# Annotations: XML (transcript, overlap, IPU, MRDA dialogue acts)
# Size: ~8GB compressed
download_icsi() {
  local dest="${DATA_ROOT}/icsi"
  mkdir -p "${dest}"
  log "Downloading ICSI Meeting Corpus → ${dest}"

  need wget

  # Primary audio source: OpenSLR 101 (LDC LDC2004S02 open mirror)
  # OpenSLR hosts the ICSI audio that ICSI Berkeley makes free
  local OPENSLR_BASE="https://www.openslr.org/resources/101"

  log "ICSI: checking OpenSLR availability..."
  # The ICSI corpus on OpenSLR is split into multiple tar files
  # Try to list them; fall back to ICSI direct if OpenSLR doesn't have it
  if wget -q --spider "${OPENSLR_BASE}/ICSI_core_NXT.tar.gz" 2>/dev/null; then
    log "ICSI: downloading NXT annotations from OpenSLR"
    fetch "${OPENSLR_BASE}/ICSI_core_NXT.tar.gz" "${dest}/ICSI_core_NXT.tar.gz"
    tar -xzf "${dest}/ICSI_core_NXT.tar.gz" -C "${dest}/"
  else
    warn "OpenSLR ICSI not available; falling back to Edinburgh AMI-ICSI annotations mirror"
    # Edinburgh hosts ICSI NXT annotations (no audio)
    ANNOT_URL="https://groups.inf.ed.ac.uk/ami/icsi/ICSICorpusAnnotations.zip"
    fetch "${ANNOT_URL}" "${dest}/ICSICorpusAnnotations.zip"
    unzip -q "${dest}/ICSICorpusAnnotations.zip" -d "${dest}/"
  fi

  # ICSI audio direct from Berkeley (if available)
  # Note: the Berkeley download page requires a form, so we try the ICSI
  # NXT mirror which has audio in a single tarball
  local ICSI_AUDIO_URL="https://groups.inf.ed.ac.uk/ami/icsi/ICSI_audio_part1.tar.gz"
  if wget -q --spider "${ICSI_AUDIO_URL}" 2>/dev/null; then
    log "ICSI: downloading audio part 1 (~3GB)"
    fetch "${ICSI_AUDIO_URL}" "${dest}/ICSI_audio_part1.tar.gz"
    tar -xzf "${dest}/ICSI_audio_part1.tar.gz" -C "${dest}/"
    local ICSI_AUDIO_URL2="https://groups.inf.ed.ac.uk/ami/icsi/ICSI_audio_part2.tar.gz"
    if wget -q --spider "${ICSI_AUDIO_URL2}" 2>/dev/null; then
      log "ICSI: downloading audio part 2 (~3GB)"
      fetch "${ICSI_AUDIO_URL2}" "${dest}/ICSI_audio_part2.tar.gz"
      tar -xzf "${dest}/ICSI_audio_part2.tar.gz" -C "${dest}/"
    fi
  else
    warn "ICSI audio mirror not reachable. Try manually from:"
    warn "  https://icsi.berkeley.edu/icsi/projects/2002/09/icsi-meeting-corpus/downloading"
    warn "  Place downloaded tarballs in ${dest}/ and re-run with icsi_extract"
  fi

  # Generate manifest from whatever transcripts are available
  need python3
  python3 - "${dest}" <<'PYEOF'
import sys, json, pathlib, re

dest = pathlib.Path(sys.argv[1])
records = []

# Try to parse any .trans or NXT XML transcription files found
for xml_file in sorted(dest.rglob("*.mrt")):
    # Minimal .mrt (meeting transcript) parser
    meeting_id = xml_file.stem
    try:
        text = xml_file.read_text(errors="replace")
        # Segments look like: <Segment ... StartTime="X" EndTime="Y" ...>text</Segment>
        for m in re.finditer(
            r'<Segment[^>]+StartTime="([^"]+)"[^>]+EndTime="([^"]+)"[^>]*>([^<]*)</Segment>',
            text, re.DOTALL
        ):
            start, end, seg_text = float(m.group(1)), float(m.group(2)), m.group(3).strip()
            if end <= start or not seg_text:
                continue
            records.append({
                "id": f"{meeting_id}__{len(records):06d}",
                "text": seg_text,
                "audio": f"{meeting_id}.sph",
                "sample_rate": 16000,
                "start": round(start, 6),
                "end": round(end, 6),
                "language": "en",
                "source": "icsi",
                "metadata": {"meeting_id": meeting_id},
            })
    except Exception as e:
        print(f"  Warning: could not parse {xml_file}: {e}")

out = dest / "icsi_all.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
print(f"Wrote {len(records)} segments → {out}")
if not records:
    print("No segments found yet; audio/annotations may not be downloaded.")
PYEOF

  ok "ICSI download complete → ${dest}"
}

# ── MagicData-RAMC ────────────────────────────────────────────────────────
# OpenSLR 123: https://openslr.org/123/
# License: CC BY 4.0
# Size: ~43GB compressed, ~60GB extracted
# 180h, 351 speakers, spontaneous Mandarin conversations
download_magicdata() {
  local dest="${DATA_ROOT}/magicdata_ramc"
  mkdir -p "${dest}"
  log "Downloading MagicData-RAMC → ${dest}"

  need wget

  # OpenSLR 123 parts (each ~3-5GB)
  # As of 2025, the corpus is split into 10 resource files on OpenSLR
  local BASE="https://www.openslr.org/resources/123"

  log "Fetching MagicData-RAMC file list from OpenSLR..."
  local index_html
  index_html=$(wget -qO- "https://www.openslr.org/123/")

  # Extract all .tar.gz links
  local urls=()
  while IFS= read -r line; do
    if [[ "${line}" =~ href=\"(${BASE}/[^\"]+\.tar\.gz)\" ]] || \
       [[ "${line}" =~ href=\"([^\"]+MagicData[^\"]+\.tar\.gz)\" ]]; then
      urls+=("${BASH_REMATCH[1]}")
    fi
  done <<< "${index_html}"

  # Fallback: known file names for MagicData-RAMC
  if [[ ${#urls[@]} -eq 0 ]]; then
    warn "Could not parse OpenSLR index; using known file names"
    for part in RAMC_dev.tar.gz RAMC_test.tar.gz \
                RAMC_train_set1.tar.gz RAMC_train_set2.tar.gz \
                RAMC_train_set3.tar.gz RAMC_train_set4.tar.gz \
                RAMC_train_set5.tar.gz RAMC_train_set6.tar.gz \
                RAMC_train_set7.tar.gz RAMC_train_set8.tar.gz; do
      urls+=("${BASE}/${part}")
    done
  fi

  log "Downloading ${#urls[@]} files with ${JOBS} parallel connections..."
  local pairs=()
  for url in "${urls[@]}"; do
    local fname
    fname=$(basename "${url}")
    [[ -f "${dest}/${fname}" ]] && { ok "Already exists: ${fname}"; continue; }
    pairs+=("${url} ${dest}/${fname}")
  done
  [[ ${#pairs[@]} -gt 0 ]] && fetch_parallel "${pairs[@]}"

  log "Extracting MagicData-RAMC archives..."
  for f in "${dest}"/*.tar.gz; do
    [[ -f "${f}" ]] || continue
    log "  Extracting $(basename "${f}")..."
    tar -xzf "${f}" -C "${dest}/" && rm -f "${f}"
  done

  # Build manifest
  need python3
  python3 - "${dest}" <<'PYEOF'
import sys, json, pathlib

dest = pathlib.Path(sys.argv[1])
records = []

# MagicData-RAMC transcripts are in JSON or TextGrid format per recording
# Try JSON metadata files first
for meta_file in sorted(dest.rglob("*.json")):
    if meta_file.name in ("ami_train.jsonl", "icsi_all.jsonl"):
        continue
    try:
        data = json.loads(meta_file.read_text())
        if isinstance(data, list):
            for seg in data:
                wav = meta_file.with_suffix(".wav")
                records.append({
                    "id": seg.get("session_id", meta_file.stem) + f"__{len(records):06d}",
                    "text": seg.get("text", seg.get("sentence", "")),
                    "audio": str(wav.relative_to(dest)) if wav.exists() else str(meta_file.with_suffix(".wav")),
                    "sample_rate": 16000,
                    "start": float(seg.get("begin_time", seg.get("start", 0.0))),
                    "end": float(seg.get("end_time", seg.get("end", 0.0))),
                    "language": "zh",
                    "source": "magicdata_ramc",
                    "metadata": {
                        "speaker_id": seg.get("speaker_id", ""),
                        "session_id": seg.get("session_id", meta_file.stem),
                        "emotion": seg.get("emotion", ""),
                        "topic": seg.get("topic", ""),
                    },
                })
    except Exception:
        pass

# Fallback: any .wav with a parallel .txt
if not records:
    for wav in sorted(dest.rglob("*.wav")):
        txt = wav.with_suffix(".txt")
        text = txt.read_text(errors="replace").strip() if txt.exists() else ""
        records.append({
            "id": wav.stem,
            "text": text,
            "audio": str(wav.relative_to(dest)),
            "sample_rate": 16000,
            "start": 0.0,
            "end": 0.0,
            "language": "zh",
            "source": "magicdata_ramc",
            "metadata": {},
        })

out = dest / "magicdata_ramc_all.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
print(f"Wrote {len(records)} segments → {out}")
PYEOF

  ok "MagicData-RAMC download complete → ${dest}"
}

# ── AISHELL-4 ──────────────────────────────────────────────────────────────
# OpenSLR 111: https://www.openslr.org/111/
# License: CC BY 4.0
# Size: ~20GB compressed, ~30GB extracted
# 120h, 211 meeting sessions, 4-8 speakers, Mandarin
download_aishell4() {
  local dest="${DATA_ROOT}/aishell4"
  mkdir -p "${dest}"
  log "Downloading AISHELL-4 → ${dest}"

  need wget

  local BASE="https://www.openslr.org/resources/111"

  # Known files for AISHELL-4
  local parts=(
    "train_S.tar.gz"
    "train_M.tar.gz"
    "train_L.tar.gz"
    "test.tar.gz"
  )

  local pairs=()
  for part in "${parts[@]}"; do
    local url="${BASE}/${part}"
    local fname="${dest}/${part}"
    if [[ -f "${fname}" ]]; then
      ok "Already exists: ${part}"
    elif [[ -d "${dest}/${part%.tar.gz}" ]]; then
      ok "Already extracted: ${part%.tar.gz}"
    else
      pairs+=("${url} ${fname}")
    fi
  done

  if [[ ${#pairs[@]} -gt 0 ]]; then
    log "Downloading ${#pairs[@]} AISHELL-4 files..."
    fetch_parallel "${pairs[@]}"
  fi

  log "Extracting AISHELL-4 archives..."
  for f in "${dest}"/*.tar.gz; do
    [[ -f "${f}" ]] || continue
    log "  Extracting $(basename "${f}")..."
    tar -xzf "${f}" -C "${dest}/" && rm -f "${f}"
  done

  # Build manifest from AISHELL-4 structure:
  # Each session: <session_id>/<session_id>_L.wav and <session_id>.TextGrid
  need python3
  python3 - "${dest}" <<'PYEOF'
import sys, json, pathlib, re

dest = pathlib.Path(sys.argv[1])
records = []

# Parse TextGrid files for speaker segments
def parse_textgrid(tg_path):
    """Extract speaker segments from a Praat TextGrid file."""
    segs = []
    text = tg_path.read_text(errors="replace")
    # Find each IntervalTier
    tiers = re.split(r'item\s*\[(\d+)\]', text)[1:]
    for i in range(0, len(tiers) - 1, 2):
        tier_text = tiers[i + 1]
        name_m = re.search(r'name\s*=\s*"([^"]*)"', tier_text)
        if not name_m:
            continue
        speaker = name_m.group(1)
        for m in re.finditer(
            r'intervals\s*\[(\d+)\][^x]*xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"',
            tier_text, re.DOTALL
        ):
            xmin, xmax, text_val = float(m.group(2)), float(m.group(3)), m.group(4).strip()
            if text_val and xmax > xmin:
                segs.append((speaker, xmin, xmax, text_val))
    return segs

for tg in sorted(dest.rglob("*.TextGrid")):
    session_id = tg.stem
    # Find the corresponding 8-channel audio or mix
    wav_candidates = list(tg.parent.glob(f"{session_id}*.wav"))
    # Prefer the linear mix (L suffix)
    wav = next((w for w in wav_candidates if w.stem.endswith("_L")), None)
    if wav is None and wav_candidates:
        wav = wav_candidates[0]
    audio_path = str(wav.relative_to(dest)) if wav else f"{session_id}/{session_id}_L.wav"

    try:
        segs = parse_textgrid(tg)
    except Exception as e:
        print(f"  Warning: {tg}: {e}")
        continue

    for speaker, start, end, text in segs:
        records.append({
            "id": f"{session_id}__{speaker}_{len(records):06d}",
            "text": text,
            "audio": audio_path,
            "sample_rate": 16000,
            "start": round(start, 6),
            "end": round(end, 6),
            "language": "zh",
            "source": "aishell4",
            "metadata": {
                "session_id": session_id,
                "speaker_id": speaker,
            },
        })

out = dest / "aishell4_all.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
print(f"Wrote {len(records)} segments → {out}")
PYEOF

  ok "AISHELL-4 download complete → ${dest}"
}

# ── Status ─────────────────────────────────────────────────────────────────
show_status() {
  echo ""
  echo "Dataset download status (DATA_ROOT=${DATA_ROOT})"
  echo "──────────────────────────────────────────────────────────────────"
  printf '%-20s  %-12s  %-12s  %s\n' "Dataset" "Downloaded" "Disk used" "Manifest"

  check_ds() {
    local name="$1" dir="$2" manifest="$3"
    local size="—"
    local manifest_info="—"
    if [[ -d "${DATA_ROOT}/${dir}" ]]; then
      size=$(du -sh "${DATA_ROOT}/${dir}" 2>/dev/null | cut -f1)
      [[ -f "${DATA_ROOT}/${dir}/${manifest}" ]] && manifest_info="${manifest}"
      printf '%-20s  %-12s  %-12s  %s\n' "${name}" "yes" "${size}" "${manifest_info}"
    else
      printf '%-20s  %-12s  %-12s  %s\n' "${name}" "no" "—" "—"
    fi
  }

  check_ds "AMI"           "ami"           "ami_train.jsonl"
  check_ds "ICSI"          "icsi"          "icsi_all.jsonl"
  check_ds "MagicData-RAMC" "magicdata_ramc" "magicdata_ramc_all.jsonl"
  check_ds "AISHELL-4"     "aishell4"      "aishell4_all.jsonl"
  echo ""
  echo "LDC datasets (require license — https://www.ldc.upenn.edu/):"
  echo "  Switchboard  LDC97S62   — standard corpus for VAP/TurnGPT/SmartTurn"
  echo "  Fisher       LDC2004S13 — standard corpus for SmartTurn/EasyTurn"
  echo "  CallHome     LDC97S42   — standard corpus for VAP"
  echo ""
  df -h "${DATA_ROOT}" | awk 'NR==1{print "Disk: "$0} NR==2{print "     "$0}'
}

# ── LDC instructions ───────────────────────────────────────────────────────
show_ldc_instructions() {
  cat <<'EOF'

LDC Dataset Application Instructions
══════════════════════════════════════

These three datasets are the standard benchmarks for turn-taking research
(VAP, TurnGPT, SmartTurn all trained/evaluated on these).

1. Check if your institution is already an LDC member:
   https://www.ldc.upenn.edu/members
   If yes, your institution coordinator can order at no marginal cost.

2. If not a member, apply for individual Non-Member License Agreements:
   - Switchboard: https://catalog.ldc.upenn.edu/LDC97S62
   - Fisher Pt1:  https://catalog.ldc.upenn.edu/LDC2004S13
   - Fisher Pt2:  https://catalog.ldc.upenn.edu/LDC2005S13
   - CallHome:    https://catalog.ldc.upenn.edu/LDC97S42

   Click "Add to Cart" → complete the Non-Member License Agreement form.
   Typical fee: $150–$500 per corpus. Processing: 1–2 weeks.

3. Once downloaded (DVD ISO or direct download), place under:
   /data/data/switchboard/
   /data/data/fisher/
   /data/data/callhome/

   Then run:
   scripts/download_datasets.sh ldc_convert

EOF
}

# ── LDC post-processing (after manual download) ────────────────────────────
# Run this after you have placed the LDC archives in /data/data/ldc_raw/
ldc_convert() {
  log "Converting LDC data to stable-asr manifests"
  need python3

  for corpus in switchboard fisher callhome; do
    local src="${DATA_ROOT}/ldc_raw/${corpus}"
    local dest="${DATA_ROOT}/${corpus}"
    if [[ ! -d "${src}" ]]; then
      warn "${corpus}: no source directory at ${src} — skipping"
      continue
    fi
    mkdir -p "${dest}"
    python3 - "${src}" "${dest}" "${corpus}" <<'PYEOF'
import sys, json, pathlib, re

src, dest, corpus = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
records = []

# Switchboard / Fisher / CallHome share a similar transcript structure:
# .trans or .txt files with lines: <start> <end> <speaker> <text>
# or .stm files with: <file> <channel> <speaker> <start> <end> <text>

for stm in sorted(src.rglob("*.stm")):
    try:
        for line in stm.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith(";;"):
                continue
            parts = line.split(None, 5)
            if len(parts) < 5:
                continue
            fname, chan, speaker = parts[0], parts[1], parts[2]
            start, end = float(parts[3]), float(parts[4])
            text = parts[5] if len(parts) > 5 else ""
            text = re.sub(r'<[^>]+>', '', text).strip()  # strip SGML tags
            if not text or text == "ignore_time_segment_in_scoring":
                continue
            records.append({
                "id": f"{fname}__{chan}_{speaker}_{len(records):06d}",
                "text": text,
                "audio": str(fname),
                "sample_rate": 8000,
                "start": round(start, 6),
                "end": round(end, 6),
                "language": "en",
                "source": corpus,
                "metadata": {
                    "speaker_id": speaker,
                    "channel": chan,
                    "file": fname,
                },
            })
    except Exception as e:
        print(f"  Warning: {stm}: {e}")

out = dest / f"{corpus}_all.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
print(f"Wrote {len(records)} segments → {out}")
PYEOF
  done
}

# ── Main ───────────────────────────────────────────────────────────────────

if [[ $# -eq 0 ]]; then
  cat >&2 <<'EOF'
Usage: scripts/download_datasets.sh <dataset> [<dataset> ...]

Datasets (free download):
  ami          AMI Meeting Corpus    (~17GB download)
  icsi         ICSI Meeting Corpus   (~8GB download)
  magicdata    MagicData-RAMC        (~43GB download)
  aishell4     AISHELL-4             (~20GB download)
  all          all four free datasets

LDC datasets (manual apply + download first):
  ldc_convert  convert LDC data in /data/data/ldc_raw/ to manifests

Information:
  status       show download status and disk usage
  ldc          show LDC application instructions

DATA_ROOT env var sets base directory (default: /data/data)
EOF
  exit 2
fi

for arg in "$@"; do
  case "${arg}" in
    ami)         download_ami ;;
    icsi)        download_icsi ;;
    magicdata)   download_magicdata ;;
    aishell4)    download_aishell4 ;;
    all)
      download_ami
      download_icsi
      download_magicdata
      download_aishell4
      ;;
    ldc_convert) ldc_convert ;;
    ldc)         show_ldc_instructions ;;
    status)      show_status ;;
    *)
      die "Unknown dataset: ${arg}. Run without args to see usage."
      ;;
  esac
done

show_status
