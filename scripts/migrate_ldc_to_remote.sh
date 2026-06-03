#!/usr/bin/env bash
# Stable-ASR — migrate LDC corpora to a remote H100 host.
#
# What this is:
#   A safe, resume-capable wrapper around rsync-over-SSH for shipping
#   downloaded LDC archives from this machine to the Japan H100 cluster
#   (or any remote machine that is part of your LDC-licensed
#   institution's infrastructure).
#
# Why rsync (not scp / aws s3 / cloud drives):
#   - Real per-byte resume via --partial + --append-verify
#   - Skips already-transferred files via size+mtime check
#   - Encrypted in transit (over SSH)
#   - LDC license safe: stays inside your own institution's machines
#
# Usage:
#   bash scripts/migrate_ldc_to_remote.sh user@japan-h100:/mnt/ldc/
#   bash scripts/migrate_ldc_to_remote.sh user@japan-h100:/mnt/ldc/ /data/ldc/
#   DRY=1 bash scripts/migrate_ldc_to_remote.sh user@japan-h100:/mnt/ldc/  # dry-run
#
# Args:
#   $1  remote target  (required)  — e.g. li@10.0.0.5:/mnt/ldc/
#   $2  local source   (optional)  — default: /data/ldc/
#
# Environment:
#   DRY=1                 dry-run, list what would be transferred
#   BWLIMIT_KBPS=20000    cap upstream bandwidth (KB/s); default 20MB/s
#   SSH_KEY=~/.ssh/id_ed25519   override SSH key
#   SSH_PORT=22           override SSH port
#
# Verification:
#   This script also writes a manifest of (path, size, sha256) for every
#   transferred file, both locally and remotely. After transfer you can
#   run:
#     bash scripts/migrate_ldc_to_remote.sh --verify user@japan-h100:/mnt/ldc/
#   to compare the two manifests.
#
# License note (read once):
#   LDC license requires data to remain on machines covered by your
#   institution's license. This script assumes the remote host is part
#   of the same licensed institution. It does NOT enforce this — you do.

set -euo pipefail

# ---- pretty -----------------------------------------------------------------

log()  { printf '\n\033[1;34m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---- args -------------------------------------------------------------------

VERIFY_MODE=0
if [[ "${1:-}" == "--verify" ]]; then
  VERIFY_MODE=1
  shift
fi

TARGET="${1:-}"
SRC="${2:-/data/ldc/}"
[[ -n "${TARGET}" ]] || die "usage: $0 [--verify] user@host:/remote/path/  [/local/src/]"
[[ -d "${SRC}" ]]    || die "local source not found: ${SRC}"

BWLIMIT_KBPS="${BWLIMIT_KBPS:-20000}"
SSH_KEY="${SSH_KEY:-}"
SSH_PORT="${SSH_PORT:-22}"

SSH_OPTS=( -o ControlMaster=auto -o ControlPath="${HOME}/.ssh/cm-%r@%h:%p" -o ControlPersist=600 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 )
[[ -n "${SSH_KEY}" ]] && SSH_OPTS+=( -i "${SSH_KEY}" )
SSH_OPTS+=( -p "${SSH_PORT}" )

SSH_CMD="ssh ${SSH_OPTS[*]}"

# ---- manifests --------------------------------------------------------------
#
# We write a flat sha256 manifest for every regular file under SRC. Verify
# mode rsyncs the remote manifest back and diffs.

LOG_DIR="${HOME}/ldc_dl/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d-%H%M%S)
LOCAL_MANIFEST="${LOG_DIR}/ldc-local-${TS}.sha256"
REMOTE_MANIFEST="${LOG_DIR}/ldc-remote-${TS}.sha256"

build_local_manifest() {
  log "Building local sha256 manifest of ${SRC}"
  ( cd "${SRC}" && find . -type f \( -name "*.zip*" -o -name "*.tgz" -o -name "*.tar*" -o -name "*.flac" -o -name "*.wav" -o -name "*.sph" \) -print0 \
      | xargs -0 -P 4 sha256sum ) > "${LOCAL_MANIFEST}"
  ok "Local manifest: ${LOCAL_MANIFEST} ($(wc -l < "${LOCAL_MANIFEST}") files)"
}

# ---- verify mode ------------------------------------------------------------

if (( VERIFY_MODE == 1 )); then
  build_local_manifest
  REMOTE_DIR="${TARGET#*:}"
  REMOTE_HOST="${TARGET%%:*}"
  log "Building remote sha256 manifest on ${REMOTE_HOST}:${REMOTE_DIR}"
  # shellcheck disable=SC2029
  ${SSH_CMD} "${REMOTE_HOST}" "cd '${REMOTE_DIR}' && find . -type f \\( -name '*.zip*' -o -name '*.tgz' -o -name '*.tar*' -o -name '*.flac' -o -name '*.wav' -o -name '*.sph' \\) -print0 | xargs -0 -P 4 sha256sum" > "${REMOTE_MANIFEST}"
  ok "Remote manifest: ${REMOTE_MANIFEST} ($(wc -l < "${REMOTE_MANIFEST}") files)"
  log "Diffing manifests"
  if diff <(sort "${LOCAL_MANIFEST}") <(sort "${REMOTE_MANIFEST}"); then
    ok "Identical: every file in local and remote matches by sha256."
    exit 0
  else
    die "Mismatch between local and remote manifests. See diff above."
  fi
fi

# ---- transfer ---------------------------------------------------------------

log "Stable-ASR LDC migration"
echo "  source:       ${SRC}"
echo "  target:       ${TARGET}"
echo "  bandwidth:    ${BWLIMIT_KBPS} KB/s"
echo "  ssh:          ${SSH_CMD}"
echo "  dry-run:      ${DRY:-0}"

build_local_manifest

RSYNC_FLAGS=(
  --archive             # -rlptgoD: preserve everything
  --partial             # keep partial files for resume
  --append-verify       # resume + verify the appended bytes
  --inplace             # do not write to a temp file
  --human-readable
  --info=progress2,stats2
  --bwlimit="${BWLIMIT_KBPS}"
  --rsh="${SSH_CMD}"
  --include="*/"
  --include="*.zip*"
  --include="*.tgz"
  --include="*.tar*"
  --include="*.flac"
  --include="*.wav"
  --include="*.sph"
  --include="CHECKSUMS-*.txt"
  --exclude="*"
)

[[ "${DRY:-0}" == "1" ]] && RSYNC_FLAGS+=( --dry-run )

log "Starting rsync (Ctrl-C to stop; safe to re-run for resume)"
rsync "${RSYNC_FLAGS[@]}" "${SRC%/}/" "${TARGET%/}/"
ok "rsync completed"

# Also ship the local manifest so the remote can self-check anytime.
log "Shipping local manifest to remote for future verification"
rsync "${RSYNC_FLAGS[@]}" "${LOCAL_MANIFEST}" "${TARGET%/}/_manifests/"

ok "Migration done. To verify byte-identity:"
echo "    bash scripts/migrate_ldc_to_remote.sh --verify ${TARGET} ${SRC}"
