#!/usr/bin/env bash
# Stable-ASR — LDC batch downloader.
#
# What this is:
#   A robust, resume-capable downloader for LDC catalog token URLs. Built
#   for the Stable-ASR v1 paper plan (see ROADMAP.md §"Paper Direction
#   (locked 2026-06-03)") which requires Switchboard / Fisher / CallHome
#   and related LDC corpora.
#
# Why aria2c instead of wget:
#   - LDC ships large multi-part archives (.zip.001/.002/...). aria2c can
#     download the parts in parallel and resume each one accurately.
#   - Real per-byte resume controlled by .aria2 control files. wget's
#     --continue is fragile against LDC's content-disposition + token URLs.
#   - Multi-connection download (3-5x faster than wget on the same link).
#
# Why a urls file instead of hardcoded URLs:
#   - LDC download tokens are 24h short-lived. You will refresh them.
#   - The urls file lives outside the repo (under ~/ldc_dl/) so the tokens
#     never get committed.
#
# Usage:
#   1) Refresh your LDC cookie:
#        - Log in to https://catalog.ldc.upenn.edu in a browser
#        - Use a "Get cookies.txt LOCALLY" extension to export cookies
#        - Save as ~/ldc_dl/cookies.txt
#   2) Refresh your urls file:
#        - On each LDC catalog page (e.g. https://catalog.ldc.upenn.edu/LDC97S62),
#          right-click the green Download button → Copy link
#        - Edit ~/ldc_dl/urls.txt following ldc_dl/urls.example.txt
#   3) Run:
#        bash scripts/download_ldc.sh ~/ldc_dl/urls.txt /data/ldc/
#
# Sanity checks:
#   - If a downloaded file is < 50KB and starts with "<", we assume the
#     cookie expired and aria2c was redirected to the login HTML. The
#     script then aborts with a clear error.
#   - After download, MD5/SHA256 are computed and printed for manual
#     comparison with the LDC catalog page checksums.
#
# This script does NOT:
#   - extract zip archives (that's prepare_ldc.py's job)
#   - validate LDC license eligibility (that's on you)
#   - upload to any cloud (use scripts/migrate_ldc_to_remote.sh for that)

set -euo pipefail

# ---- args -------------------------------------------------------------------

URLS_FILE="${1:-${HOME}/ldc_dl/urls.txt}"
OUT_DIR="${2:-/data/ldc}"
COOKIES="${LDC_COOKIES:-${HOME}/ldc_dl/cookies.txt}"

# ---- pretty -----------------------------------------------------------------

log()  { printf '\n\033[1;34m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---- pre-flight -------------------------------------------------------------

command -v aria2c >/dev/null 2>&1 || die "aria2c not installed. Install:  sudo apt-get install -y aria2"
[[ -f "${URLS_FILE}" ]] || die "urls file not found: ${URLS_FILE}  (see ~/ldc_dl/urls.example.txt)"
[[ -f "${COOKIES}" ]]   || die "cookies.txt not found: ${COOKIES}  (export from your browser after logging in to LDC)"

# Cookie freshness: LDC sessions live ~24h. Warn if older.
COOKIE_AGE_HOURS=$(( ($(date +%s) - $(stat -c %Y "${COOKIES}")) / 3600 ))
if (( COOKIE_AGE_HOURS > 20 )); then
  warn "cookies.txt is ${COOKIE_AGE_HOURS}h old; LDC sessions usually expire after ~24h. Consider refreshing before a long download."
fi

mkdir -p "${OUT_DIR}"
LOG_DIR="${HOME}/ldc_dl/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/aria2c-$(date +%Y%m%d-%H%M%S).log"

log "Stable-ASR LDC downloader"
echo "  urls file:    ${URLS_FILE}"
echo "  output dir:   ${OUT_DIR}"
echo "  cookies:      ${COOKIES}  (age: ${COOKIE_AGE_HOURS}h)"
echo "  log file:     ${LOG_FILE}"

# ---- run aria2c -------------------------------------------------------------
#
# Flag rationale (all chosen for LDC's quirks; do not casually edit):
#   --continue=true                  resume partial downloads
#   --max-tries=20                   LDC sometimes flakes mid-stream
#   --retry-wait=15                  back off between retries
#   --max-connection-per-server=4    LDC allows up to ~4-8; 4 is safe
#   --split=4                        each file split into 4 parallel ranges
#   --min-split-size=10M             do not split tiny files
#   --max-concurrent-downloads=2     two distinct corpora at once
#   --auto-file-renaming=false       respect the `out=` in urls.txt
#   --check-certificate=true         LDC has valid TLS, no need to disable
#   --remote-time=true               preserve LDC's mtime, useful for audits
#   --file-allocation=falloc         pre-allocate to avoid fragmentation
#   --conditional-get=false          LDC token URLs do not support If-Modified-Since
#   --allow-overwrite=false          never silently destroy completed files

set +e
aria2c \
  --load-cookies="${COOKIES}" \
  --dir="${OUT_DIR}" \
  --input-file="${URLS_FILE}" \
  --continue=true \
  --max-tries=20 \
  --retry-wait=15 \
  --timeout=60 \
  --connect-timeout=30 \
  --max-connection-per-server=4 \
  --split=4 \
  --min-split-size=10M \
  --max-concurrent-downloads=2 \
  --auto-file-renaming=false \
  --check-certificate=true \
  --remote-time=true \
  --file-allocation=falloc \
  --conditional-get=false \
  --allow-overwrite=false \
  --console-log-level=warn \
  --summary-interval=30 \
  --log="${LOG_FILE}" \
  --log-level=info
ARIA_STATUS=$?
set -e

if (( ARIA_STATUS != 0 )); then
  warn "aria2c exited with status ${ARIA_STATUS}; some files may be incomplete."
  warn "Re-running this script will resume from where it stopped."
fi

# ---- sanity: detect cookie-expired login HTML drops -------------------------

log "Sanity-checking downloaded files"
SUSPICIOUS=()
while IFS= read -r f; do
  [[ -f "${f}" ]] || continue
  size=$(stat -c %s "${f}")
  if (( size < 51200 )); then
    head1=$(head -c 1 "${f}" 2>/dev/null || true)
    if [[ "${head1}" == "<" ]]; then
      SUSPICIOUS+=("${f}")
    fi
  fi
done < <(find "${OUT_DIR}" -maxdepth 4 -type f \( -name "*.zip*" -o -name "*.tgz" -o -name "*.tar*" \))

if (( ${#SUSPICIOUS[@]} > 0 )); then
  warn "These files look like login-page HTML, not LDC archives:"
  for f in "${SUSPICIOUS[@]}"; do printf '  %s (%d bytes)\n' "${f}" "$(stat -c %s "${f}")"; done
  warn "→ Your LDC cookie is most likely expired. Refresh ${COOKIES} and re-run."
  warn "→ Delete the bad files first:  rm ${SUSPICIOUS[*]}"
  exit 3
fi

# ---- checksums --------------------------------------------------------------

log "Computing checksums (compare against the LDC catalog page for each corpus)"
CKSUM_FILE="${OUT_DIR}/CHECKSUMS-$(date +%Y%m%d-%H%M%S).txt"
{
  echo "# Stable-ASR LDC download — $(date -Iseconds)"
  echo "# Verify each row against the corresponding LDC catalog page."
  echo
  printf '%-12s  %-12s  %-64s  %s\n' "SIZE" "MD5" "SHA256" "PATH"
  while IFS= read -r f; do
    [[ -f "${f}" ]] || continue
    size=$(stat -c %s "${f}")
    md5=$(md5sum "${f}" | awk '{print $1}')
    sha=$(sha256sum "${f}" | awk '{print $1}')
    printf '%-12s  %-32s  %-64s  %s\n' "${size}" "${md5}" "${sha}" "${f}"
  done < <(find "${OUT_DIR}" -maxdepth 4 -type f \( -name "*.zip*" -o -name "*.tgz" -o -name "*.tar*" \) | sort)
} | tee "${CKSUM_FILE}"

ok "Checksums saved to ${CKSUM_FILE}"
ok "Done. If this run was interrupted, just re-invoke the same command — aria2c will resume."
