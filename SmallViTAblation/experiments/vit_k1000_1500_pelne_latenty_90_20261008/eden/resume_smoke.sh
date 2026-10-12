#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
MARKER="$ROOT/submission/SMOKE_SUBMITTED.json"
if [[ ! -f "$MARKER" ]]; then
  echo "Brak markera poprzedniego smoke" >&2
  exit 7
fi
JOB=$("$PY" - "$MARKER" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(value["jobs"]["smoke"].split(";", 1)[0])
PY
)
if [[ -n "$(squeue -h -j "$JOB" -o '%T')" ]]; then
  echo "Poprzedni smoke $JOB nadal jest aktywny" >&2
  exit 8
fi
STATE=$(sacct -X -j "$JOB" --format=State -n -P | head -n 1 | cut -d'|' -f1)
case "$STATE" in
  FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|BOOT_FAIL*)
    ;;
  *)
    echo "Poprzedni smoke $JOB ma stan $STATE; automatyczne wznowienie zablokowane" >&2
    exit 9
    ;;
esac
ARCHIVE="$ROOT/submission/SMOKE_SUBMITTED.failed_${JOB}.json"
if [[ -e "$ARCHIVE" ]]; then
  echo "Historia poprzedniego smoke już istnieje: $ARCHIVE" >&2
  exit 10
fi
mv -- "$MARKER" "$ARCHIVE"
"$HERE/submit_smoke.sh"
