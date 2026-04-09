#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-configs/cm_autoresearch_v1.json}"
shift || true

ABS_CONFIG="$(cd "$ROOT_DIR" && python - <<'PY' "$CONFIG_PATH"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve() if Path(sys.argv[1]).is_absolute() else (Path.cwd() / sys.argv[1]).resolve()
print(root)
PY
)"

OUTPUT_DIR="$(python - <<'PY' "$ABS_CONFIG"
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())
root = config_path.parent.parent
print((root / config["output_dir"]).resolve())
PY
)"

PID_FILE="$OUTPUT_DIR/autoresearch.pid"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Autoresearch already running with PID $OLD_PID"
    exit 1
  fi
  rm -f "$PID_FILE"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/autoresearch_${TIMESTAMP}.log"

cd "$ROOT_DIR"
nohup "$PYTHON_BIN" scripts/cm_autoresearch.py --config "$ABS_CONFIG" "$@" >"$LOG_FILE" 2>&1 &
PID="$!"
echo "$PID" > "$PID_FILE"

echo "pid=$PID"
echo "log=$LOG_FILE"
echo "output_dir=$OUTPUT_DIR"
