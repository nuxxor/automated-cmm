#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-configs/cm_autoresearch_v1.json}"

ABS_CONFIG="$(cd "$ROOT_DIR" && python - <<'PY' "$CONFIG_PATH"
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

if [[ ! -f "$PID_FILE" ]]; then
  echo "No pid file found at $PID_FILE"
  exit 1
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped PID $PID"
else
  echo "PID $PID is not running"
fi
rm -f "$PID_FILE"
