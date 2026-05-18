#!/usr/bin/env bash
# Train the production model on Kaggle (not locally), then download artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_DIR="$ROOT/kaggle"
METADATA="$KAGGLE_DIR/kernel-metadata.json"
ARTIFACTS_DIR="$ROOT/artifacts"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-30}"

die() {
  echo "error: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

read_kernel_id() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["id"])' "$METADATA"
}

configure_kernel_id() {
  local username
  username="$(kaggle config view | awk -F': ' '/^\- username:/{print $2}')"
  [[ -n "$username" ]] || die "Could not read Kaggle username. Run: kaggle config set -n username -v YOUR_USERNAME"
  python3 - "$METADATA" "$username" <<'PY'
import json
import sys

path, username = sys.argv[1], sys.argv[2]
meta = json.load(open(path, encoding="utf-8"))
slug = meta["id"].split("/", 1)[-1]
meta["id"] = f"{username}/{slug}"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(meta, handle, indent=2)
    handle.write("\n")
print(f"Configured kernel id: {meta['id']}")
PY
}

wait_for_kernel() {
  local kernel_id="$1"
  local status
  echo "Waiting for Kaggle kernel: $kernel_id"
  while true; do
    status="$(kaggle kernels status "$kernel_id" 2>&1 || true)"
    echo "$status"
    case "$status" in
      *COMPLETE*|*complete*)
        return 0
        ;;
      *ERROR*|*error*|*FAILED*|*failed*)
        die "Kernel failed. Fetch logs: kaggle kernels output $kernel_id -p /tmp/kaggle-logs"
        ;;
    esac
    sleep "$POLL_SECONDS"
  done
}

flatten_artifacts() {
  while [[ -f "$ARTIFACTS_DIR/artifacts/risk_model.joblib" ]]; do
    mv -f "$ARTIFACTS_DIR/artifacts/"* "$ARTIFACTS_DIR/"
    rmdir "$ARTIFACTS_DIR/artifacts" 2>/dev/null || break
  done
}

main() {
  require_cmd kaggle
  require_cmd python3

  local kernel_id="${KAGGLE_KERNEL_ID:-}"
  if [[ -z "$kernel_id" ]]; then
    configure_kernel_id
    kernel_id="$(read_kernel_id)"
  fi

  echo "Bundling kaggle_train.py from training_pipeline.py"
  python3 "$ROOT/scripts/sync_kaggle_train.py"

  echo "Pushing kernel from $KAGGLE_DIR"
  kaggle kernels push -p "$KAGGLE_DIR"

  wait_for_kernel "$kernel_id"

  mkdir -p "$ARTIFACTS_DIR"
  echo "Downloading outputs to $ARTIFACTS_DIR"
  kaggle kernels output "$kernel_id" -p "$ARTIFACTS_DIR" -f
  flatten_artifacts

  [[ -f "$ARTIFACTS_DIR/risk_model.joblib" ]] || die "risk_model.joblib was not downloaded"

  echo "Done. Artifact: $ARTIFACTS_DIR/risk_model.joblib"
  if [[ -f "$ARTIFACTS_DIR/metrics.json" ]]; then
    python3 -c '
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
sel = m["selected_model"]
print(f"Selected model: {sel}")
print(json.dumps(m["metrics"][sel], indent=2))
' "$ARTIFACTS_DIR/metrics.json"
  fi
}

main "$@"
