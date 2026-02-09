#!/usr/bin/env bash
WANDB_ROOT="${WANDB_ROOT:-$WORK/wandb}"   # 你的 WANDB_DIR
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"
MARKER_FILE="${MARKER_FILE:-.wandb_synced}"
DRY_RUN="${DRY_RUN:-0}"

trap 'echo ""; echo "Stopping wandb sync..."; exit 0' INT TERM

if [ ! -d "$WANDB_ROOT" ]; then
  echo "Error: WANDB_ROOT does not exist: $WANDB_ROOT"
  exit 1
fi

echo "Continuous wandb sync"
echo "WANDB_ROOT: $WANDB_ROOT"
echo "SYNC_INTERVAL: ${SYNC_INTERVAL}s"
echo "DRY_RUN: $DRY_RUN"
echo "======================================"
echo ""

sync_one () {
  local run_dir="$1"

  # 已同步過就跳過，避免一直重複打 API
  if [ -f "$run_dir/$MARKER_FILE" ]; then
    return 0
  fi

  echo "Syncing: $run_dir"
  if [ "$DRY_RUN" = "1" ]; then
    echo "(dry-run) wandb sync \"$run_dir\""
    return 0
  fi

  set +e
  wandb sync "$run_dir" 2>&1 | sed '/^wandb: /d'
  local code=${PIPESTATUS[0]}
  set -e

  if [ "$code" -eq 0 ]; then
    echo "✓ Synced: $(basename "$run_dir")"
    touch "$run_dir/$MARKER_FILE" || true
  else
    echo "✗ Failed: $(basename "$run_dir") (exit code: $code)"
  fi
}

while true; do
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$TS] Scan & sync..."

  # 找 offline-run-* 與 run-*
  mapfile -d '' runs < <(
    find "$WANDB_ROOT" -maxdepth 1 -mindepth 1 -type d \( -name "offline-run-*" -o -name "run-*" \) -print0 2>/dev/null
  )

  if [ "${#runs[@]}" -eq 0 ]; then
    echo "No runs found in $WANDB_ROOT"
  else
    for r in "${runs[@]}"; do
      sync_one "$r"
    done
  fi

  echo "Sleep ${SYNC_INTERVAL}s..."
  echo ""
  sleep "$SYNC_INTERVAL"
done