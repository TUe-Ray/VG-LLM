#!/bin/bash


WANDB_ROOT="${WANDB_ROOT:-$WORK/wandb/wandb}"   # 指到「含有 offline-run-* 的那層」
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"
MARKER_FILE="${MARKER_FILE:-.wandb_synced}"  # 成功 sync 後寫 marker，避免重複 sync

# Check if base directory exists
if [ ! -d "$WANDB_ROOT" ]; then
    echo "Error: Base directory $WANDB_ROOT does not exist."
    exit 1
fi

# Change to base directory
cd "$WANDB_ROOT" || exit 1

# Handle Ctrl+C gracefully
trap 'echo ""; echo "Stopping wandb sync..."; exit 0' INT TERM

echo "Starting continuous wandb sync from $WANDB_ROOT"
echo "Sync interval: ${SYNC_INTERVAL} seconds"
echo "Press Ctrl+C to stop"
echo "======================================"
echo ""

# If no matching dirs, glob expands to nothing (instead of literal pattern)
shopt -s nullglob

# Continuous loop
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] Starting sync iteration..."
    echo "--------------------------------------"

    synced_runs=()

    # Find offline run directories directly under WANDB_ROOT
    run_dirs=(offline-run-* run-*)

    if [ ${#run_dirs[@]} -eq 0 ]; then
        echo "No offline runs found under $WANDB_ROOT (expected offline-run-* or run-*)."
    else
        for run_dir in "${run_dirs[@]}"; do
            [ -d "$run_dir" ] || continue

            # Skip if already synced successfully before
            if [ -f "$run_dir/$MARKER_FILE" ]; then
                continue
            fi

            echo "Syncing: $run_dir"

            # Temporarily disable exit on error to ensure we continue processing
            set +e
            wandb sync "$run_dir" 2>&1 | grep -v "^wandb:"
            sync_exit_code=${PIPESTATUS[0]}
            set -e

            if [ $sync_exit_code -eq 0 ]; then
                echo "  ✓ Successfully synced: $(basename "$run_dir")"
                synced_runs+=("$(basename "$run_dir")")
                # mark as synced to avoid repeated syncing
                touch "$run_dir/$MARKER_FILE" || true
            else
                echo "  ✗ Failed to sync: $(basename "$run_dir") (exit code: $sync_exit_code)"
            fi
        done
    fi

    if [ ${#synced_runs[@]} -gt 0 ]; then
        echo "Synced runs in this iteration: ${synced_runs[*]}"
    fi

    echo "--------------------------------------"
    echo "[$TIMESTAMP] Sync iteration completed. Waiting ${SYNC_INTERVAL} seconds before next iteration..."
    echo ""
    sleep "$SYNC_INTERVAL"
done
