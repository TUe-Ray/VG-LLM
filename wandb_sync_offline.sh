#!/bin/bash


WANDB_ROOT="${WANDB_ROOT:-$WORK/wandb/wandb}"   # 指到「含有 offline-run-* 的那層」
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"
ACTIVE_MINUTES="${ACTIVE_MINUTES:-5}"
EXPERIMENT_NAME="${1:-}"   # Optional: pass an experiment name to sync that run regardless of age

# Load wandb credentials
if [ -f "$HOME/.wandb_env" ]; then
    source "$HOME/.wandb_env"
else
    echo "Error: ~/.wandb_env not found."
    exit 1
fi

# Ensure API key is set
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY is not set."
    exit 1
fi

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
if [ -n "$EXPERIMENT_NAME" ]; then
    echo "Mode: sync runs matching experiment name '$EXPERIMENT_NAME' (no age filter)"
else
    echo "Mode: sync runs active in last ${ACTIVE_MINUTES} minutes"
fi
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

    # ---- Determine which runs to sync ----
    if [ -n "$EXPERIMENT_NAME" ]; then
        # Sync all runs matching the experiment name, regardless of age 
        mapfile -d '' active_runs < <(
          find "$WANDB_ROOT" -maxdepth 1 -mindepth 1 -type d -name "offline-run-*${EXPERIMENT_NAME}*" -print0
        )

        if [ ${#active_runs[@]} -eq 0 ]; then
            echo "No runs found matching experiment name: '$EXPERIMENT_NAME'"
        else
            echo "Experiment filter active: syncing runs matching '$EXPERIMENT_NAME' (ignoring age)"
        fi
    else
        # ---- Only sync runs updated recently ----
        ACTIVE_MINUTES="${ACTIVE_MINUTES:-5}"

        mapfile -d '' active_runs < <(
          find "$WANDB_ROOT" -maxdepth 1 -mindepth 1 -type d -name "offline-run-*" -print0 \
          | while IFS= read -r -d '' d; do
              # If any file inside run was modified in last ACTIVE_MINUTES minutes
              if find "$d" -type f -mmin "-$ACTIVE_MINUTES" -print -quit | grep -q .; then
                printf '%s\0' "$d"
              fi
            done
        )

        if [ ${#active_runs[@]} -eq 0 ]; then
            echo "No active runs updated in last ${ACTIVE_MINUTES} minutes."
        fi
    fi

    if [ ${#active_runs[@]} -gt 0 ]; then
        for run_dir in "${active_runs[@]}"; do
            echo "Syncing active run: $(basename "$run_dir")"

            set +e
            wandb sync "$run_dir" 2>&1 |            # ...existing code...
            
            DIRECTORY_PATH="${2:-}"  # Optional: pass a specific directory path to sync
            
            # ...existing code...
            
            if [ -n "$DIRECTORY_PATH" ]; then
                echo "Mode: syncing specific directory '$DIRECTORY_PATH'"
                if [ -d "$DIRECTORY_PATH" ]; then
                    echo "Syncing directory: $DIRECTORY_PATH"
                    wandb sync "$DIRECTORY_PATH" 2>&1 | grep -v "^wandb:"
                else
                    echo "Error: Directory '$DIRECTORY_PATH' does not exist."
                    exit 1
                fi
                exit 0
            fi
            
            # ...existing code... grep -v "^wandb:"
            sync_exit_code=${PIPESTATUS[0]}
            set -e

            if [ $sync_exit_code -eq 0 ]; then
                synced_runs+=("$(basename "$run_dir")")
            else
                echo "  ✗ Failed (exit code: $sync_exit_code)"
            fi
        done
    fi

    if [ ${#synced_runs[@]} -gt 0 ]; then
        echo "Synced runs: ${synced_runs[*]}"
    fi

    echo "--------------------------------------"
    echo "[$TIMESTAMP] Waiting ${SYNC_INTERVAL} seconds..."
    echo ""
    sleep "$SYNC_INTERVAL"
done
