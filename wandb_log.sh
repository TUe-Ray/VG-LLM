#!/bin/bash

BASE_DIR="$WORK/logs/vgllm"  # 根據你的 launch_training.sh 裡的 FAST_DIR 修改

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory $BASE_DIR does not exist."
    exit 1
fi

# Change to base directory
cd "$BASE_DIR" || exit 1

# Sync interval in seconds (default: 60 seconds)
SYNC_INTERVAL=${SYNC_INTERVAL:-60}

# Handle Ctrl+C gracefully
trap 'echo ""; echo "Stopping wandb sync..."; exit 0' INT TERM

echo "Starting continuous wandb sync from $BASE_DIR"
echo "Sync interval: ${SYNC_INTERVAL} seconds"
echo "Press Ctrl+C to stop"
echo "======================================"
echo ""

# Continuous loop
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] Starting sync iteration..."
    echo "--------------------------------------"
    
    # Iterate through all subdirectories
    for experiment_dir in */; do
        # Remove trailing slash
        experiment_name="${experiment_dir%/}"
        
        # Check if wandb directory exists
        wandb_dir="$experiment_name/wandb"
        if [ ! -d "$wandb_dir" ]; then
            continue
        fi
        
        # Find offline run directories
        offline_runs_dir="$wandb_dir/offline-run-*"
        
        # Check if any offline runs exist
        if ! ls $offline_runs_dir 1> /dev/null 2>&1; then
            continue
        fi
        
        echo "Processing experiment: $experiment_name"
        
        # Sync each offline run
        for offline_run in $offline_runs_dir; do
            if [ -d "$offline_run" ]; then
                echo "  Syncing: $offline_run"
                # Temporarily disable exit on error to ensure we continue processing
                set +e
                # Run wandb sync and capture exit status
                wandb sync "$offline_run" 2>&1 | grep -v "^wandb:"
                sync_exit_code=${PIPESTATUS[0]}
                set -e
                
                # Check if sync was successful
                if [ $sync_exit_code -eq 0 ]; then
                    echo "  ✓ Successfully synced: $(basename $offline_run)"
                else
                    echo "  ✗ Failed to sync: $(basename $offline_run) (exit code: $sync_exit_code)"
                fi
            fi
        done
    done
    
    echo "--------------------------------------"
    echo "[$TIMESTAMP] Sync iteration completed. Waiting ${SYNC_INTERVAL} seconds before next iteration..."
    echo ""
    
    # Wait before next iteration
    sleep "$SYNC_INTERVAL"
done