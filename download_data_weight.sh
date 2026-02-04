#!/bin/bash

BASE_DIR="$FAST"
BASE_DIR="$FAST"HOSTNAME=$(hostname)


# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory $BASE_DIR does not exist."
    exit 1
fi

case "$HOSTNAME" in
    *leonardo*)
        echo "Running on Leonardo cluster."
        DATASET_PATH="$BASE_DIR/data"
esac