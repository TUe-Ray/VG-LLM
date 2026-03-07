#!/bin/bash
# ============================================================
# build_hdf5.sh  –  Convert training media to sharded HDF5
#
# This is a SINGLE-NODE CPU job (no GPU needed).
# Run BEFORE training to prepare the HDF5 cache.
#
# Submit:
#   sbatch scripts/preprocess/build_hdf5.sh
# ============================================================
#SBATCH --job-name=build_hdf5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --output=logs/preprocess/%x_%j.out
#SBATCH --error=logs/preprocess/%x_%j.err
#SBATCH --mem=30G

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
# Annotation JSON files on HPC (comma-separated, no spaces)
JSON_FILES="\
/leonardo_scratch/fast/EUHPC_D32_006/data/vgllm/train/spar_234k.json,\
/leonardo_scratch/fast/EUHPC_D32_006/data/vgllm/train/llava_hound_64k.json"

# Root directory that contains all media (images / video frame dirs)
MEDIA_ROOT="/leonardo_scratch/fast/EUHPC_D32_006/data/vgllm/media"

# Output directory for HDF5 shards  (keep it on scratch/fast storage)
OUTPUT_DIR="/leonardo_scratch/fast/EUHPC_D32_006/data/vgllm/hdf5"

# ── HDF5 parameters ────────────────────────────────────────────────────────────
NUM_SHARDS=32       # Match or exceed your total dataloader workers during training
                    # Training uses 4 workers/GPU × 4 GPU/node × 2 nodes = 32 workers
                    # 32 shards → each shard is opened by ~1 worker on average

NUM_WORKERS=4      # Parallel conversion workers (= NUM_SHARDS for max throughput,
                    # capped at --cpus-per-task)

JPEG_QUALITY=95     # 95 is virtually lossless; use 85 to halve storage at minor quality cost

# ── Environment ────────────────────────────────────────────────────────────────
echo "=================================="
echo "  build_hdf5.sh  –  Job $SLURM_JOB_ID"
echo "=================================="
echo "Node:         $(hostname)"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "JSON_FILES:   $JSON_FILES"
echo "MEDIA_ROOT:   $MEDIA_ROOT"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
echo "NUM_SHARDS:   $NUM_SHARDS"
echo "NUM_WORKERS:  $NUM_WORKERS"
echo "JPEG_QUALITY: $JPEG_QUALITY"
echo ""

# Load conda / Python environment (same as training)
export PATH="$WORK/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate vgllmN

# Verify required packages
python -c "import h5py, PIL, tqdm, numpy; print('All packages OK')"

# ── Lustre stripe optimisation ─────────────────────────────────────────────────
# Stripe the output directory across 8 OSTs before writing any large files.
# This significantly improves parallel read throughput during training.
mkdir -p "$OUTPUT_DIR"
if command -v lfs &>/dev/null; then
    echo "Setting Lustre stripe on $OUTPUT_DIR …"
    lfs setstripe -c 8 "$OUTPUT_DIR" || echo "[WARN] lfs setstripe failed (non-fatal)"
else
    echo "[WARN] lfs command not found – skipping Lustre striping"
fi

mkdir -p logs/preprocess

# ── Dry run first to report what will be converted ────────────────────────────
echo ""
echo "=== DRY RUN (no data written) ==="
python scripts/preprocess/convert_to_hdf5.py \
    --json_files  "$JSON_FILES" \
    --media_root  "$MEDIA_ROOT" \
    --output_dir  "$OUTPUT_DIR" \
    --num_shards  "$NUM_SHARDS" \
    --num_workers "$NUM_WORKERS" \
    --quality     "$JPEG_QUALITY" \
    --dry_run

# ── Actual conversion ──────────────────────────────────────────────────────────
echo ""
echo "=== ACTUAL CONVERSION ==="
python scripts/preprocess/convert_to_hdf5.py \
    --json_files  "$JSON_FILES" \
    --media_root  "$MEDIA_ROOT" \
    --output_dir  "$OUTPUT_DIR" \
    --num_shards  "$NUM_SHARDS" \
    --num_workers "$NUM_WORKERS" \
    --quality     "$JPEG_QUALITY"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Shard file sizes ==="
du -sh "$OUTPUT_DIR"/shard_*.h5 2>/dev/null | head -10
echo ""
echo "=== metadata.json ==="
cat "$OUTPUT_DIR/metadata.json"

echo ""
echo "Done. HDF5 shards are ready at: $OUTPUT_DIR"
echo "Pass to training with:  --use_hdf5 true  --hdf5_path $OUTPUT_DIR"
