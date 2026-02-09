#!/usr/bin/env bash

#SBATCH --job-name=extract_train300k
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=30000M
#SBATCH --array=0-15%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err



set -euo pipefail

########################
# Config
########################
FAST=/leonardo_scratch/fast/EUHPC_D32_006
VGLLM_DATA_ROOT="$FAST/vgllm/data"
SRC="$FAST/vgllm/downloads/ShareGPTVideo/train_video_and_instruction/train_300k"
DST="$VGLLM_DATA_ROOT/media/llava_hound/frames"
MARK="$VGLLM_DATA_ROOT/.extract_markers/train_300k"

########################
# Prep
########################
mkdir -p "$DST" "$MARK" logs

echo "========================================"
echo " ShareGPTVideo train_300k extraction (ARRAY)"
echo " Host      : $(hostname)"
echo " Job ID    : ${SLURM_JOB_ID:-N/A}"
echo " Array job : ${SLURM_ARRAY_JOB_ID:-N/A}"
echo " Task ID   : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo " CPUs      : ${SLURM_CPUS_PER_TASK:-1}"
echo " Source    : $SRC"
echo " Target    : $DST"
echo " Markers   : $MARK"
echo "========================================"

# Build file list (sorted, stable)
shopt -s nullglob
mapfile -t files < <(ls -1 "$SRC"/chunk_*.tar.gz | sort)

total="${#files[@]}"
if [ "$total" -eq 0 ]; then
  echo "[ERROR] No chunk_*.tar.gz found in: $SRC"
  exit 2
fi

idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
if [ "$idx" -ge "$total" ]; then
  echo "[INFO] Task $idx >= total files $total; nothing to do."
  exit 0
fi

f="${files[$idx]}"
bn="$(basename "$f")"
marker="$MARK/$bn.done"

echo "[TASK] index=$idx / total=$total"
echo "[FILE] $bn"
echo "Size      : $(du -h "$f" | cut -f1)"
echo "Started at: $(date)"

if [ -f "$marker" ]; then
  echo "[SKIP] $bn (already done)"
  exit 0
fi

# choose decompressor
TAR_CMD=(tar -xzf "$f" -C "$DST")
if command -v pigz >/dev/null 2>&1; then
  THREADS="${SLURM_CPUS_PER_TASK:-4}"
  export PIGZ="-p ${THREADS}"
  TAR_CMD=(tar --use-compress-program=pigz -xf "$f" -C "$DST")
  echo "[INFO] pigz found -> using parallel gzip (threads=${THREADS})"
else
  echo "[INFO] pigz not found -> using plain tar -xzf"
fi

start=$(date +%s)
"${TAR_CMD[@]}"
touch "$marker"
end=$(date +%s)

echo "[OK] $bn extracted in $((end-start)) sec"
echo "Finished at: $(date)"