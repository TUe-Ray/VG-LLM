#!/usr/bin/env bash
#SBATCH --job-name=extract_train300k
#SBATCH --partition=lrd_all_serial
#SBATCH --account=euhpc_d32_006
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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
echo " ShareGPTVideo train_300k extraction"
echo " Running on: $(hostname)"
echo " SLURM job : ${SLURM_JOB_ID:-N/A}"
echo " CPUs      : ${SLURM_CPUS_PER_TASK:-1}"
echo " Source    : $SRC"
echo " Target    : $DST"
echo " Markers   : $MARK"
echo "========================================"

shopt -s nullglob
files=("$SRC"/chunk_*.tar.gz)
total="${#files[@]}"
count=0

if [ "$total" -eq 0 ]; then
  echo "[ERROR] No chunk_*.tar.gz found in: $SRC"
  exit 2
fi

# Prefer pigz for parallel gzip on the same node
THREADS="${SLURM_CPUS_PER_TASK:-4}"
if command -v pigz >/dev/null 2>&1; then
  export PIGZ="-p ${THREADS}"
  TAR_CMD_PREFIX=(tar --use-compress-program=pigz -xf)
  echo "[INFO] pigz found -> parallel gzip enabled (threads=${THREADS})"
else
  TAR_CMD_PREFIX=(tar -xzf)
  echo "[INFO] pigz not found -> using plain tar -xzf (no gzip parallelism)"
fi

########################
# Main loop
########################
for f in "${files[@]}"; do
  bn=$(basename "$f")
  count=$((count+1))
  marker="$MARK/$bn.done"

  if [ -f "$marker" ]; then
    echo "[SKIP $count/$total] $bn (already done)"
    continue
  fi

  echo "----------------------------------------"
  echo "[EXTRACT $count/$total] $bn"
  echo "Size      : $(du -h "$f" | cut -f1)"
  echo "Started at: $(date)"

  start=$(date +%s)

  # Use srun so the work is properly bound to the allocated node/cpus
  if srun --ntasks=1 --cpus-per-task="${THREADS}" "${TAR_CMD_PREFIX[@]}" "$f" -C "$DST"; then
    touch "$marker"
    end=$(date +%s)
    echo "[OK] $bn extracted in $((end-start)) sec"
  else
    echo "[ERROR] Failed extracting $bn"
    exit 1
  fi
done

echo "========================================"
echo " All chunks processed!"
echo " Finished at: $(date)"
echo "========================================"