cat > extract_train_300k.sh << 'EOF'
#!/usr/bin/env bash
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
mkdir -p "$DST" "$MARK"

total=$(ls "$SRC"/chunk_*.tar.gz | wc -l)
count=0

echo "========================================"
echo " ShareGPTVideo train_300k extraction"
echo " Source : $SRC"
echo " Target : $DST"
echo " Chunks : $total"
echo "========================================"

########################
# Main loop
########################
for f in "$SRC"/chunk_*.tar.gz; do
  bn=$(basename "$f")
  count=$((count+1))
  marker="$MARK/$bn.done"

  if [ -f "$marker" ]; then
    echo "[SKIP $count/$total] $bn (already done)"
    continue
  fi

  echo "----------------------------------------"
  echo "[EXTRACT $count/$total] $bn"
  echo "Size: $(du -h "$f" | cut -f1)"
  echo "Started at: $(date)"

  start=$(date +%s)
  if tar -xzf "$f" -C "$DST"; then
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
echo "========================================"
EOF
