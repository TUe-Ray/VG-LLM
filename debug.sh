#!/bin/bash
#SBATCH --job-name=OOM
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4             # 依你的叢集格式：也可能是 --gpus-per-node=1
#SBATCH --ntasks-per-node=1       # 通常 1 個 task，裡面用 torchrun 起多 GPU processes
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --partition=boost_usr_prod  
#SBATCH --qos=boost_qos_dbg    # normal/boost_qos_dbg/boost_qos_bprod/boost_qos_Iprod
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400


#INCOMPLETE: memory 獨占整個節點（不和別人搶 GPU），可以加 --exclusive；但如果你只用 1 GPU，通常不需要獨占整個節點
# 若要 4 GPU：把 --gpus-per-node=4 (以及視需要調 time / exclusive)



echo "=== SLURM Job Specifications ==="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Partition: $SLURM_JOB_PARTITION"
echo "QOS: $SLURM_JOB_QOS"
echo "Time Limit: $SLURM_TIME_LIMIT"
echo "Memory per Node: $SLURM_MEM_PER_NODE"
echo "Output: $SLURM_STDOUT"
echo "Error: $SLURM_STDERR"
echo "=============================="


set -euo pipefail

# ======================
# Cluster-specific modules (依你的 launch_training.sh 的想法補完整)
# ======================
HOSTNAME=$(hostname)
which nvidia-smi || true
nvidia-smi -L || true

module load cuda/12.6
module load cudnn
module load profile/deeplrn

echo "[DEBUG] after modules:"
nvidia-smi -L || true
echo "[DEBUG] LD_LIBRARY_PATH after modules:"
echo "$LD_LIBRARY_PATH" | tr ":" "\n" | head -n 30

export PATH="$WORK/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate vgllmN

echo "[DEBUG] after conda activate:"
nvidia-smi -L || true
echo "[DEBUG] LD_LIBRARY_PATH after conda:"
echo "$LD_LIBRARY_PATH" | tr ":" "\n" | head -n 30

echo "======================================"
echo " Multi-node NVML health check (fail-fast)"
echo "======================================"

BAD_NODES=$(srun --ntasks=$SLURM_JOB_NUM_NODES \
                 --ntasks-per-node=1 \
                 --export=ALL \
                 bash -lc '
echo "===== NODE $(hostname) ====="

OUT=$(nvidia-smi 2>&1 || true)
echo "$OUT"

echo "--- /proc/driver/nvidia/version ---"
cat /proc/driver/nvidia/version 2>/dev/null | head -n 5 || true

echo "--- LD_LIBRARY_PATH (top) ---"
echo "$LD_LIBRARY_PATH" | tr ":" "\n" | head -n 20

if echo "$OUT" | grep -q "Driver/library version mismatch"; then
    echo "[BAD] $(hostname)"
fi
' | grep "\[BAD\]" | awk '{print $2}' | tr '\n' ',' | sed 's/,$//')

echo "======================================"

if [ -n "${BAD_NODES:-}" ]; then
    echo ""
    echo "❌ NVML mismatch detected on nodes:"
    echo "   $BAD_NODES"
    echo ""
    echo "👉 You can exclude them next time with:"
    echo "   #SBATCH --exclude=$BAD_NODES"
    echo ""
    echo "Aborting before training."
    exit 1
fi

echo "✅ All nodes passed NVML check."
echo "======================================"




# ======================
# Distributed (Slurm-aware)
# ======================
# 多節點時，MASTER_ADDR 用第一個節點；單節點也 OK
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# 用固定或隨機都行；固定比較好 debug，隨機比較不容易撞 port
# MASTER_PORT=${MASTER_PORT:-29500}
# INCOMPLETE: shuffle for official training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
MASTER_PORT=29500


# NPROC_PER_NODE：用 Slurm 提供的 GPU 數，沒有就 fallback 到 nvidia-smi
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC_PER_NODE="$SLURM_GPUS_ON_NODE"
elif [ -n "${SLURM_GPUS_PER_NODE:-}" ]; then
  NPROC_PER_NODE="$SLURM_GPUS_PER_NODE"
else
  NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
fi

NNODES="${SLURM_JOB_NUM_NODES:-1}"
NODE_RANK=${SLURM_NODEID}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

export MASTER_ADDR MASTER_PORT

echo "[DDP] MASTER_ADDR=$MASTER_ADDR"
echo "[DDP] MASTER_PORT=$MASTER_PORT"
echo "[DDP] NNODES=$NNODES NODE_RANK=$NODE_RANK"
echo "[DDP] NPROC_PER_NODE=$NPROC_PER_NODE WORLD_SIZE=$WORLD_SIZE"

# ======================
# Paths / Config (從 train_sr.sh 來的參數，改成你自己的)
# ======================
MODEL_PATH="$FAST/hf_models/qwen2_5"  # [ModelArguments] Pretrained model path
GEOMETRY_ENCODER_TYPE="vggt"
GEOMETRY_ENCODER_PATH="$FAST/hf_models/vggt"

OUTPUT_DIR="$FAST/hf_models/checkpoints"                   # Directory for saving checkpoints
CACHE_DIR="$FAST/hf_models/cache"                        # [TrainingArguments] Cache directory for models
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"


export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0
export WANDB_DIR="$WORK/wandb"    
export WANDB_CACHE_DIR="$WORK/wandb_cache"
export WANDB_CONFIG_DIR="$WORK/wandb_config"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

DATASETS="spar_234k,llava_hound_64k"
DATASETS="llava_hound_64k"

LR="1e-5"

# 你可以用環境變數調整：
# PER_DEVICE_BS：每張卡的 micro-batch（預設 1，跟你原本一致）
# TOTAL_BATCH_SIZE：你想要的 global batch（預設：等於 WORLD_SIZE * PER_DEVICE_BS）
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-$((WORLD_SIZE * PER_DEVICE_BS))}"


# global_batch = WORLD_SIZE * PER_DEVICE_BS * GRAD_ACC
# => GRAD_ACC = TOTAL_BATCH_SIZE / (WORLD_SIZE * PER_DEVICE_BS)
denom=$((WORLD_SIZE * PER_DEVICE_BS))
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / denom))
if [ "$GRADIENT_ACCUMULATION_STEPS" -lt 1 ]; then
  echo "[WARN] TOTAL_BATCH_SIZE($TOTAL_BATCH_SIZE) < WORLD_SIZE*PER_DEVICE_BS($denom). Set GRAD_ACC=1"
  GRADIENT_ACCUMULATION_STEPS=1
fi

echo "[BATCH] PER_DEVICE_BS=$PER_DEVICE_BS"
echo "[BATCH] TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
echo "[BATCH] GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"

# PyTorch CUDA memory management optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# INCOMPLETE: debug info
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO



echo "========================================"
echo " Pre-flight check"

srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --export=ALL bash -lc '
  echo "===== NODE $(hostname) ====="
  free -h
  grep -E "MemTotal|MemAvailable" /proc/meminfo
  nvidia-smi -L
  echo "cgroup memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo NA)"
'



#=======================
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --export=ALL bash -lc '
  echo "[$(hostname)] start mem monitor"
  for i in $(seq 1 120); do
    echo "[$(hostname)] $(date +%H:%M:%S) $(free -h | awk "/Mem:/ {print \$3\"/\"\$2\" used, avail=\"\$7}")"
    sleep 1
  done
' &



# ======================
# Launch training
# ======================
# ✅ 建議用 srun 包 torchrun：Slurm 會幫你把 rank/world size 管好
# 這裡每個 node 啟 1 個 task（--ntasks-per-node=1），task 裡 torchrun 再起 NPROC_PER_NODE 個進程
echo "========================================"
echo " Starting training"
srun --export=ALL \
  torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    src/qwen_vl/train/train_qwen.py \
      --run_name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
      --model_name_or_path "$MODEL_PATH" \
      --tune_mm_llm True \
      --tune_mm_vision False \
      --tune_mm_mlp False \
      --dataset_use "$DATASETS" \
      --output_dir "$OUTPUT_DIR" \
      --cache_dir "$CACHE_DIR" \
      --bf16 \
      --per_device_train_batch_size "$PER_DEVICE_BS" \
      --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
      --learning_rate "$LR" \
      --mm_projector_lr 1e-5 \
      --vision_tower_lr 1e-6 \
      --optim adamw_torch \
      --model_max_length 12800 \
      --data_flatten False \
      --max_pixels $((576*28*28)) \
      --min_pixels $((16*28*28)) \
      --base_interval 2 \
      --video_max_frames 8 \
      --video_min_frames 4 \
      --video_max_frame_pixels $((1664*28*28)) \
      --video_min_frame_pixels $((256*28*28)) \
      --num_train_epochs 1 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --weight_decay 0.01 \
      --logging_steps 50 \
      --save_steps 1000 \
      --save_total_limit 1 \
      --deepspeed "scripts/zero2_opt.json" \
      --gradient_checkpointing \
      --dataloader_num_workers 0 \
      --group_by_modality_length true \
      --seed 0 \
      --report_to "wandb" \
      --use_geometry_encoder true \
      --geometry_encoder_type "$GEOMETRY_ENCODER_TYPE" \
      --geometry_encoder_path "$GEOMETRY_ENCODER_PATH" \
      --feature_fusion_method "add" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
