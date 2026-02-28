#!/bin/bash
#SBATCH --job-name=GRAD
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4             
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod  
#SBATCH --qos=boose_qos_dbg   # normal/boost_qos_dbg/boost_qos_bprod/boost_qos_Iprod
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0

#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080


#SBATCH --exclusive
#INCOMPLETE: memory 獨占整個節點（不和別人搶 GPU），可以加 --exclusive；但如果你只用 1 GPU，通常不需要獨占整個節點
# 若要 4 GPU：把 --gpus-per-node=4 (以及視需要調 time / exclusive)

DATASETS="spar_234k,llava_hound_64k"
LR="1e-5"


JOB_TIME_LIMIT=$(squeue -j $SLURM_JOB_ID -h -o "%l")
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
echo "Memory per Node: $SLURM_MEM_PER_NODE"
echo "Output: $SLURM_STDOUT"
echo "Error: $SLURM_STDERR"
echo "Job Time Limit: $JOB_TIME_LIMIT"


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
OUT=$(nvidia-smi -L 2>&1) || {
  echo "[ERROR] nvidia-smi failed on $(hostname)"
  echo "$OUT"
  exit 1
}
if echo "$OUT" | grep -q "Driver/library version mismatch"; then
  echo "[ERROR] NVML mismatch on $(hostname)"
  echo "$OUT"
  exit 1
fi
echo "$OUT"

echo "[DEBUG] LD_LIBRARY_PATH after modules:"
echo "$LD_LIBRARY_PATH" | tr ":" "\n" | head -n 30

export PATH="$WORK/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate vgllmN

echo "[DEBUG] after conda activate:"
OUT=$(nvidia-smi -L 2>&1) || {
  echo "[ERROR] nvidia-smi failed on $(hostname) after conda"
  echo "$OUT"
  exit 1
}
if echo "$OUT" | grep -q "Driver/library version mismatch"; then
  echo "[ERROR] NVML mismatch on $(hostname) after conda"
  echo "$OUT"
  exit 1
fi
echo "$OUT"

echo "[DEBUG] LD_LIBRARY_PATH after conda:"
echo "$LD_LIBRARY_PATH" | tr ":" "\n" | head -n 30


# echo "==== multi-node NVML sanity check ===="
# srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --export=ALL bash -lc '
#   echo "===== $(hostname) ====="
#   which nvidia-smi || true
#   nvidia-smi -L || true
#   echo "--- /proc/driver/nvidia/version ---"
#   cat /proc/driver/nvidia/version 2>/dev/null | head -n 5 || true
#   echo "--- LD_LIBRARY_PATH (top) ---"
#   echo "$LD_LIBRARY_PATH" | tr ":" "\n" | head -n 20
# '
# echo "======================================"
echo "======================================"
echo " Per-node NVML health check"
echo "======================================"

# 展開本次 allocation 的 node 清單
NODE_LIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

for NODE in $NODE_LIST; do
  echo "----- Checking $NODE -----"

  OUT=$(srun -N1 -n1 -w "$NODE" bash -lc 'nvidia-smi -L' 2>&1)
  RET=$?

  if [ $RET -ne 0 ]; then
    echo "[ERROR] nvidia-smi failed on $NODE"
    echo "$OUT"
    echo "Aborting job."
    exit 1
  fi

  if echo "$OUT" | grep -q "Driver/library version mismatch"; then
    echo "[ERROR] NVML mismatch detected on $NODE"
    echo "$OUT"
    echo "You may exclude it next time with:"
    echo "#SBATCH --exclude=$NODE"
    echo "Aborting job."
    exit 1
  fi

  echo "$OUT"
  echo "Node $NODE OK"
done

echo "All nodes passed NVML check."
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
#MASTER_PORT=29500


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

export OMP_NUM_THREADS=2

export MASTER_ADDR MASTER_PORT

echo "[DDP] MASTER_ADDR=$MASTER_ADDR"
echo "[DDP] MASTER_PORT=$MASTER_PORT"
echo "[DDP] NNODES=$NNODES NODE_RANK=$NODE_RANK"
echo "[DDP] NPROC_PER_NODE=$NPROC_PER_NODE WORLD_SIZE=$WORLD_SIZE"
echo "[DDP] OMP_NUM_THREADS=$OMP_NUM_THREADS"

# ======================
# Paths / Config (從 train_sr.sh 來的參數，改成你自己的)
# ======================
MODEL_PATH="$FAST/hf_models/qwen2_5"  # [ModelArguments] Pretrained model path
GEOMETRY_ENCODER_TYPE="vggt"          # INCOMPLETE: Later "pi3"
GEOMETRY_ENCODER_PATH="$FAST/hf_models/vggt" #INCOMPLETE: download pi3

OUTPUT_DIR="$FAST/hf_models/${SLURM_JOB_NAME}/checkpoints"                   # Directory for saving checkpoints
CACHE_DIR="$FAST/hf_models/${SLURM_JOB_NAME}/cache"                        # [TrainingArguments] Cache directory for models
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"


export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0
export WANDB_DIR="$WORK/wandb"    
export WANDB_CACHE_DIR="$WORK/wandb_cache"
export WANDB_CONFIG_DIR="$WORK/wandb_config"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"



# 你可以用環境變數調整：
# PER_DEVICE_BS：每張卡的 micro-batch（預設 1，跟你原本一致）
# TOTAL_BATCH_SIZE：你想要的 global batch（預設：等於 WORLD_SIZE * PER_DEVICE_BS）
#TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-$((WORLD_SIZE * PER_DEVICE_BS))}"

# PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
# TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-64}"  # INCOMPLETE: 先重現實驗

# # global_batch = WORLD_SIZE * PER_DEVICE_BS * GRAD_ACC
# # => GRAD_ACC = TOTAL_BATCH_SIZE / (WORLD_SIZE * PER_DEVICE_BS)
# denom=$((WORLD_SIZE * PER_DEVICE_BS))
# GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / denom))
# if [ "$GRADIENT_ACCUMULATION_STEPS" -lt 1 ]; then
#   echo "[WARN] TOTAL_BATCH_SIZE($TOTAL_BATCH_SIZE) < WORLD_SIZE*PER_DEVICE_BS($denom). Set GRAD_ACC=1"
#   GRADIENT_ACCUMULATION_STEPS=1
# fi

PER_DEVICE_BS=1
TOTAL_BATCH_SIZE=64

denom=$((WORLD_SIZE * PER_DEVICE_BS))

if (( TOTAL_BATCH_SIZE % denom != 0 )); then
  echo "[ERROR] TOTAL_BATCH_SIZE($TOTAL_BATCH_SIZE) not divisible by WORLD_SIZE*PER_DEVICE_BS($denom)"
  echo "This would change the effective global batch size."
  exit 1
fi

GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / denom))
GRADIENT_ACCUMULATION_STEPS=8

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
  for i in $(seq 1 2); do
    echo "[$(hostname)] $(date +%H:%M:%S) $(free -h | awk "/Mem:/ {print \$3\"/\"\$2\" used, avail=\"\$7}")"
    sleep 3
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
      --save_steps 200 \
      --save_total_limit 2 \
      --deepspeed "scripts/zero2_opt.json" \
      --gradient_checkpointing \
      --dataloader_num_workers 4 \
      --group_by_modality_length true \
      --seed 0 \
      --report_to "wandb" \
      --use_geometry_encoder true \
      --geometry_encoder_type "$GEOMETRY_ENCODER_TYPE" \
      --geometry_encoder_path "$GEOMETRY_ENCODER_PATH" \
      --feature_fusion_method "add" \
      --use_hdf5 True \
      --hdf5_path "/leonardo_scratch/fast/EUHPC_D32_006/data/vgllm/hdf5/train.hdf5" \
  2>&1 | tee "$OUTPUT_DIR/train.log"