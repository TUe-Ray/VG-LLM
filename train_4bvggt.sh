#!/bin/bash
#SBATCH --job-name=lora_0.25data_crossattn
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=14:00:00
#SBATCH --partition=boost_usr_prod  
#SBATCH --qos=normal # normal/boost_qos_dbg/boost_qos_bprod/boost_qos_Iprod
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0868,lrdn0808,lrdn0182,lrdn0680,lrdn0831,lrdn0084,lrdn0088,lrdn0186
#SBATCH --exclusive

NOTE="train lora with 0.25 data, Cross-Attention fusion, vggt geometry encoder, 4k max length, 1 epoch, lr 5e-6, cosine scheduler, warmup 3%, seed 0"

echo "-------- Note --------"
echo "  note: $NOTE"

DATASETS="spar_234k,llava_hound_64k"
LR="5e-6"

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
# ======================
# Paths / Config (從 train_sr.sh 來的參數，改成你自己的)
# ======================
MODEL_PATH="$FAST/hf_models/qwen2_5_3b"  
GEOMETRY_ENCODER_TYPE="vggt"          
GEOMETRY_ENCODER_PATH="$FAST/hf_models/vggt" 

OUTPUT_DIR="$FAST/hf_models/train/${SLURM_JOB_NAME}/checkpoints"                   # Directory for saving checkpoints
CACHE_DIR="$FAST/hf_models/train/${SLURM_JOB_NAME}/cache"                        # [TrainingArguments] Cache directory for models
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

PER_DEVICE_BS=1
TOTAL_BATCH_SIZE=64
JOB_TIME_LIMIT=$(squeue -j $SLURM_JOB_ID -h -o "%l")

echo "=== Job Configuration ==="
echo "MODEL_PATH: $MODEL_PATH"
echo "GEOMETRY_ENCODER_TYPE: $GEOMETRY_ENCODER_TYPE"
echo "GEOMETRY_ENCODER_PATH: $GEOMETRY_ENCODER_PATH"
echo "PER_DEVICE_BS: $PER_DEVICE_BS"
echo "TOTAL_BATCH_SIZE: $TOTAL_BATCH_SIZE"

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
export PATH="$WORK/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate vgllmN
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
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
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
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0
export WANDB_DIR="$WORK/wandb"    
export WANDB_CACHE_DIR="$WORK/wandb_cache"
export WANDB_CONFIG_DIR="$WORK/wandb_config"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"






denom=$((WORLD_SIZE * PER_DEVICE_BS))
if (( TOTAL_BATCH_SIZE % denom != 0 )); then
  echo "[ERROR] TOTAL_BATCH_SIZE($TOTAL_BATCH_SIZE) not divisible by WORLD_SIZE*PER_DEVICE_BS($denom)"
  echo "This would change the effective global batch size."
  exit 1
fi
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / denom))
echo "[BATCH] PER_DEVICE_BS=$PER_DEVICE_BS"
echo "[BATCH] TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
echo "[BATCH] GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"

# PyTorch CUDA memory management optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ======================
# Define all parameters using associative arrays
# ======================

declare -A MODEL_ARGS=(
  [model_name_or_path]="$MODEL_PATH"
  [tune_mm_llm]="True"
  [tune_mm_mlp]="False"
  [tune_mm_vision]="False"
  [use_geometry_encoder]="true"
  [geometry_encoder_type]="$GEOMETRY_ENCODER_TYPE"
  [geometry_encoder_path]="$GEOMETRY_ENCODER_PATH"
  [feature_fusion_method]="cross_attention"
  [geometry_encoder_random_init]="false"
  #LORA 相關參數
  [use_lora]="true"
  [lora_r]="64"
  [lora_alpha]="128"
  [lora_dropout]="0.05"
  [lora_bias]="none"
  [lora_target_modules]="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

)

declare -A DATA_ARGS=(
  [dataset_use]="$DATASETS"
  [data_flatten]="False"
  [max_pixels]=$((576*28*28))
  [min_pixels]=$((16*28*28))
  [base_interval]="2"
  [video_max_frames]="8"
  [video_min_frames]="4"
  [video_max_frame_pixels]=$((1664*28*28))
  [video_min_frame_pixels]=$((256*28*28))
  [use_hdf5]="false"
  [hdf5_path]="None"
  [hdf5_num_shards]="32"
  [dataset_fraction]="0.25"
)

declare -A TRAINING_ARGS=(
  [run_name]="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
  [output_dir]="$OUTPUT_DIR"
  [cache_dir]="$CACHE_DIR"
  [bf16]="true"
  [per_device_train_batch_size]="$PER_DEVICE_BS"
  [gradient_accumulation_steps]="$GRADIENT_ACCUMULATION_STEPS"
  [learning_rate]="$LR"
  [mm_projector_lr]="1e-5"
  [vision_tower_lr]="1e-6"
  [optim]="adamw_torch"
  [model_max_length]="12800"
  [num_train_epochs]="1"
  [warmup_ratio]="0.03"
  [lr_scheduler_type]="cosine"
  [weight_decay]="0.01"
  [logging_steps]="50"
  [save_steps]="200"
  [save_total_limit]="1"
  [deepspeed]="scripts/zero2_opt.json"
  [gradient_checkpointing]="true"
  [dataloader_num_workers]="6"
  [group_by_modality_length]="true"
  [seed]="0"
  [report_to]="wandb"
)


# ======================
# Print configuration (from arrays)
# ======================
echo "========================================"
echo " Training Configuration"
echo "========================================"

echo "--- ModelArguments ---"
for key in "${!MODEL_ARGS[@]}"; do
  printf "  %-35s %s\n" "$key:" "${MODEL_ARGS[$key]}"
done

echo ""
echo "--- DataArguments ---"
for key in "${!DATA_ARGS[@]}"; do
  printf "  %-35s %s\n" "$key:" "${DATA_ARGS[$key]}"
done

echo ""
echo "--- TrainingArguments ---"
for key in "${!TRAINING_ARGS[@]}"; do
  printf "  %-35s %s\n" "$key:" "${TRAINING_ARGS[$key]}"
done

# ======================
# Build parameter arrays for torchrun
# ======================
declare -a TORCHRUN_ARGS=()

for key in "${!MODEL_ARGS[@]}"; do
  TORCHRUN_ARGS+=("--${key//_/-}")
  TORCHRUN_ARGS+=("${MODEL_ARGS[$key]}")
done

for key in "${!DATA_ARGS[@]}"; do
  TORCHRUN_ARGS+=("--${key//_/-}")
  TORCHRUN_ARGS+=("${DATA_ARGS[$key]}")
done

for key in "${!TRAINING_ARGS[@]}"; do
  TORCHRUN_ARGS+=("--${key//_/-}")
  TORCHRUN_ARGS+=("${TRAINING_ARGS[$key]}")
done

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
    "${TORCHRUN_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_DIR/train.log"