#!/bin/bash
#SBATCH --job-name=mulNode_Eval
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4             # 依你的叢集格式：也可能是 --gpus-per-node=1
#SBATCH --ntasks-per-node=1       # 通常 1 個 task，裡面用 torchrun 起多 GPU processes
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod  
#SBATCH --qos=boost_qos_dbg   # normal/boost_qos_dbg/boost_qos_bprod/boost_qos_Iprod
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159

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



export HF_HOME=/leonardo_scratch/fast/EUHPC_D32_006/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
# 強制離線（compute node 不能連外就該這樣）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 讓 datasets 不要一直想去網路 check
export HF_UPDATE_DOWNLOAD_COUNTS=0

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
# 多節點時，MASTER_ADDR 用第一個節點；單節點也 OK
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# 用固定或隨機都行；固定比較好 debug，隨機比較不容易撞 port
# MASTER_PORT=${MASTER_PORT:-29500}
# INCOMPLETE: shuffle for official training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
MASTER_PORT=29500



export LMMS_EVAL_LAUNCHER="accelerate"
export NCCL_NVLS_ENABLE=0

benchmark=vsibench # choices: [vsibench, cvbench, blink_spatial]
output_path=/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/$(date "+%Y%m%d_%H%M%S")
model_path=/leonardo_scratch/fast/EUHPC_D32_006/hf_models/vgllm-qa-vggt-8b


# ranks / world size
export MACHINE_RANK=$SLURM_NODEID
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
export NUM_MACHINES=$SLURM_JOB_NUM_NODES
export TOTAL_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))

srun --export=ALL \
    accelerate launch \
        --num_machines=$NUM_MACHINES \
        --num_processes=$TOTAL_PROCESSES \
        --machine_rank=$MACHINE_RANK \
        --main_process_ip=$MASTER_ADDR \
        --main_process_port=$MASTER_PORT \
        --mixed_precision=bf16 \
        -m lmms_eval \
            --model vgllm \
            --model_args pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800 \
            --tasks ${benchmark} \
            --batch_size 1 \
            --output_path $output_path

# INCOMPLETE: register my own model class for --model
# Note: Remember to add in in __init__.py