#!/bin/bash
#SBATCH --job-name=Evaluate_Reproduce_Exp
#SBATCH --nodes=1
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


module load cuda/12.6
module load cudnn
module load profile/deeplrn

export PATH="$WORK/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate vgllmN

export LMMS_EVAL_LAUNCHER="accelerate"
export NCCL_NVLS_ENABLE=0

benchmark=vsibench # choices: [vsibench, cvbench, blink_spatial]
output_path=/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/$(date "+%Y%m%d_%H%M%S")
model_path=/leonardo_scratch/fast/EUHPC_D32_006/hf_models/vgllm-qa-vggt-8b

accelerate launch --num_processes=4 -m lmms_eval \
    --model vgllm \
    --model_args pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800 \
    --tasks ${benchmark} \
    --batch_size 1 \
    --output_path $output_path