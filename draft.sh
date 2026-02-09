#!/bin/bash

REPO_URL="https://github.com/TUe-Ray/VG-LLM.git"
REPO_DIR="$HOME/VG-LLM"
echo "Pulling latest changes..."
git -C "$REPO_DIR" pull


JOB_NAME="download_data_weight"          # job name
NUMBER_OF_GPUS=1                     # number of GPUs
NUMBER_OF_NODES=1                    # number of nodes
TIME_LIMIT="1:00:00"               # time limit: 1 hour
CPUS_PER_TASK=4                  # number of CPU cores per task
NTASKS_PER_NODE=$NUMBER_OF_GPUS                # number of tasks per node
QOS_NAME="boost_qos_dbg"            # normal/boost_qos_dbg/boost_qos_bprod/boost_qos_Iprod
ERROR_FILE="myJob.err"            # standard error file
OUTPUT_FILE="myJob.out"           # standard output file


export PATH="$WORK/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate vgllmN

FAST_DIR="$FAST"
HOSTNAME=$(hostname)
export WANDB_MODE=offline



# Check if base directory exists
if [ ! -d "$FAST_DIR" ]; then
    echo "Error: Base directory $FAST_DIR does not exist."
    exit 1
fi

case "$HOSTNAME" in
    *leonardo*)
        echo "Running on Leonardo cluster."
        DATASET_PATH="$FAST_DIR/data"
        module load cuda/12.6
        module load autoload cudnn
        module load profile/deeplrn
        sbatch/
esac

