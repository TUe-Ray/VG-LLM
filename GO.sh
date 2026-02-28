#!/bin/bash

git pull
Job="train_4b.sh"

echo "======================================"
echo " run $Job "
echo "======================================"

sbatch $Job
