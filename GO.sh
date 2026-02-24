#!/bin/bash

git pull
Job="train.sh"

echo "======================================"
echo " run $Job "
echo "======================================"

sbatch $Job
