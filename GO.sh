#!/bin/bash

git pull
Job="Run.sh"

echo "======================================"
echo " run $Job "
echo "======================================"

sbatch $Job
