#!/bin/bash

git pull
Job="debug.sh"

echo "======================================"
echo " run $Job "
echo "======================================"

sbatch $Job