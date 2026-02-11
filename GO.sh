#!/bin/bash

git pull
Job="debug.sh"
echo "Launching training job with sbatch $Job"

sbatch $Job