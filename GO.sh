#!/bin/bash

git pull
Job="debug.sh"
echo "Launching training job..."

sbatch $Job