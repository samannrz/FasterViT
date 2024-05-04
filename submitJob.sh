#!/bin/bash

##Resource Request

#SBATCH --job-name FastAll
#SBATCH --output=stdout
#SBATCH --error=sdterr
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --gpus-per-task=1 # number of gpus per task
#SBATCH --cpus-per-task=3 # number of gpus per task

#SBATCH --mem-per-gpu=20000M # Memory allocated per gpu
#SBATCH --time=0-24:00:00  ## time for analysis (day-hour:min:sec)

## Run the script
srun python train.py --epochs 300 --batch 10 --scheduler