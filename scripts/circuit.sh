#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=dali
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=circuit
#SBATCH --output=experiment-logs/o%j
#SBATCH --error=experiment-logs/e%j
#SBATCH --mail-user=yichia3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

cd ~/crossformer
python run.py --n 5 --grid 10 --k 5

