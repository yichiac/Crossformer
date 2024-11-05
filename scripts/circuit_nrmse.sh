#!/usr/bin/env bash

#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --job-name=circuit_nrmse
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mail-user=yichia3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load anaconda/2023-Mar/3
source activate
conda activate dali

cd ~/Crossformer

python3 run_nrmse.py
