#!/usr/bin/env bash

#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --job-name=cyp
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:A100:1

conda activate dali
cd ~/Crossformer

python3 run_cyp.py