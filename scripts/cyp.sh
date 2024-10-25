#!/usr/bin/env bash

#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --job-name=cyp
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:A100:1
#SBATCH --output=.experiment-logs/o%j
#SBATCH --error=.experiment-logs/e%j
#SBATCH --mail-user=yichia3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load anaconda/2023-Mar/3
source activate
conda activate dali

cd ~/Crossformer

python3 run_cyp.py