#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name %name
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx8000:1

module --force purge
module load anaconda/3
conda init
conda activate /home/mila/f/floresl/miniconda3/envs/al

echo "Date:     $(date)"
echo "Hostname: $(hostname)"
python run_analyses.py "$@"