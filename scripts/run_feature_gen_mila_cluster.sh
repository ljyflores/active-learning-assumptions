#!/bin/bash
#SBATCH --job-name %name
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4

module --force purge
module load anaconda/3
conda init
conda activate /home/mila/f/floresl/miniconda3/envs/al

echo "Date:     $(date)"
echo "Hostname: $(hostname)"
python run_feature_gen.py "$@"