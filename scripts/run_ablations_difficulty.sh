declare -a datasets=("eng_afr" "eng_deu" "eng_fil") # "gsm8k" "squad" "math" "hotpotqa"
declare -a models=("gemma" "llama" "t5") # "bart"
declare -a seeds=("16" "27" "38")

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        sbatch --job-name=$dataset-$model-$seed --output=/home/mila/f/floresl/active-learning-assumptions/logs/$dataset-$model/output-$seed --error=/home/mila/f/floresl/active-learning-assumptions/logs/$dataset-$model/error-$seed ./scripts/run_ablations_mila_cluster.sh --dataset data/$dataset --model_name $model --epochs 200 --seed $seed --finetune_size 2000 --wandb_off
    done
  done
done