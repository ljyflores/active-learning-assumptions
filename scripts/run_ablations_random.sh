declare -a datasets=("eng_afr" "eng_deu" "eng_fil" "gsm8k" "math" "squad" "hotpotqa")
declare -a models=("llama" "bart" "t5")
declare -a seeds=("38" "27") # "16"

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        sbatch --job-name=$dataset-$model-$seed-rdm --output=/home/mila/f/floresl/active-learning-assumptions/logs/$dataset-$model/output-$seed-rdm --error=/home/mila/f/floresl/active-learning-assumptions/logs/$dataset-$model/error-$seed-rdm ./scripts/run_ablations_mila_cluster.sh --dataset data/$dataset --model_name $model --seed $seed --random_case --wandb_off
    done
  done
done