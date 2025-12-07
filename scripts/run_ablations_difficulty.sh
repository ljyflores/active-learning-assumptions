declare -a datasets=("eng_afr" "gsm8k" "squad") # "eng_deu" "eng_fil" "math" "hotpotqa"
declare -a models=("llama" "bart" "t5")
declare -a seeds=("16") # "38" "27"

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        sbatch --job-name=$dataset-$model-$seed --output=/home/mila/f/floresl/active-learning-assumptions/logs/$dataset-$model/output-$seed --error=/home/mila/f/floresl/active-learning-assumptions/logs/$dataset-$model/error-$seed ./scripts/run_ablations_mila_cluster.sh --dataset data/$dataset --model_name $model --seed $seed
    done
  done
done