declare -a languages=("eng_afr" "eng_fil" "eng_deu")
declare -a models=("t5" "llama" "gemma")

for model in "${models[@]}"; do
  for language in "${languages[@]}"; do
    sbatch --job-name=fg-$language-$model --output=/home/mila/f/floresl/active-learning-assumptions/logs/$language/output-$model-fg --error=/home/mila/f/floresl/active-learning-assumptions/logs/$language/error-$model-fg ./scripts/run_feature_gen_mila_cluster.sh --dataset data/$language --model_name $model
  done
done