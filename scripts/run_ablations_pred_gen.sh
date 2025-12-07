declare -a datasets=("eng_afr" "eng_deu" "eng_fil")
declare -a models=("llama" "gemma" "t5")
declare -a seeds=("16") # "38" "27"

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        sbatch --job-name=$dataset-$model-$seed --output=/home/mila/f/floresl/active-learning-assumptions/logs/$dataset-$model/output-$seed-pred --error=/home/mila/f/floresl/active-learning-assumptions/logs/$dataset-$model/error-$seed-pred ./scripts/run_ablations_mila_cluster.sh --dataset data/$dataset --model_name $model --seed $seed --epochs 10 --wandb_off --get_preds_only
    done
  done
done