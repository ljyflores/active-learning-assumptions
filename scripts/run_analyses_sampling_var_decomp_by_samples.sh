declare -a languages=("eng_afr" "eng_fil" "eng_deu")
declare -a models=("t5" "llama" "gemma")
declare -a num_datapoints=(100 500)

for model in "${models[@]}"; do
  for language in "${languages[@]}"; do
    for num_datapoint in "${num_datapoints[@]}"; do
      sbatch --job-name=vdc-$language-$model-$num_datapoint --output=/home/mila/f/floresl/active-learning-assumptions/logs/$language/output-$model-$num_datapoint-vdc --error=/home/mila/f/floresl/active-learning-assumptions/logs/$language/error-$model-$num_datapoint-vdc ./scripts/run_analyses_mila_cluster.sh --dataset data/$language --model_name $model --num_samples 10 --num_datapoints_per_sample $num_datapoint --num_shuffles_per_sample 10 --epochs 200
    done
  done
done