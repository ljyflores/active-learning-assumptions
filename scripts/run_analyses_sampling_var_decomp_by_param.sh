declare -a languages=("fil" "hat") # "afr" "deu"
declare -a batch_sizes=(8 16 32)
declare -a learning_rates=(1e-5 5e-5 1e-4)

for language in "${languages[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
      sbatch --job-name=als-$language-sampling-$bs-$lr --output=/home/mila/f/floresl/active-learning-for-nlg/logs/eng-$language/output-sampling-$bs-$lr --error=/home/mila/f/floresl/active-learning-for-nlg/logs/eng-$language/error-sampling-$bs-$lr ./scripts/run_analyses_mila_cluster.sh --dataset data/eng_$language --num_samples 100 --num_rounds 2 --num_reps 20 --epochs 5 --experiment_mode sampling --batch_size $bs --learning_rate $lr --shuffle_across_reps --use_test_sample
    done
  done
done
