declare -a languages=("afr" "fil" "hat" "deu")
declare -a modes=("sampling")
declare -a suffixes=("_5eps_500_full_shuffle")

for mode in "${modes[@]}"; do
  for language in "${languages[@]}"; do
    for suffix in "${suffixes[@]}"; do
      sbatch --job-name=als-$language-$mode-$suffix --output=/home/mila/f/floresl/active-learning-for-nlg/logs/eng-$language/output-$mode-$suffix --error=/home/mila/f/floresl/active-learning-for-nlg/logs/eng-$language/error-$mode-$suffix ./scripts/run_analyses_mila_cluster.sh --dataset data/eng_$language --num_samples 100 --num_rounds 500 --epochs 5 --experiment_mode $mode --suffix $suffix
    done
  done
done