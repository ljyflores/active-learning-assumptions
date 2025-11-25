declare -a languages=("hat") # "afr" "deu" "fil"

for language in "${languages[@]}"; do
  sbatch --job-name=als-$language-sampling-var-decomp --output=/home/mila/f/floresl/active-learning-for-nlg/logs/eng-$language/output-sampling-var-decomp --error=/home/mila/f/floresl/active-learning-for-nlg/logs/eng-$language/error-sampling-var-decomp ./scripts/run_analyses_mila_cluster.sh --dataset data/eng_$language --num_samples 100 --num_rounds 5 --num_reps 20 --epochs 5 --experiment_mode sampling --shuffle_across_reps --use_test_sample --suffix var_decomp
done
