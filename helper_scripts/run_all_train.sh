
dataset=$1
search_port=${2:-7502}
trained_port=${3:-8001}

bash helper_scripts/run_grpo.sh $dataset qwen 7 $search_port $trained_port
bash helper_scripts/run_grpo.sh $dataset qwen 3 $search_port $trained_port
bash helper_scripts/run_grpo.sh $dataset llama 8 $search_port $trained_port