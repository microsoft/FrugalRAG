set -e
set -o pipefail

dataset=$1
model=$2
size=$3
search_port=${4:-7501}
trained_port=${5:-8000}
retriever_type=${6:-"colbert"}

# Function to cleanup all background processes
cleanup() {
    local exit_code=${1:-1}
    echo "Cleaning up processes..."
    if [[ -n "$colbert_pid" ]]; then
        echo "Killing ColBERT server (PID: $colbert_pid)"
        kill $colbert_pid 2>/dev/null
    fi
    if [[ -n "$answer_pid" ]]; then
        echo "Killing answer server (PID: $answer_pid)"
        kill $answer_pid 2>/dev/null
    fi
    if [[ -n "$trained_pid" ]]; then
        echo "Killing trained server (PID: $trained_pid)"
        kill $trained_pid 2>/dev/null
    fi
    if [[ -n "$train_pid" ]]; then
        echo "Killing train evaluation process (PID: $train_pid)"
        kill $train_pid 2>/dev/null
    fi
    # Kill any remaining train_grpo.py processes
    echo "Killing any remaining train_grpo.py processes..."
    pkill -f "train_grpo.py" 2>/dev/null
    echo "Cleanup completed."
    exit $exit_code
}

# Function to handle interrupt signal (Ctrl+C)
handle_interrupt() {
    echo "Received interrupt signal. Cleaning up processes..."
    cleanup 1
}

# Set up signal handler for Ctrl+C (SIGINT)
trap handle_interrupt SIGINT

echo "Starting evaluation for $dataset-$model-${size}B on ports: search=$search_port, trained=$trained_port"

# Start ColBERT server in background
if [[ "$dataset" == "2wiki" || "$dataset" == "musique" ]]; then
  echo "Starting ColBERT server for $dataset on port $search_port"
  PORT=$search_port CUDA_VISIBLE_DEVICES=0 python -m src.search.serve_colbert --index_root ../data/ColBERT/colbert_scripts/experiments/wiki20M/indexes --index wiki20M.2bits --colbert_path ../data/leret/colbertv2.0 --collection_path ../data/data/wiki20M/collection.tsv &
  colbert_pid=$!
  echo "Waiting for ColBERT server to start..."
  sleep 60

elif [[ "$dataset" == "hotpot" ]]; then
  echo "Starting ColBERT server for $dataset on port $search_port"
  PORT=$search_port CUDA_VISIBLE_DEVICES=0 python -m src.search.serve_colbert --index_root ../data/leret/ --index wiki17.nbits.local --colbert_path ../data/leret/colbertv2.0 --collection_path ../data/wiki.abstracts.2017/collection.tsv &
  colbert_pid=$!

  echo "Waiting for ColBERT server to start..."
  sleep 60

fi


if [[ "$model" == "qwen" ]]; then
  echo "Starting VLLM servers for Qwen ${size}B model"
  # start vllm server for trained model
  CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model ../data/frugalrag/sft/${retriever_type}/${dataset}/qwen${size}b/m5_0.90/Qwen_Qwen2.5-${size}B-Instruct \
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 1 \
  --port $trained_port &
  trained_pid=$!

  echo "Waiting for VLLM servers to start..."
  sleep 120

  echo "Starting GRPO training for Qwen ${size}B model..."
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch -m src.train.train_grpo --config configs/grpo/${retriever_type}/${dataset}_qwen${size}b_m5_0.90.json --search_port $search_port --vllm_port $trained_port &
  train_pid=$!

elif [[ "$model" == "llama" ]]; then
  echo "Starting VLLM servers for Llama ${size}B model"
  # start vllm server for trained model
  CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model ../data/frugalrag/sft/${retriever_type}/${dataset}/llama${size}b/m5_0.90/meta-llama_Meta-Llama-3.1-${size}B-Instruct \
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 1 \
  --port $trained_port &
  trained_pid=$!

  echo "Waiting for VLLM servers to start..."
  sleep 120

  echo "Starting GRPO training for Llama ${size}B model..."
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch -m src.train.train_grpo --config configs/grpo/${retriever_type}/${dataset}_llama${size}b_m5_0.90.json --search_port $search_port --vllm_port $trained_port &
  train_pid=$!
  
fi

# Wait for the training process to complete
if [[ -n "$train_pid" ]]; then
    echo "Waiting for GRPO training to complete (PID: $train_pid)..."
    wait $train_pid
    train_exit_code=$?
    
    if [[ $train_exit_code -eq 0 ]]; then
        echo "Training completed successfully for $dataset-$model-${size}B"
        cleanup 0
    else
        echo "Training failed with exit code $train_exit_code for $dataset-$model-${size}B"
        cleanup $train_exit_code
    fi
else
    echo "No training process was started. Cleaning up..."
    cleanup 1
fi