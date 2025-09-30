set -e
set -o pipefail

dataset=$1
model=$2
size=$3
search_port=${4:-7501}
trained_port=${5:-8000}
retriever_type=${6:-"colbert"}
sft_model=${7:-"m5"}

# Function to cleanup all background processes
cleanup() {
    local exit_code=${1:-1}
    echo "Cleaning up processes..."
    
    # First try graceful termination, then force kill
    local pids_to_kill=()
    
    if [[ -n "$colbert_pid" ]]; then
        echo "Killing ColBERT server (PID: $colbert_pid)"
        pids_to_kill+=($colbert_pid)
    fi
    if [[ -n "$answer_pid" ]]; then
        echo "Killing answer server (PID: $answer_pid)"
        pids_to_kill+=($answer_pid)
    fi
    if [[ -n "$trained_pid" ]]; then
        echo "Killing trained server (PID: $trained_pid)"
        pids_to_kill+=($trained_pid)
    fi
    if [[ -n "$train_pid" ]]; then
        echo "Killing train evaluation process (PID: $train_pid)"
        pids_to_kill+=($train_pid)
    fi
    
    # Kill processes by PID
    if [[ ${#pids_to_kill[@]} -gt 0 ]]; then
        echo "Sending SIGTERM to processes: ${pids_to_kill[*]}"
        kill ${pids_to_kill[*]} 2>/dev/null || true
        sleep 3
        echo "Force killing processes: ${pids_to_kill[*]}"
        kill -9 ${pids_to_kill[*]} 2>/dev/null || true
    fi
    
    # Kill processes by name pattern
    echo "Killing any remaining train_grpo.py processes..."
    pkill -f "train_grpo.py" 2>/dev/null || true
    sleep 1
    pkill -9 -f "train_grpo.py" 2>/dev/null || true
    
    # Kill processes by port
    echo "Killing processes using ports $search_port and $trained_port..."
    local port_pids
    port_pids=$(lsof -ti:$search_port 2>/dev/null || true)
    if [[ -n "$port_pids" ]]; then
        echo "Killing processes on port $search_port: $port_pids"
        kill -9 $port_pids 2>/dev/null || true
    fi
    
    port_pids=$(lsof -ti:$trained_port 2>/dev/null || true)
    if [[ -n "$port_pids" ]]; then
        echo "Killing processes on port $trained_port: $port_pids"
        kill -9 $port_pids 2>/dev/null || true
    fi
    
    # Kill any remaining VLLM or ColBERT processes
    echo "Killing any remaining VLLM and ColBERT processes..."
    pkill -f "vllm-serve" 2>/dev/null || true
    sleep 1
    pkill -9 -f "vllm-serve" 2>/dev/null || true
    
    pkill -f "serve_colbert" 2>/dev/null || true
    sleep 1
    pkill -9 -f "serve_colbert" 2>/dev/null || true
    
    echo "Cleanup completed."
    exit $exit_code
}

# Function to check if a port is free
check_port_free() {
    local port=$1
    local port_name=$2
    if lsof -ti:$port >/dev/null 2>&1; then
        echo "Warning: Port $port ($port_name) is already in use. Cleaning up..."
        local port_pids=$(lsof -ti:$port 2>/dev/null || true)
        if [[ -n "$port_pids" ]]; then
            echo "Killing processes on port $port: $port_pids"
            kill -9 $port_pids 2>/dev/null || true
            sleep 2
        fi
        
        # Check again after cleanup
        if lsof -ti:$port >/dev/null 2>&1; then
            echo "Error: Failed to free port $port ($port_name). Exiting."
            exit 1
        else
            echo "Port $port ($port_name) is now free."
        fi
    else
        echo "Port $port ($port_name) is free."
    fi
}

# Function to handle interrupt signal (Ctrl+C)
handle_interrupt() {
    echo "Received interrupt signal. Cleaning up processes..."
    cleanup 1
}

# Set up signal handler for Ctrl+C (SIGINT) and script exit
trap handle_interrupt SIGINT
trap 'cleanup $?' EXIT

echo "Starting evaluation for $dataset-$model-${size}B on ports: search=$search_port, trained=$trained_port"

# Clean up any existing processes first
echo "Performing initial cleanup of any existing processes..."
# Kill any existing VLLM or ColBERT processes
pkill -9 -f "vllm-serve" 2>/dev/null || true
pkill -9 -f "serve_colbert" 2>/dev/null || true
pkill -9 -f "train_grpo.py" 2>/dev/null || true

# Check and free ports if needed
check_port_free $search_port "search"
check_port_free $trained_port "trained"

echo "Starting fresh processes..."

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
  --model ../data/frugalrag/sft/${retriever_type}/${dataset}/qwen${size}b/${sft_model}_0.90/Qwen_Qwen2.5-${size}B-Instruct \
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 1 \
  --port $trained_port &
  trained_pid=$!

  echo "Waiting for VLLM servers to start..."
  sleep 120

  echo "Starting GRPO training for Qwen ${size}B model..."
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch -m src.train.train_grpo --config configs/grpo/${retriever_type}/${dataset}_qwen${size}b_${sft_model}_0.90.json --search_port $search_port --vllm_port $trained_port 
  train_pid=$!

elif [[ "$model" == "llama" ]]; then
  echo "Starting VLLM servers for Llama ${size}B model"
  # start vllm server for trained model
  CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model ../data/frugalrag/sft/${retriever_type}/${dataset}/llama${size}b/${sft_model}_0.90/meta-llama_Meta-Llama-3.1-${size}B-Instruct \
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 1 \
  --port $trained_port &
  trained_pid=$!

  echo "Waiting for VLLM servers to start..."
  sleep 120

  echo "Starting GRPO training for Llama ${size}B model..."
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch -m src.train.train_grpo --config configs/grpo/${retriever_type}/${dataset}_llama${size}b_${sft_model}_0.90.json --search_port $search_port --vllm_port $trained_port --format_coeff 0.25 
  train_pid=$!
  
fi

# Wait for the training process to complete
if [[ -n "$train_pid" ]]; then
    echo "Waiting for GRPO training to complete (PID: $train_pid)..."
    wait $train_pid
    train_exit_code=$?
    
    echo "Training process finished. Performing cleanup..."
    
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