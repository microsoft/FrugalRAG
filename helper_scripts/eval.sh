set -e
set -o pipefail

dataset=$1
model=$2
size=$3
search_port=${4:-8000}
answer_port=${5:-7502}
trained_port=${6:-7501}
device_map_a=${7:-"0"}
device_map_b=${8:-"1"}
type=${9:-"sft"}
save_name=${10:-"m5"}
m=${11:-"5"}
rm=${12:-"wiki5M"}
ndocs=${13:-"3"}

# Function to cleanup all background processes
cleanup() {
    echo "Received interrupt signal. Cleaning up processes..."
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
    if [[ -n "$eval_train_pid" ]]; then
        echo "Killing train evaluation process (PID: $eval_train_pid)"
        kill $eval_train_pid 2>/dev/null
    fi
    if [[ -n "$eval_test_pid" ]]; then
        echo "Killing test evaluation process (PID: $eval_test_pid)"
        kill $eval_test_pid 2>/dev/null
    fi
    # # Kill any remaining eval_mp.py processes
    # echo "Killing any remaining eval_mp.py processes..."
    # pkill -f "eval_mp.py" 2>/dev/null
    echo "Cleanup completed. Exiting..."
    exit 1
}

# Set up signal handler for Ctrl+C (SIGINT)
trap cleanup SIGINT

echo "Starting evaluation for $dataset-$model-${size}B on ports: search=$search_port, answer=$answer_port, trained=$trained_port"

# Start ColBERT server in background
if [[ "$dataset" == "2wiki" || "$dataset" == "musique" ]]; then
  echo "Starting ColBERT server for $dataset on port $search_port"
  
  if [[ "$dataset" == "musique" ]]; then
    testfile="dgslibisey/MuSiQue"
  elif [[ "$dataset" == "2wiki" ]]; then
    testfile="../data/2wikimhqa/data/dev.json"
  fi

  PORT=$search_port CUDA_VISIBLE_DEVICES=$device_map_a python -m src.search.serve_colbert --index_root ../data/ColBERT/colbert_scripts/experiments/wiki20M/indexes --index wiki20M.2bits --colbert_path ../data/leret/colbertv2.0 --collection_path ../data/data/wiki20M/collection.tsv &
  colbert_pid=$!
  
  echo "Waiting for ColBERT server to start..."
  sleep 60

elif [[ "$dataset" == "hotpot" ]]; then
  echo "Starting ColBERT server for $dataset on port $search_port"

  testfile="../data/data/eval-v2/hopo/hotpot_dev_distractor_v1.json"

  PORT=$search_port CUDA_VISIBLE_DEVICES=$device_map_a python -m src.search.serve_colbert --index_root ../data/leret/ --index wiki17.nbits.local --colbert_path ../data/leret/colbertv2.0 --collection_path ../data/wiki.abstracts.2017/collection.tsv &
  colbert_pid=$!

  echo "Waiting for ColBERT server to start..."
  sleep 30

elif [[ "$dataset" == "bcplus" ]]; then
  testfile="../BrowseComp-Plus/data/test.jsonl"

fi

if [[ "$model" == "qwen" ]]; then
  echo "Starting VLLM servers for Qwen ${size}B model"
  # start vllm server for answer model
  CUDA_VISIBLE_DEVICES=$device_map_b python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-${size}B-Instruct \
    --gpu-memory-utilization 0.70 \
    --tensor-parallel-size 1 \
    --port $answer_port &
  answer_pid=$!

  if [[ "${type}" == "grpo" ]]; then
    model_path="../data/frugalrag/${type}/colbert/${dataset}/qwen${size}b/${save_name}_0.90/final"
  elif [[ "${type}" == "sft" ]]; then
    model_path="../data/frugalrag/${type}/colbert/${dataset}/qwen${size}b/${save_name}_0.90/Qwen_Qwen2.5-${size}B-Instruct"
  elif [[ "${type}" == "base" ]]; then
    model_path="Qwen/Qwen2.5-${size}B-Instruct"
  fi

  echo "using model path: ${model_path}"

  # start vllm server for trained model
  CUDA_VISIBLE_DEVICES=$device_map_a python -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --gpu-memory-utilization 0.70 \
    --tensor-parallel-size 1 \
    --port $trained_port &
  trained_pid=$!

  # Wait for VLLM servers to start
  echo "Waiting for VLLM servers to start..."
  sleep 120

  # on the train data
  if [ -e "datasets/${dataset}_1000.json" ]; then
    echo "Running evaluation on train data..."
    python -m src.evaluation.eval_mp \
      --max_iters $m \
      --ndocs $ndocs \
      --rm "${rm}" \
      --model_name_or_path "${model_path}" \
      --output_path "../data/frugalrag/eval/${type}/colbert/${dataset}/qwen${size}b/train1000_${save_name}_nofinish/" \
      --prompt_path "src/prompts/colbert/${dataset}/qwen${size}b/m5_nofinish/ans/bootstrapped_1.json" \
      --answer_model "Qwen/Qwen2.5-${size}B-Instruct" \
      --port $trained_port $answer_port \
      --search_port $search_port \
      --dataset_name "${dataset}" \
      --input_file "datasets/${dataset}_1000.json" \
      --no_finish True &
    eval_train_pid=$!
  fi

  # on the test data
  echo "Running evaluation on test data..."
  python -m src.evaluation.eval_mp \
    --max_iters $m \
    --ndocs $ndocs \
    --rm "${rm}" \
    --model_name_or_path "${model_path}" \
    --output_path "../data/frugalrag/eval/${type}/colbert/${dataset}/qwen${size}b/${save_name}_0.90/" \
    --prompt_path "src/prompts/colbert/${dataset}/qwen${size}b/m5_nofinish/ans/bootstrapped_1.json" \
    --answer_model "Qwen/Qwen2.5-${size}B-Instruct" \
    --port $trained_port $answer_port \
    --search_port $search_port \
    --dataset_name "${dataset}" \
    --input_file "${testfile}" &
  eval_test_pid=$!

elif [[ "$model" == "llama" ]]; then
  echo "Starting VLLM servers for Llama ${size}B model"
  # start vllm server for answer model
  CUDA_VISIBLE_DEVICES=$device_map_b python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-${size}B-Instruct \
    --gpu-memory-utilization 0.70 \
    --tensor-parallel-size 1 \
    --port $answer_port &
  answer_pid=$!

  if [[ "${type}" == "grpo" ]]; then
    model_path="../data/frugalrag/${type}/colbert/${dataset}/llama${size}b/${save_name}_0.90/final"
  elif [[ "${type}" == "sft" ]]; then
    model_path="../data/frugalrag/${type}/colbert/${dataset}/llama${size}b/${save_name}_0.90/meta-llama_Meta-Llama-3.1-${size}B-Instruct"
  elif [[ "${type}" == "base" ]]; then
    model_path="meta-llama/Meta-Llama-3.1-${size}B-Instruct"
  fi

echo "using model path: ${model_path}"

  # start vllm server for trained model
  CUDA_VISIBLE_DEVICES=$device_map_a python -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --gpu-memory-utilization 0.70 \
    --tensor-parallel-size 1 \
    --port "${trained_port}" &
  trained_pid=$!

  # Wait for VLLM servers to start
  echo "Waiting for VLLM servers to start..."
  sleep 30

  # on the train data
  if [ -e "datasets/${dataset}_1000.json" ]; then
    echo "Running evaluation on train data..."
    python -m src.evaluation.eval_mp \
      --max_iters $m \
      --rm "${rm}" \
      --ndocs $ndocs \
      --model_name_or_path "${model_path}" \
      --output_path "../data/frugalrag/eval/${type}/colbert/${dataset}/llama${size}b/train1000_${save_name}_nofinish/" \
      --prompt_path "src/prompts/colbert/${dataset}/llama${size}b/m5_nofinish/ans/bootstrapped_1.json" \
      --answer_model "meta-llama/Meta-Llama-3.1-${size}B-Instruct" \
      --port $trained_port $answer_port \
      --search_port $search_port \
      --dataset_name "${dataset}" \
      --input_file "datasets/${dataset}_1000.json" \
      --no_finish True &
    eval_train_pid=$!
  fi

  # on the test data
  echo "Running evaluation on test data..."
  python -m src.evaluation.eval_mp \
    --max_iters $m \
    --rm "${rm}" \
    --ndocs $ndocs \
    --model_name_or_path "${model_path}" \
    --output_path "../data/frugalrag/eval/${type}/colbert/${dataset}/llama${size}b/${save_name}_0.90/" \
    --prompt_path "src/prompts/colbert/${dataset}/llama${size}b/m5_nofinish/ans/bootstrapped_1.json" \
    --answer_model "meta-llama/Meta-Llama-3.1-${size}B-Instruct" \
    --dataset_name "${dataset}" \
    --port $trained_port $answer_port \
    --search_port $search_port \
    --input_file $testfile &
  eval_test_pid=$!
fi

# wait $eval_test_pid $eval_train_pid
wait $eval_test_pid

# Cleanup background processes
echo "Cleaning up processes..."
cleanup

echo "Evaluation completed for $dataset-$model-${size}B"