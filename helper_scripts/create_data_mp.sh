#!/bin/bash
# Usage: bash create_data_mp.sh [dataset-name] [model-name] [model-size] [search-port] [lm-port] [gpu-id]
# Example: bash create_data_mp.sh hotpot qwen 7 8000 7501 0
set -e
set -o pipefail

dataset=$1
model=$2
size=$3
search_port=${4:-8000}
lm_port=${5:-7501}
gpu_id=${6:-0}
gpu_id_1=${7:-1}
retriever_type=${8:-"colbert"}

if [[ -z "$dataset" || -z "$model" || -z "$size" ]]; then
    echo "Usage: bash create_data_mp.sh [dataset-name] [model-name] [model-size] [search-port] [lm-port] [gpu-id]"
    echo "  dataset-name: 2wiki | hotpot | musique"
    echo "  model-name: qwen | llama"
    echo "  model-size: 3 | 7 | 8"
    exit 1
fi

# Function to cleanup all background processes
cleanup() {
    echo "Received interrupt signal. Cleaning up processes..."
    if [[ -n "$colbert_pid" ]]; then
        echo "Killing ColBERT server (PID: $colbert_pid)"
        kill $colbert_pid 2>/dev/null
    fi
    if [[ -n "$lm_pid" ]]; then
        echo "Killing LM server (PID: $lm_pid)"
        kill $lm_pid 2>/dev/null
    fi
    echo "Cleanup completed. Exiting..."
    exit 1
}

trap cleanup SIGINT

echo "Starting data creation for $dataset-$model-${size}B"

if [[ "$retriever_type" == "colbert" ]]; then
    # Start ColBERT server
    if [[ "$dataset" == "2wiki" || "$dataset" == "musique" ]]; then
        PORT=$search_port CUDA_VISIBLE_DEVICES=$gpu_id python -m src.search.serve_colbert \
            --index_root ../data/ColBERT/colbert_scripts/experiments/wiki20M/indexes \
            --index wiki20M.2bits \
            --colbert_path ../data/leret/colbertv2.0 \
            --collection_path ../data/data/wiki20M/collection.tsv &
        colbert_pid=$!
        echo "Waiting for ColBERT server to start..."
        sleep 60

    elif [[ "$dataset" == "hotpot" ]]; then
        PORT=$search_port CUDA_VISIBLE_DEVICES=$gpu_id python -m src.search.serve_colbert \
            --index_root ../data/leret/ \
            --index wiki17.nbits.local \
            --colbert_path ../data/leret/colbertv2.0 \
            --collection_path ../data/wiki.abstracts.2017/collection.tsv &
        colbert_pid=$!
        echo "Waiting for ColBERT server to start..."
        sleep 60
    fi

elif [[ "$retriever_type" == "e5" ]]; then
    echo "Using E5 as retriever."
    CUDA_VISIBLE_DEVICES=$gpu_id E5_MODEL_NAME_OR_PATH="intfloat/e5-large-v2" TOP_K=5 uvicorn src.search.start_e5_server_main:app  --port $search_port &
    colbert_pid=$!
    sleep 60
fi


# Start LM server
if [[ "$model" == "qwen" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu_id_1 python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-${size}B-Instruct \
        --gpu-memory-utilization 0.70 \
        --tensor-parallel-size 1 \
        --port $lm_port &
    lm_pid=$!
elif [[ "$model" == "llama" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu_id_1 python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3.1-${size}B-Instruct \
        --gpu-memory-utilization 0.70 \
        --tensor-parallel-size 1 \
        --port $lm_port &
    lm_pid=$!
fi

echo "Waiting for LM server to start..."
sleep 120

# Run optimize_prompt for finish and nofinish configs
pids_op=()
for finish_type in finish nofinish; do
    config_file="configs/create_data/${retriever_type}/${dataset}_${model}${size}b_${finish_type}.json"
    if [[ -f "$config_file" ]]; then
        echo "Running optimize_prompt for $config_file"
        python -m src.data.optimize_prompt --config "$config_file" --search_port $search_port --port $lm_port &
        pids_op+=($!)
    else
        echo "Config file $config_file not found, skipping."
    fi
done

# Wait for all background create_data_mp processes to complete
for pid in "${pids_op[@]}"; do
    wait $pid
done


# Run create_data_mp for finish and nofinish configs
pids=()
for finish_type in finish nofinish; do
    config_file="configs/create_data/${retriever_type}/${dataset}_${model}${size}b_${finish_type}.json"
    if [[ -f "$config_file" ]]; then
        echo "Running create_data_mp for $config_file"
        python -m src.data.create_data_mp --config "$config_file" --search_port $search_port --port $lm_port &
        pids+=($!)
    else
        echo "Config file $config_file not found, skipping."
    fi
done

# Wait for all background create_data_mp processes to complete
for pid in "${pids[@]}"; do
    wait $pid
done

echo "Data creation completed. Cleaning up..."
cleanup
