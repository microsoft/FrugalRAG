# FrugalRAG

**A Retrieval-Augmented Generation approach for efficient multi-hop question answering**

[![arXiv](https://img.shields.io/badge/arXiv-2507.07634-b31b1b.svg)](https://arxiv.org/abs/2507.07634)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Paper:** *FrugalRAG: Learning to retrieve and reason for multi-hop QA*  

## Overview
Reinforcement learning (RL) based on the final answer’s reward has driven recent progress in small language models (SLMs) on reasoning-heavy tasks such as math and code. However, applying the same techniques to retrieval-augmented generation (RAG) benchmarks like multi-hop QA has yielded limited gains—often trailing supervised or prompting-only baselines. Instead, we argue that a viable path for RL in multi-hop QA is to use test-time scaling judiciously, for optimizing both the final answer accuracy and the efficiency in reaching that answer. We propose FrugalRAG, a two-stage finetuning framework that adaptively reduces the number of retrieval steps based on a question’s difficulty. First, we train an SLM with supervised finetuning on a full-exploration policy that generates broad sub-queries. Then, we apply RL to adaptively prune search depth based on question difficulty, directly rewarding policies that balance correctness with frugality. Unlike prior approaches requiring 100× more data, our method achieves competitive performance with only 1,000 examples. On HotPotQA and other multi-hop QA benchmarks, FrugalRAG attains state-of-the-art efficiency–accuracy tradeoffs, cutting retrieval cost nearly in half. Moreover, on the challenging BrowseCompPlus benchmark, it generalizes zero-shot and surpasses SLM-based and other baselines. These results demonstrate the use of RL—not to increase reasoning steps but to optimize them—as an effective solution for scalable, efficient RAG.


## Table of Contents

- [Installation](#installation)
- [Data Setup](#data-setup)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU
- 16GB+ GPU memory (recommended)

### Environment Setup

We recommend using Conda for environment management:

```bash
git clone https://github.com/microsoft/FrugalRAG.git
cd FrugalRAG
conda env create -n frag --file environment.yaml
conda activate frag
pip install vllm==0.8.3 --no-deps
```

## Data Setup

### Index and Embeddings

Download the required indices and embeddings:

```bash
# Download Stanford wiki abstracts index
wget https://downloads.cs.stanford.edu/nlp/data/colbert/baleen/wiki.abstracts.2017.tar.gz

# Download E5 embeddings provided by FlashRAG
wget https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/resolve/master/retrieval_corpus/wiki18_100w_e5_index.zip
```

Download the 21M Wikipedia Index (both abstracts and body) using `wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz`

For ColBERT, we use `https://github.com/stanford-futuredata/ColBERT` to index the files in the correct format. For ease of reproducibility, we plan on releasing the models and data soon.

### Evaluation Datasets

This repository includes the 1000 example train indices in the `datasets/` folder. For complete evaluation datasets, download from:

| Dataset | Source | Alternative |
|---------|--------|-------------|
| **2WikiMultihopQA** | [Official](https://www.dropbox.com/scl/fi/heid2pkiswhfaqr5g0piw/data.zip?rlkey=ira57daau8lxfj022xvk1irju&e=1) | `framolfese/2WikiMultihopQA` |
| **HotpotQA** | [Official](https://hotpotqa.github.io/) | `hotpotqa/hotpot_qa` |
| **MuSiQue** | — | `dgslibisey/MuSiQue` |

## Quick Start

### 1. Start Language Model Server

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.70 --tensor-parallel-size 1 --port 7501
```

### 2. Start Retrieval Backend

**Option A: ColBERT (Abstracts)**
```bash
CUDA_VISIBLE_DEVICES=1 PORT=8000 python -m src.search.serve_colbert.py --index_root ../data/index/ --index wiki17.nbits.local --colbert_path ../data/colbertv2.0 --collection_path ../data/wiki.abstracts.2017/collection.tsv
```

**Option B: ColBERT (Abstracts and Body)**
```bash
CUDA_VISIBLE_DEVICES=1 PORT=8000 python -m src.search.serve_colbert.py --index_root ../data/index/ --index wiki18.nbits.local --colbert_path ../data/colbertv2.0 --collection_path ../data/wiki.2018/collection.tsv
```

**Option C: E5**
```bash
# convert faiss flat index to pytorch shards for ease of use
python src/search/shard_embeddings.py

# start server
INDEX_DIR=../data/e5-base-v2/pytorch-shards/ E5_MODEL_NAME_OR_PATH="intfloat/e5-base-v2" TOP_K=5 uvicorn src.search.start_e5_server_main:app --port 8001 
```

## Training Pipeline

The complete training pipeline consists of three main stages:

### Step 1: Data Generation

#### 1.1 Prompt Optimization

Ensure the `search_port` and `port` arguments in your config file match the servers from the Quick Start section.

```bash
# Generate finish prompts
python -m src.data.optimize_prompt --config configs/create_data/e5/hotpot_qwen7b_finish.json

# Generate no-finish prompts  
python -m src.data.optimize_prompt --config configs/create_data/e5/hotpot_qwen7b_nofinish.json
```

#### 1.2 SFT Data Generation

```bash
# Generate finish training data
python -m src.data.create_data_mp --config configs/create_data/e5/hotpot_qwen7b_finish.json

# Generate no-finish training data
python -m src.data.create_data_mp --config configs/create_data/e5/hotpot_qwen7b_nofinish.json

# Combine datasets
python src/data/combine_sft_data.py --finish_path "../data/frugalrag/sft_data/e5/hotpot/qwen7b/m5_finish/train_sft.json" --nofinish_path "../data/frugalrag/sft_data/e5/hotpot/qwen7b/m5_nofinish/train_sft.json" --out_dir "../data/frugalrag/sft_data/e5/hotpot/qwen7b/m5_0.90/"

# retains extract prompt
python gen_ans_prompt.py --prompt_root src/prompts/
```

### Step 2: Supervised Fine-Tuning (SFT)

#### 2.1 Train Models

```bash
# Train base model
python -m src.train.sft --config configs/sft/e5/hotpot_qwen7b_m5_0.90.json
```

#### 2.3 Generate Answerability Threshold (τ)

First, start the required servers and run evaluation to generate threshold logs:

```bash
python -m src.evaluation.eval_mp --model_name_or_path "../data/frugalrag/sft/e5/hotpot/qwen7b/m5_0.90/Qwen_Qwen2.5-7B-Instruct" --output_path "../data/frugalrag/eval/sft/e5/hotpot/qwen7b/train1000_m5_0.90/" --answer_model "Qwen/Qwen2.5-7B-Instruct" --port 7501 7502 --search_port 8000 --dataset_name "hotpot" --input_file "datasets/hotpot_1000.json" --no_finish True
```

### Step 3: Reinforcement Learning (GRPO)

#### 3.1 Setup Accelerate

Configure accelerate for multi-GPU training:

```bash
accelerate config
# Use DeepSpeed Zero2, 7 GPUs for training, 1 for generation
# See configs/default_config.yaml for reference
```

#### 3.2 Serve SFT Model
Use command `trl vllm-serve` compatible with `trl` for training
```bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "../data/frugalrag/sft/e5/hotpot/qwen7b/m5_0.90/Qwen_Qwen2.5-7B-Instruct" --gpu-memory-utilization 0.70 --tensor-parallel-size 1 --port 8000
```

#### 3.3 Train with GRPO
First, start the retriever backend.

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch -m src.train.train_grpo --config configs/grpo/e5/hotpot_qwen7b_m5_0.90.json
```

## Evaluation

### Running Evaluation
Ensure all required services are running before evaluation.

Run evaluation:

```bash
python -m src.evaluation.eval_mp --model_name_or_path [MODEL_PATH] --output_path [OUTPUT_PATH] --prompt_path [PROMPT_PATH] --answer_model "Qwen/Qwen2.5-7B-Instruct" \ # any base mode--port 7501 7502 --search_port 8000 --dataset_name "hotpot" --input_file [DEV_FILE_PATH]

# extract the final answer with CoT prompt
python -m src.evaluation.eval_mp --model_name_or_path [MODEL_PATH] --output_path [OUTPUT_PATH] --prompt_path [PROMPT_PATH] --answer_model "Qwen/Qwen2.5-7B-Instruct" \ # any base mode--port 7501 7502 --search_port 8000 --dataset_name "hotpot" --input_file [DEV_FILE_PATH] --answer_only True
```

Run MBE (ensure you set the path in the grade_all.py script)
```
python -m src.evaluation.grade_all
```


### Available Evaluation Metrics

The evaluation framework automatically computes:

- **Exact Match (EM)**: Binary accuracy for correct answers
- **Match**: Checks if gold answer is present in the generated answer
- **F1 Score**: Token-level overlap between predicted and gold answers  
- **Cost Efficiency**: Retrieval operations per query
- **Recall/Support F1**: Retrieval peformance
- **MBE**: LLM Judge Score

## Configuration

### Configuration Files

Configuration files are organized in the `configs/` directory:

```
configs/
├── create_data/          # Data generation configs
│   └── colbert/
│       ├── hotpot_qwen7b_finish.json
│       └── hotpot_qwen7b_nofinish.json
├── sft/                  # Supervised fine-tuning configs
│   └── colbert/
│       ├── hotpot_qwen7b_m5_0.90.json
│       └── hotpot_qwen7b_m5_nofinish.json
├── grpo/                 # GRPO reinforcement learning configs
│   └── colbert/
│       └── hotpot_qwen7b_m5_0.90.json
└── default_config.yaml   # Accelerate configuration
```

> Please set the correct paths, port numbers to ensure the models run smoothly.

### Key Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model_name_or_path` | Base model for training | `"Qwen/Qwen2.5-7B-Instruct"` |
| `search_port` | Retrieval server port | `8000` |
| `port` | Model server ports (reasoner, answer generator) | `[7501, 7502]` |
| `max_iters` | Maximum reasoning iterations | `5` |
| `ndocs` | Documents retrieved per iteration | `3` or `5` |
| `dataset_name` | Target dataset (hotpot, 2wiki, musique) | `"hotpot"` |

---

## Troubleshooting

### Common Issues & Solutions

#### Import Errors in ColBERTv2

**Issue**: `ImportError: cannot import name 'AdamW' from 'transformers'`

**Solution**: Comment out the import in the relevant files. We use a newer `transformers` version.

#### Device Mismatch During SFT

**Issue**: `RuntimeError: Expected all tensors to be on the same device`

**Solution**: In the transformers package file `loss_utils.py`, line 38, use:
```python
if reduction == "sum":
    loss = loss / num_items_in_batch.to(loss.device)
return loss
```

#### NCCL Error
```
Exception: Call to collective_rpc method failed: Weight update group already initialized. Call close_communicator first.                                                                                                                                              
```
Just rerun `trl vllm-serve`, sometimes it does not call `close_communicator` on its own.

### Performance Optimization

- **Memory Usage**: Adjust `--gpu-memory-utilization` based on your GPU memory
- **Training Speed**: Use DeepSpeed Zero3 for large model training
- **Inference Speed**: Use `--tensor-parallel-size` for multi-GPU inference

## Citation

If you use this repository, please cite our paper:

```bibtex
@misc{java2025frugalraglearningretrievereason,
      title={FrugalRAG: Learning to retrieve and reason for multi-hop QA}, 
      author={Abhinav Java and Srivathsan Koundinyan and Nagarajan Natarajan and Amit Sharma},
      year={2025},
      eprint={2507.07634},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.07634}, 
}
```

## Acknowledgments
- [ColBERT](https://github.com/stanford-futuredata/ColBERT) for the efficient retrieval framework
- [DSPy](https://github.com/stanfordnlp/dspy) for the programming framework for language models
- [vLLM](https://github.com/vllm-project/vllm) for high-performance LLM inference
