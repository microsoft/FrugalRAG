# FrugalRAG

**A novel Retrieval-Augmented-Generation framework for answering complex multi-hop questions on a large corpus, focusing on inference-time efficiency.**

[![arXiv](https://img.shields.io/badge/arXiv-2507.07634-b31b1b.svg)](https://arxiv.org/abs/2507.07634)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Paper:** *FrugalRAG: Learning to retrieve and reason for multi-hop QA*  

## Overview

FrugalRAG is designed to retrieve less, think smarter, and answer multi-hop questions efficiently. Our approach focuses on:

- **Smart Prompting:** Better prompting strategies over brute-force fine-tuning
- **Retrieval Efficiency:** Fewer retrieval calls leading to lower cost and latency
- **Minimal Training Data:** High accuracy with minimal training data (~1K examples)
- **Supports Multiple Retrievers:** Compatible with ColBERT and E5-large retrievers

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
git clone https://github.com/java-abhinav07/msr-rag.git
cd msr-rag
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

# Download E5 KILT embeddings
bash download_embeddings.sh
```

Download the 21M Wikipedia Index (both abstracts and body) using `wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz`

Use `https://github.com/stanford-futuredata/ColBERT` to index the files in the correct format. We plan on releasing the models and data soon.

### Evaluation Datasets

This repository includes the 1000 example train datasets in the `datasets/` folder. For complete evaluation datasets, download from:

| Dataset | Source | Alternative |
|---------|--------|-------------|
| **2WikiMultihopQA** | [Official](https://www.dropbox.com/scl/fi/heid2pkiswhfaqr5g0piw/data.zip?rlkey=ira57daau8lxfj022xvk1irju&e=1) | `framolfese/2WikiMultihopQA` |
| **HotpotQA** | [Official](https://hotpotqa.github.io/) | `hotpotqa/hotpot_qa` |
| **MuSiQue** | — | `dgslibisey/MuSiQue` |

## Quick Start

### 1. Start Language Model Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 1 \
  --port 7501
```

### 2. Start Retrieval Backend

**Option A: ColBERT**
```bash
CUDA_VISIBLE_DEVICES=1 PORT=8000 python -m src.search.serve_colbert.py \
  --index_root ../data/leret/ \
  --index wiki17.nbits.local \
  --colbert_path ../data/leret/colbertv2.0 \
  --collection_path ../data/wiki.abstracts.2017/collection.tsv
```

**Option B: E5-large**
```bash
INDEX_DIR="e5-large-index" \
E5_MODEL_NAME_OR_PATH="intfloat/e5-large-v2" \
TOP_K=5 \
uvicorn src.search.start_e5_server_main:app --port 8000
```

## Training Pipeline

The complete training pipeline consists of three main stages:

### Step 1: Data Generation

#### 1.1 Prompt Optimization

Ensure the `search_port` and `port` arguments in your config file match the servers from the Quick Start section.

```bash
# Generate finish prompts
python -m src.data.optimize_prompt --config configs/create_data/colbert/hotpot_qwen7b_finish.json

# Generate no-finish prompts  
python -m src.data.optimize_prompt \
  --config configs/create_data/colbert/hotpot_qwen7b_nofinish.json
```

#### 1.2 SFT Data Generation

```bash
# Generate finish training data
python -m src.data.create_data_mp \
  --config configs/create_data/colbert/hotpot_qwen7b_finish.json

# Generate no-finish training data
python -m src.data.create_data_mp \
  --config configs/create_data/colbert/hotpot_qwen7b_nofinish.json

# Combine datasets
python src/data/combine_sft_data.py \
  --finish_path "../data/frugalrag/sft_data/colbert/hotpot/qwen7b/m5_finish/train_sft.json" \
  --nofinish_path "../data/frugalrag/sft_data/colbert/hotpot/qwen7b/m5_nofinish/train_sft.json" \
  --out_dir "../data/frugalrag/sft_data/colbert/hotpot/qwen7b/m5_0.90/"
```

### Step 2: Supervised Fine-Tuning (SFT)

#### 2.1 Generate Answer Prompts

This step removes ReAct demonstrations for final answer generation, we use the same downstream prompt for all baselines:

```bash
python gen_ans_prompt.py --prompt_root src/prompts/
```

#### 2.2 Train Models

```bash
# Train base model
python -m src.train.sft --config configs/sft/colbert/hotpot_qwen7b_m5_0.90.json
```

#### 2.3 Generate Answerability Threshold (τ)

Start the required servers:

```bash
# Start retrieval backend
CUDA_VISIBLE_DEVICES=0 PORT=8000 python -m src.search.serve_colbert \
  --index_root ../data/leret/ \
  --index wiki17.nbits.local \
  --colbert_path ../data/leret/colbertv2.0 \
  --collection_path ../data/wiki.abstracts.2017/collection.tsv

# Start trained model
python -m vllm.entrypoints.openai.api_server \
  --model "../data/frugalrag/sft/colbert/hotpot/qwen7b/m5_0.90/Qwen_Qwen2.5-7B-Instruct" \
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 1 \
  --port 7501

# Start answer model
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 1 \
  --port 7502
```

Run evaluation to generate threshold logs:

```bash
python -m src.evaluation.eval_mp \
  --model_name_or_path "../data/frugalrag/sft/colbert/hotpot/qwen7b/m5_0.90/Qwen_Qwen2.5-7B-Instruct" \
  --output_path "../data/frugalrag/eval/sft/colbert/hotpot/qwen7b/train1000_m5_0.90/" \
  --prompt_path "src/prompts/colbert/hotpot/qwen7b/m5_nofinish/ans/bootstrapped_1.json" \
  --answer_model "Qwen/Qwen2.5-7B-Instruct" \
  --port 7501 7502 \
  --search_port 8000 \
  --dataset_name "hotpot" \
  --input_file "datasets/hotpot_1000.json" \
  --no_finish True
```

Alternatively you may use the bash script which automatically launches the servers:
`bash eval.sh [dataset-name (2wiki | hotpot | musique)] [model-name (qwen | llama)] [model-size (3 | 7 | 8)] [search-port] [answer-port] [reasoner-port] [gpu-id-1] [gpu-id-2] [type (base | sft | grpo)]`

### Step 3: Reinforcement Learning (GRPO)

#### 3.1 Setup Accelerate

Configure accelerate for multi-GPU training:

```bash
accelerate config
# Use DeepSpeed Zero2, 7 GPUs for training, 1 for generation
# See configs/default_config.yaml for reference
```

#### 3.2 Host SFT Model
Use command `trl vllm-serve` compatible with `trl` for training
```bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model "../data/frugalrag/sft/colbert/hotpot/qwen7b/m5_0.90/Qwen_Qwen2.5-7B-Instruct" \
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 1 \
  --port 8000
```

#### 3.3 Start Retrieval Server

Example: ColBERTV2
```bash
CUDA_VISIBLE_DEVICES=1 PORT=8001 python -m src.search.serve_colbert \
  --index_root ../data/leret/ \
  --index wiki17.nbits.local \
  --colbert_path ../data/leret/colbertv2.0 \
  --collection_path ../data/wiki.abstracts.2017/collection.tsv
```

#### 3.4 Train with GRPO

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch -m src.train.train_grpo \
  --config configs/grpo/colbert/hotpot_qwen7b_m5_0.90.json
```

## Evaluation

### Running Evaluation

Ensure all required services are running before evaluation:

```bash
# Start retrieval backend
CUDA_VISIBLE_DEVICES=0 PORT=8000 python -m src.search.serve_colbert \
  --index_root ../data/leret/ \
  --index wiki17.nbits.local \
  --colbert_path ../data/leret/colbertv2.0 \
  --collection_path ../data/wiki.abstracts.2017/collection.tsv

# Start trained model
python -m vllm.entrypoints.openai.api_server \
  --model [MODEL_PATH] \
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 1 \
  --port 7501

# Start answer model
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \ # any base model
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 1 \
  --port 7502
```

Run evaluation:

```bash
python -m src.evaluation.eval_mp \
  --model_name_or_path [MODEL_PATH] \
  --output_path [OUTPUT_PATH] \
  --prompt_path [PROMPT_PATH] \
  --answer_model "Qwen/Qwen2.5-7B-Instruct" \ # any base model
  --port 7501 7502 \
  --search_port 8000 \
  --dataset_name "hotpot" \
  --input_file [DEV_FILE_PATH]
```

### Available Evaluation Metrics

The evaluation framework automatically computes:

- **Exact Match (EM)**: Binary accuracy for correct answers
- **Match**: Checks if gold answer is present in the generated answer
- **F1 Score**: Token-level overlap between predicted and gold answers  
- **Cost Efficiency**: Retrieval operations per query
- **Recall/Support F1**: Retrieval peformance

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

### Model Adaptations

To adapt FrugalRAG to new models or datasets:

1. **New Models**: Update `model_name_or_path` in config files
2. **New Datasets**: Add dataset-specific prompts in `src/prompts/`
3. **New Retrievers**: Implement retriever interface in `src/search/`

---
### BrowseCompPlus

`git clone https://github.com/texttron/BrowseComp-Plus.git`<br>
Follow installation instructions provided in the repository and start the server.

`python searcher/mcp_server.py --searcher-type faiss --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" --port 9000 --transport streamable-http --normalize --model-name "Qwen/Qwen3-Embedding-8B"`

Follow Steps 1 through 3

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

## Intended Uses

FrugalRAG codebase is best suited for training and evaluating large language models for complex multi-hop questions against a large corpus of documents. The codebase is being shared with the research community to facilitate reproduction of our results and foster further research in this area. The codebase is intended to be used by domain experts who are independently capable of evaluating the quality of outputs before acting on them. 

## Out-of-Scope Uses
FrugalRAG codebase is not well suited for scenarios outside of the scope of training a reasoning model for multi-hop question answering, for example, our codebase does not support training or fine-tuning a good retrieval index on the corpus. We do not recommend using FrugalRAG codebase in commercial or real-world applications without further testing and development. It is being released for research purposes. The codebase was not designed or evaluated for all possible downstream purposes. Developers should consider its inherent limitations as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use. Without further testing and development, FrugalRAG codebase should not be used in sensitive domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities. We do not recommend using FrugalRAG codebase in the context of high-risk decision making (e.g. in law enforcement, legal, finance, or healthcare). 

## Best Practices
Better performance can be achieved by leveraging the latest LLM models, curating a comprehensive domain-specific knowledge base, and involving experts to actively verify and refine the model-generated responses. We also recommend following the instructions provided in the README of  FrugalRAG codebase, as well as the hyperparameter configurations included in the paper. 

We strongly encourage users to use LLMs/MLLMs that support robust Responsible AI mitigations, such as Azure Open AI (AOAI) services. Such services continually update their safety and RAI mitigations with the latest industry standards for responsible use. For more on AOAI’s best practices when employing foundations models for scripts and applications: 

[Blog post on responsible AI features in AOAI that were presented at Ignite 2023](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/announcing-new-ai-safety-amp-responsible-ai-features-in-azure/ba-p/3983686) 

[Overview of Responsible AI practices for Azure OpenAI models](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview) 

[Azure OpenAI Transparency Note](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note) 

[OpenAI’s Usage policies](https://openai.com/policies/usage-policies) 

[Azure OpenAI’s Code of Conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct) 

Users are responsible for sourcing their datasets legally and ethically. This could include securing appropriate rights, ensuring consent for use of audio/images, and/or the anonymization of data prior to use in research.    

Users are reminded to be mindful of data privacy concerns and are encouraged to review the privacy policies associated with any models and data storage solutions interfacing with FrugalRAG codebase.  

It is the user’s responsibility to ensure that the use of FrugalRAG codebase complies with relevant data protection regulations and organizational guidelines. 

Developers should follow transparency best practices and inform end-users they are interacting with an AI system. 

## License

Code in this Github repository is licensed under the [MIT License](https://github.com/microsoft/frugalrag/blob/main/LICENSE).
The FrugalRAG codebase is released under the CDLA v2 license.

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
