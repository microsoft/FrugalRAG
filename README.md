# FrugalRAG

**A Retrieval-Augmented Generation approach for efficient multi-hop question answering**

[![arXiv](https://img.shields.io/badge/arXiv-2507.07634-b31b1b.svg)](https://arxiv.org/abs/2507.07634)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Paper:** *FrugalRAG: Learning to retrieve and reason for multi-hop QA*  

## Overview
Reinforcement learning (RL) based on the final answer’s reward has driven recent progress in small language models (SLMs) on reasoning-heavy tasks such as math and code. However, applying the same techniques to retrieval-augmented generation (RAG) benchmarks like multi-hop QA has yielded limited gains—often trailing supervised or prompting-only baselines. Instead, we argue that a viable path for RL in multi-hop QA is to use test-time scaling judiciously, for optimizing both the final answer accuracy and the efficiency in reaching that answer. We propose FrugalRAG, a two-stage finetuning framework that adaptively reduces the number of retrieval steps based on a question’s difficulty. First, we train an SLM with supervised finetuning on a full-exploration policy that generates broad sub-queries. Then, we apply RL to adaptively prune search depth based on question difficulty, directly rewarding policies that balance correctness with frugality. Unlike prior approaches requiring 100× more data, our method achieves competitive performance with only 1,000 examples. On HotPotQA and other multi-hop QA benchmarks, FrugalRAG attains state-of-the-art efficiency–accuracy tradeoffs, cutting retrieval cost nearly in half. Moreover, on the challenging BrowseCompPlus benchmark, it generalizes zero-shot and surpasses SLM-based and other baselines. These results demonstrate the use of RL—not to increase reasoning steps but to optimize them—as an effective solution for scalable, efficient RAG.


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

> See [dataset setup](src/data/README.md) and [training guide](src/train/README.md) before evaluation.


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
# convert faiss flat index to pytorch shards for fast execution
python src/search/shard_embeddings.py

# start server
INDEX_DIR=../data/e5-base-v2/pytorch-shards/ E5_MODEL_NAME_OR_PATH="intfloat/e5-base-v2" TOP_K=5 uvicorn src.search.start_e5_server_main:app --port 8001 
```


## Evaluation

### Running Evaluation
Ensure all required services are running before evaluation.

Run evaluation:

```bash
python -m src.evaluation.eval_mp --model_name_or_path [MODEL_PATH] --output_path [OUTPUT_PATH] --prompt_path [PROMPT_PATH] --answer_model [BASE_MODEL_NAME] --port 7501 7502 --search_port 8000 --dataset_name [DATASET_NAME] --input_file [DEV_FILE_PATH]

# extract the final answer with CoT prompt
python -m src.evaluation.eval_mp --model_name_or_path [MODEL_PATH] --output_path [OUTPUT_PATH] --prompt_path [PROMPT_PATH] --answer_model [BASE_MODEL_NAME] --port 7501 7502 --search_port 8000 --dataset_name [DATASET_NAME] --input_file [DEV_FILE_PATH] --answer_only True
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
