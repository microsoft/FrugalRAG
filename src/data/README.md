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

### Data Generation

#### Prompt Optimization

Ensure the `search_port` and `port` arguments in your config file match the servers from the Quick Start section.

```bash
# Generate finish prompts
python -m src.data.optimize_prompt --config configs/create_data/e5/hotpot_qwen7b_finish.json

# Generate no-finish prompts  
python -m src.data.optimize_prompt --config configs/create_data/e5/hotpot_qwen7b_nofinish.json
```

#### SFT Data Generation

```bash
# Generate finish training data
python -m src.data.create_data_mp --config configs/create_data/e5/hotpot_qwen7b_finish.json

# Generate no-finish training data
python -m src.data.create_data_mp --config configs/create_data/e5/hotpot_qwen7b_nofinish.json

# Combine datasets
python src/data/combine_sft_data.py --finish_path [PATH_TO_FINISH_DS] --nofinish_path [PATH_TO_NOFINISH_DS] --out_dir [OUT_DIR]

# retains extract prompt
python gen_ans_prompt.py --prompt_root src/prompts/
```
