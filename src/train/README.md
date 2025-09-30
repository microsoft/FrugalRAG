## Training

### Supervised Fine-Tuning (SFT)

#### Train Models

```bash
# Train base model
python -m src.train.sft --config configs/sft/e5/hotpot_qwen7b_m5_0.90.json
```

#### Generate Answerability Threshold (τ)

First, start the required servers and run evaluation to generate threshold logs:

```bash
python -m src.evaluation.eval_mp --model_name_or_path [SFT_MODEL] --output_path [OUT_DIR] --answer_model [BASE_MODEL] --port 7501 7502 --search_port 8000 --dataset_name [DATASET_NAME] --input_file [TRAIN_FILE_PATH] --no_finish True
```

### Reinforcement Learning (GRPO)

#### Setup Accelerate

Configure accelerate for multi-GPU training:

```bash
accelerate config
# Use DeepSpeed Zero2, 7 GPUs for training, 1 for generation
# See configs/default_config.yaml for reference
```

#### Serve SFT Model
Use command `trl vllm-serve` compatible with `trl` for training
```bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model [SFT_MODEL] --gpu-memory-utilization 0.70 --tensor-parallel-size 1 --port 8000
```

#### Train with GRPO
First, start the retriever backend as shown in the Quick Start.

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch -m src.train.train_grpo --config configs/grpo/e5/hotpot_qwen7b_m5_0.90.json
```