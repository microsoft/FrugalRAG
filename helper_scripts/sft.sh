python3 -m src.train.sft --config configs/sft/colbert/${1}_qwen3b_finish.json
python3 -m src.train.sft --config configs/sft/colbert/${1}_qwen7b_finish.json
python3 -m src.train.sft --config configs/sft/colbert/${1}_llama8b_finish.json

python3 -m src.train.sft --config configs/sft/colbert/${1}_qwen3b_m5_0.90.json
python3 -m src.train.sft --config configs/sft/colbert/${1}_qwen7b_m5_0.90.json
python3 -m src.train.sft --config configs/sft/colbert/${1}_llama8b_m5_0.90.json
