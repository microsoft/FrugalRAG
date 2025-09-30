import argparse
import os
from typing import Optional
from .config import Config, load_config


def create_parser_with_config(description: str = "FrugalRAG Project") -> argparse.ArgumentParser:
    """Create a parser that supports configuration files and command-line arguments."""
    parser = argparse.ArgumentParser(description=description)
    
    # Add config file argument
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file (JSON or YAML)"
    )
    
    # Add all possible arguments (will be ignored if not relevant)
    _add_all_arguments(parser)
    
    return parser


def _add_all_arguments(parser):
    """Add all possible arguments to the parser."""
    # Model arguments
    model_group = parser.add_argument_group('model', 'Model configuration')
    model_group.add_argument("--model_name_or_path", type=str, help="Model path")
    model_group.add_argument("--answer_model", type=str, help="Answer model path")
    model_group.add_argument("--no_finish", type=bool, help="Remove finish from ReAct")
    model_group.add_argument("--max_iters", type=int, help="Maximum iterations")
    model_group.add_argument("--max_tokens", type=int, help="Maximum tokens")
    model_group.add_argument("--max_retries", type=int, help="Maximum retries")
    model_group.add_argument("--baseline", type=str, help="for baseline eval")
    model_group.add_argument("--answer_only", type=str, help="for baseline eval")
    model_group.add_argument("--prompt_version", type=str, help="for baseline eval")
    
    # Data arguments
    data_group = parser.add_argument_group('data', 'Data configuration')
    data_group.add_argument("--dataset_name", type=str, help="Dataset name")
    data_group.add_argument("--input_file", type=str, help="Dataset json file")
    data_group.add_argument("--num_examples", type=int, help="Number of examples")
    data_group.add_argument("--candidates", type=int, help="Number of candidates")
    data_group.add_argument("--num_test", type=int, help="Number of test examples")
    
    # Search arguments
    search_group = parser.add_argument_group('search', 'Search configuration')
    search_group.add_argument("--port", nargs="*", type=int, help="Port numbers for LLMs (in order, reasoner and generator)")
    search_group.add_argument("--rm", type=str, help="Retrieval model")
    search_group.add_argument("--collection_path", type=str, help="Collection path")
    search_group.add_argument("--index_root", type=str, help="Index root directory")
    search_group.add_argument("--index", type=str, help="Index name")
    search_group.add_argument("--ndocs", type=int, help="Number of documents")
    search_group.add_argument("--colbert_path", type=str, help="Path to colbert checkpoint")
    search_group.add_argument("--search_port", type=int, help="Port")
    
    # Path arguments
    path_group = parser.add_argument_group('paths', 'Path configuration')
    path_group.add_argument("--output_path", type=str, help="Output directory")
    path_group.add_argument("--prompt_path", type=str, help="Prompt directory")
    path_group.add_argument("--prompt_save_path", type=str, help="Prompt directory")
    path_group.add_argument("--mb_logs", type=str, help="MB logs path")
    
    # SFT arguments (optional)
    sft_group = parser.add_argument_group('sft', 'Training configuration')
    sft_group.add_argument("--epochs", type=int, help="Number of epochs")
    sft_group.add_argument("--max_seq_len", type=int, help="Max len")
    sft_group.add_argument("--batch_size", type=int, help="Train batch size")
    sft_group.add_argument("--grad_acc", type=int, help="Gradient accumulation")
    sft_group.add_argument("--warmup", type=int, help="Warmup steps")
    sft_group.add_argument("--lr", type=float, help="Learning rate")
    
    # RL arguments
    grpo_group = parser.add_argument_group('grpo', 'RL configuration')
    grpo_group.add_argument("--project_name", type=str, default="frugal-rag", help="Wandb project name")
    grpo_group.add_argument("--resume", type=str, help="Resume from checkpoint")
    grpo_group.add_argument("--run_name", type=str, help="Run name")
    grpo_group.add_argument("--beta", type=float, help="Beta parameter")
    grpo_group.add_argument("--format_coeff", type=float, help="Format coefficient")
    grpo_group.add_argument("--zeta", type=float, help="Zeta parameter")
    grpo_group.add_argument("--max_steps", type=int, help="Maximum steps")
    grpo_group.add_argument("--max_reward", type=float, help="Maximum reward")
    grpo_group.add_argument("--vllm_port", type=int, default=8000, help="VLLM port")
    grpo_group.add_argument("--grad_ckpt", action="store_true", help="Use gradient checkpointing for large models")


def parse_args_with_config(parser: Optional[argparse.ArgumentParser] = None) -> Config:
    """
    Parse arguments and return a Config object.
    """
    if parser is None:
        parser = create_parser_with_config()
    
    args = parser.parse_args()
    
    # Load configuration file if specified
    if args.config and os.path.exists(args.config):
        config = Config.from_file(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        # Create default config
        config = load_config()
    
    # Override config with command-line arguments (only non-None values)
    config = config.merge_with_args(args)
    
    return config


def get_args_for_script(script_type: str = "eval") -> Config:
    """
    Get arguments for a specific script type.
    
    Args:
        script_type: Type of script ("eval", "train", "grpo", "search", "data")
    
    Returns:
        Config object with appropriate defaults
    """
    if script_type == "eval":
        parser = create_parser_with_config("Evaluation Script")
    elif script_type == "sft":
        parser = create_parser_with_config("Training Script")
    elif script_type == "grpo":
        parser = create_parser_with_config("GRPO Training Script")
    elif script_type == "search":
        parser = create_parser_with_config("Search Server Script")
    elif script_type == "data":
        parser = create_parser_with_config("Data Processing Script")
    else:
        parser = create_parser_with_config()
        
    return parse_args_with_config(parser)


def get_args(script_type="eval"):
    """Backward compatibility function - returns Config object."""
    return get_args_for_script(script_type)
