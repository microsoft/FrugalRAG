import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Union

# Try to import yaml, but make it optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

@dataclass
class ModelConfig:
    """Model-related configuration."""
    model_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    answer_model: str = ""
    no_finish: bool = False
    max_iters: int = 5
    max_tokens: int = 512
    max_retries: int = 4
    baseline : str = ""
    answer_only: bool = False
    prompt_version: str = "doc_cot"



@dataclass
class DataConfig:
    """Data-related configuration."""
    dataset_name: str = "hotpot"
    input_file: str = ""
    num_examples: int = 1000
    num_prompt_examples: int = 100
    candidates: int = 3
    num_test: int = -1


@dataclass
class SearchConfig:
    """Search-related configuration."""
    port: List[int] = None
    search_port: int = None
    rm: str = "wiki5M"
    collection_path: str = ""
    index_root: str = ""
    index: str = ""
    colbert_path: str = ""
    ndocs: int = 3
    
    def __post_init__(self):
        if self.port is None:
            self.port = [7501]


@dataclass
class PathConfig:
    """Path-related configuration."""
    output_path: str = "results/debug"
    prompt_path: str = ""
    prompt_save_path: str = ""
    mb_logs: str = ""


@dataclass
class RLConfig:
    """RL-specific configuration."""
    resume: str = ""
    run_name: str = "default"
    beta: float = 0.1
    format_coeff: float = 0.50
    zeta: float = 2.0
    max_steps: int = 400
    max_reward: float = 2.0
    dynamic_threshold: bool = False
    project_name: str = "frugal-rag"
    vllm_port: int = 8000
    grad_ckpt: bool = False

@dataclass
class SFTConfig:
    """SFT configuration"""
    epochs: int = 1
    max_seq_len: int = 4096
    batch_size: int = 4
    warmup: int = 20
    lr: float = 1e-4
    grad_acc: int = 2


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    data: DataConfig
    paths: PathConfig
    search: Optional[SearchConfig] = None
    sft: Optional[SFTConfig] = None
    grpo: Optional[RLConfig] = None
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from a file (JSON or YAML)."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            if (config_path.endswith('.yaml') or config_path.endswith('.yml')):
                if not HAS_YAML:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
                data = yaml.safe_load(f)
            else:
                # Default to JSON for all other file types
                data = json.load(f)
        
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            raise ValueError(f"Configuration file must contain a JSON/YAML object, got {type(data)}")
        
        # Create config with proper error handling
        try:
            return cls(
                model=ModelConfig(**data.get('model', {})),
                data=DataConfig(**data.get('data', {})),
                search=SearchConfig(**data.get('search', {})),
                paths=PathConfig(**data.get('paths', {})),
                grpo=RLConfig(**data.get('grpo', {})) if 'grpo' in data else None,
                sft=SFTConfig(**data.get('sft', {})) if 'sft' in data else None
            )
        except TypeError as e:
            raise ValueError(f"Invalid configuration structure in {config_path}: {e}")
    
    def to_file(self, config_path: str):
        """Save configuration to a file."""
        data = asdict(self)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if (config_path.endswith('.yaml') or config_path.endswith('.yml')):
                if not HAS_YAML:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                # Default to JSON for all other file types
                json.dump(data, f, indent=2)
    
    def merge_with_args(self, args):
        """Merge configuration with command-line arguments."""
        # Override config values with any non-None command-line arguments
        for key, value in vars(args).items():
            if value is None:  # Skip None values from command line
                continue
            
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.data, key):
                setattr(self.data, key, value)
            elif hasattr(self.search, key):
                setattr(self.search, key, value)
            elif hasattr(self.paths, key):
                setattr(self.paths, key, value)
            elif self.sft and hasattr(self.sft, key):
                setattr(self.sft, key, value)
            elif self.grpo and hasattr(self.grpo, key):
                setattr(self.grpo, key, value)
        
        return self


def load_config(config_path: Optional[str] = None, **overrides) -> Config:
    """
    Load configuration with optional file and overrides.
    
    Args:
        config_path: Path to configuration file (optional)
        **overrides: Direct configuration overrides
    
    Returns:
        Config object
    """
    if config_path and os.path.exists(config_path):
        config = Config.from_file(config_path)
        if config.sft is None:
            config.sft = SFTConfig()
        if config.grpo is None:
            config.grpo = RLConfig()
        if config.search is None:
            config.search = SearchConfig()
    else:
        # Create default configuration
        config = Config(
            model=ModelConfig(),
            data=DataConfig(),
            search=SearchConfig(),
            paths=PathConfig(),
            grpo=RLConfig(),
            sft=SFTConfig()
        )
    
    # Apply overrides
    for key, value in overrides.items():
        if value is None:  # Skip None values
            continue
            
        if hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config.search, key):
            setattr(config.search, key, value)
        elif hasattr(config.paths, key):
            setattr(config.paths, key, value)
        elif config.grpo and hasattr(config.grpo, key):
            setattr(config.grpo, key, value)
        elif config.sft and hasattr(config.sft, key):
            setattr(config.sft, key, value)
    
    return config
