import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from src.train.grpo import GRPOReActTrainer
from src.model import Hopper
from src.evaluation import metrics
from src.parser_ import get_args
from src.data.data_utils import split_into_sentences


RANDOM_SEED = 663
random.seed(RANDOM_SEED)


class CustomTrainer:
    """
    GRPO Trainer for multi-hop question answering models.
        
    Args:
        args: Training configuration arguments
    """
    
    def __init__(self, args):
        """Initialize the GRPO trainer with configuration and models."""
        self.args = args
        self._setup_output_directory()
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Load and prepare dataset
        self.dataset = self._load_dataset()
        
        # Initialize ReAct model
        self.react_model = self._initialize_react_model()
        
        print("[INFO] GRPO trainer initialized successfully")

    def _setup_output_directory(self):
        """Create output directory and save configuration."""
        os.makedirs(self.args.paths.output_path, exist_ok=True)
        
        config_path = os.path.join(self.args.paths.output_path, "training_config.json")
        # with open(config_path, "w") as f:
        #     json.dump(vars(self.args), f, indent=2, sort_keys=True)
        
        print(f"[INFO] Training configuration saved to {config_path}")

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer from checkpoint or base model.
        
        Args:
            checkpoint_path: Path to checkpoint directory (optional)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device_string = f"cuda:{local_rank}"
        
        print(f"[INFO] Loading model: {self.args.model.model_name_or_path}")
        model_path = self.args.model.model_name_or_path
        
        # Load model with proper device mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": device_string},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Configure tokenizer padding
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
            model.config.pad_token_id = tokenizer.pad_token_id
            
        tokenizer.padding_side = "left"
        
        print(f"[INFO] Model loaded on device: {device_string}")
        return model, tokenizer

    def _initialize_react_model(self):
        """Initialize the ReAct reasoning model."""
        hopper = Hopper(self.args, custom=True, debug=False)
        
        if self.args.paths.prompt_path:
            print(f"[INFO] Loading custom prompts from: {self.args.paths.prompt_path}")
            hopper.update_react_prompt(self.args.paths.prompt_path)
        
        return hopper.orchestrator

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load and process dataset based on the specified dataset type.
        
        Returns:
            List of processed dataset examples
        """
        if self.args.data.input_file.endswith(".json"):
            with open(self.args.data.input_file, 'r') as f:
                raw_dataset = json.load(f)
        elif self.args.data.input_file.endswith(".jsonl"):
            with open(self.args.data.input_file) as f:
                raw_dataset = [json.loads(i) for i in f.readlines()]
        
        print(f"[INFO] Loading {self.args.data.dataset_name} dataset with {len(raw_dataset)} examples")
        
        if self.args.data.dataset_name in ["hotpot", "hopo"]:
            return self._process_hotpot_dataset(raw_dataset)
        elif self.args.data.dataset_name in ["musique", "dgslibisey/MuSiQue"]:
            return self._process_musique_dataset(raw_dataset)
        elif self.args.data.dataset_name == "2wiki":
            return self._process_2wiki_dataset(raw_dataset)
        elif self.args.data.dataset_name == "bcplus":
            return self._process_bcplus_dataset(raw_dataset)
        else:
            raise ValueError(f"Unsupported dataset: {self.args.data.dataset_name}")

    def _process_hotpot_dataset(self, raw_dataset: List[Dict]) -> List[Dict[str, Any]]:
        """Process HotpotQA dataset format."""
        dataset = []
        
        for example in raw_dataset:
            supporting_titles = [fact[0] for fact in example["supporting_facts"]]
            supporting_sent_indices = [fact[1] for fact in example["supporting_facts"]]
            
            gold_sentences, gold_titles = {}, []
            
            for context in example["context"]:
                title = context[0]
                if title in supporting_titles:
                    # Find the sentence index for this title
                    title_idx = supporting_titles.index(title)
                    sent_idx = supporting_sent_indices[title_idx]
                    
                    if sent_idx < len(context[1]):
                        if gold_sentences.get(title):
                            gold_sentences[title].append(context[1][sent_idx])
                        else:
                            gold_sentences[title] = [context[1][sent_idx]]
                            
                        gold_titles.append(title)
            
            dataset.append({
                "prompt": example["question"],
                "gold_titles": gold_titles,
                "answers": example["answer"],
                "gold_sent": gold_sentences,
            })
        
        print(f"[INFO] Processed {len(dataset)} HotpotQA examples")
        return dataset

    def _process_musique_dataset(self, raw_dataset: List[Dict]) -> List[Dict[str, Any]]:
        """Process MuSiQue dataset format."""
        dataset = []
        
        for example in raw_dataset:
            gold_sentences, gold_titles = {}, []
            
            for paragraph in example["paragraphs"]:
                if paragraph["is_supporting"]:
                    gold_titles.append(paragraph["title"])
                    
                    if gold_sentences.get(paragraph["title"]):
                        gold_sentences[paragraph["title"]].append(paragraph["paragraph_text"])
                    else:
                        gold_sentences[paragraph["title"]] = [paragraph["paragraph_text"]]

            dataset.append({
                "prompt": example["question"],
                "gold_titles": gold_titles,
                "answers": example["answer"],
                "gold_sent": gold_sentences,
            })
        
        print(f"[INFO] Processed {len(dataset)} MuSiQue examples")
        return dataset

    def _process_2wiki_dataset(self, raw_dataset: List[Dict]) -> List[Dict[str, Any]]:
        """Process 2WikiMultihopQA dataset format."""
        dataset = []
        
        for example in raw_dataset:
            supporting_titles = [fact[0] for fact in example["supporting_facts"]]
            supporting_sent_indices = [fact[1] for fact in example["supporting_facts"]]
            
            gold_sentences, gold_titles = {}, []
            
            for context in example["context"]:
                title = context[0]
                if title in supporting_titles:
                    title_idx = supporting_titles.index(title)
                    sent_idx = supporting_sent_indices[title_idx]
                    
                    if sent_idx < len(context[1]):                        
                        if gold_sentences.get(title):
                            gold_sentences[title].append(context[1][sent_idx])
                        else:
                            gold_sentences[title] = [context[1][sent_idx]]
                        
                        gold_titles.append(title)
            
            dataset.append({
                "prompt": example["question"],
                "gold_titles": gold_titles,
                "answers": example["answer"],
                "gold_sent": gold_sentences,
            })
        
        print(f"[INFO] Processed {len(dataset)} 2WikiMultihopQA examples")
        return dataset
    
    def _process_bcplus_dataset(self, raw_dataset: List[Dict]) -> List[Dict[str, Any]]:
        dataset = []
        for example in raw_dataset:
            dataset.append(
                {
                    "prompt": example["query"],
                    "answers": example["answer"],
                    "gold_titles": [int(d["docid"]) for d in example["evidence_docs"]],
                    "gold_sent": [int(d["docid"]) for d in example["gold_docs"]]
                }
            )
        
        print(f"[INFO] Processed {len(dataset)} BCPlus examples")
        return dataset

    def train(self):
        """        
        Configures training parameters, initializes the trainer, and runs training
        with proper checkpoint management and logging.
        """
        print("[INFO] Starting GRPO training...")
        
        # Configure training arguments
        training_args = self._create_training_config()
        
        # Define reward functions
        reward_functions = [
            compute_retrieval_reward,
            compute_format_reward
        ]
        
        # Initialize GRPO trainer
        trainer = GRPOReActTrainer(
            custom_args=self.args,
            react_model=self.react_model,
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.dataset,
            args=training_args,
            reward_funcs=reward_functions,
        )
        
        # Start training
        if self.args.grpo.resume:
            print(f"[INFO] Resuming training from checkpoint: {self.args.grpo.resume}")
            trainer.train(resume_from_checkpoint=self.args.grpo.resume)
        else:
            trainer.train()
        
        # Save final model and logs
        self._save_training_artifacts(trainer)
        print("[INFO] Training completed successfully!")

    def _create_training_config(self) -> GRPOConfig:
        """Create training configuration for GRPO."""
        return GRPOConfig(
            use_vllm=True,
            vllm_server_port=getattr(self.args.grpo, 'vllm_port', 8000),
            max_prompt_length=1024,
            max_completion_length=256,
            learning_rate=1e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            warmup_ratio=0.05,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            optim="paged_adamw_32bit",
            logging_steps=10,
            bf16=True,
            fp16=False,
            per_device_train_batch_size=1 if self.args.data.dataset_name == "bcplus" else 2,
            gradient_accumulation_steps=4,
            num_generations= 4 if self.args.data.dataset_name == "bcplus" else 8,
            max_steps=getattr(self.args.grpo, 'max_steps', 400),
            save_steps=500,
            output_dir=self.args.paths.output_path,
            report_to="wandb",
            temperature=0.90,
            beta=self.args.grpo.beta,
            logging_dir=self.args.paths.output_path,
            gradient_checkpointing=False,
        )

    def _save_training_artifacts(self, trainer):
        """Save model, tokenizer, and training logs."""
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        
        if local_rank == 0:  # Only save on main process
            model_name = "final"
            save_path = os.path.join(self.args.paths.output_path, model_name)
            
            print(f"[INFO] Saving model to: {save_path}")
            
            # Save model and tokenizer
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save training logs
            if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
                log_df = pd.DataFrame(trainer.state.log_history)
                log_path = os.path.join(self.args.paths.output_path, f"{model_name}_training_logs.csv")
                log_df.to_csv(log_path, sep="\t", index=False)
                print(f"[INFO] Training logs saved to: {log_path}")


# Utility Functions
def extract_passages_by_type(passages: Dict[int, List[str]], retrieval_type: str) -> Dict[int, List[str]]:
    """
    Extract titles or text from passages based on retrieval type.
    
    Args:
        passages: Dictionary mapping hop index to list of passages
        retrieval_type: Type of content to extract ('titles' or 'text')
        
    Returns:
        Dictionary mapping hop index to extracted content
    """
    extracted = {idx: [] for idx in passages.keys()}
    
    for idx, passage_list in passages.items():
        for passage in passage_list:
            if retrieval_type == 'titles':
                if args.search.rm in ["wiki20M", "wiki5M"]:
                    extracted[idx].append(passage.split("|")[0].strip())
                elif args.search.rm in ["e5-large", "e5-base"]:
                    extracted[idx].append(passage.split("\n")[0].strip())
            elif retrieval_type == 'text':
                if args.search.rm in ["wiki20M", "wiki5M"]:
                    extracted[idx].append(passage.split("|")[1].strip())
                elif args.search.rm in ["e5-large", "e5-base"]:
                    extracted[idx].append("\n".join(passage.split("\n")[1:]))
    
    return extracted


def get_titles(all_passages: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """Extract titles from passages."""
    return extract_passages_by_type(all_passages, 'titles')


def get_text(all_passages: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """Extract text content from passages."""
    return extract_passages_by_type(all_passages, 'text')


def filter_trajectory(trajectory: Dict[str, Any], get_docids: bool=False) -> Tuple[Dict[str, Any], int, Dict[int, List[str]]]:
    """
    Filter trajectory to only include steps up to the predicted finish point.
    
    Args:
        trajectory: Raw trajectory from the model
        
    Returns:
        Tuple of (filtered_trajectory, search_count, passages_retrieved)
    """
    # Find all finish locations in the trajectory
    finish_locations = []
    for key, value in trajectory.items():
        if value == "finish" and key.startswith("tool_name"):
            finish_locations.append(int(key.split("_")[-1]))
            break
    
    # Determine the predicted finish point
    predicted_finish = min(finish_locations) if finish_locations else args.model.max_iters - 1
    
    # Filter trajectory and extract passages
    filtered_trajectory = {}
    passages_retrieved = {}
    search_count = 0
    
    for key, value in trajectory.items():
        # Include steps up to predicted finish
        if "init" in key or int(key.split("_")[-1]) <= predicted_finish:
            filtered_trajectory[key] = value
            
            if key.startswith("observation") and isinstance(value, dict) and value.get("passages_this_hop"):
                search_count += 1
                current_passages = []
                
                for passage in value["passages_this_hop"]:
                    if isinstance(passage, dict) and passage.get("doc_id") and get_docids:
                        current_passages.append(passage["doc_id"])
                    elif isinstance(passage, dict) and passage.get("long_text"):
                        current_passages.append(passage["long_text"])
                    elif isinstance(passage, str):
                        current_passages.append(passage)
                
                passages_retrieved[search_count] = current_passages

    return filtered_trajectory, search_count, passages_retrieved


def _get_support_f1(retrieved_passages: List[str], flattened_gold_sentences: List[str]):
    overall_f1 = []
    for gold_chunk in flattened_gold_sentences:
        curr_best = 0.
        for pred_paragraph in retrieved_passages:
            if args.data.dataset_name in ["2wiki", "hotpot"]:
                for pred_sent in split_into_sentences(pred_paragraph):
                    curr_best = max(metrics.f1(pred_sent, gold_chunk), curr_best)
            elif args.data.dataset_name == "musique":
                curr_best = max(metrics.f1(pred_paragraph, gold_chunk), curr_best)
            else:
                raise(NotImplementedError)
                
        overall_f1.append(curr_best)
    
    overall_f1 = sum(overall_f1) / len(overall_f1)
    
    return overall_f1

def _get_coverage(retrieved_passages: List[str], flattened_gold_sentences: List[str]):
    retrieved_units = []
    if args.data.dataset_name in ["2wiki", "hotpot"]:
        for para in retrieved_passages:
            retrieved_units.extend(split_into_sentences(para))
    elif args.data.dataset_name == "musique":
        retrieved_units = retrieved_passages
    else:
        raise NotImplementedError
    
    # compute coverage of gold sentences
    covered = 0
    matched_retrieved = set()
    for gold in flattened_gold_sentences:
        best_match = 0.0
        best_idx = None
        for i, pred in enumerate(retrieved_units):
            f1_val = metrics.f1(pred, gold)
            if f1_val > best_match:
                best_match = f1_val
                best_idx = i
        if best_match > 0.7:  # threshold for "covered"
            covered += 1
            matched_retrieved.add(best_idx)

    recall = covered / len(flattened_gold_sentences)
    return recall
    

def compute_retrieval_reward(
    questions: List[str], 
    trajectories: List[Dict], 
    gold_titles: List[List[str]], 
    answers: List[str],
    gold_sent: List[Dict[str, List[str]]]
) -> List[float]:
    """
    Compute reward based on retrieval accuracy and timing.
    
    This function implements the main reward logic that encourages models to:
    1. Retrieve relevant documents (measured by title recall)
    2. Stop searching at the optimal point
    3. Avoid unnecessary search steps
    
    Args:
        questions: List of input questions
        trajectories: List of model trajectories
        gold_titles: List of gold standard titles for each question
        answers: List of expected answers
        gold_sent: List of Dict of gold standard sentences and associated titles
        
    Returns:
        List of reward scores for each trajectory
    """
    max_iterations = args.model.max_iters + 1
    rewards = []
    
    print("##"*32)
    for question, trajectory, answer, g_sent, g_title in zip(questions, trajectories, answers, gold_sent, gold_titles):
        # Process trajectory and extract information
        filtered_traj, predicted_searches, passages = filter_trajectory(trajectory, get_docids=True if args.data.dataset_name == "bcplus" else False)
        
        if args.data.dataset_name == "bcplus":
            # Get titles from retrieved passages
            retrieved_docids = [v for value in passages.values() for v in value]  
            overall_recall = metrics.recall_ret(retrieved_docids, g_title)
            baseline_recall = mb_logs.get(question, {"recall": 0.33})["recall"]
        else:
            # Get titles from retrieved passages
            retrieved_titles = get_titles(passages)
            all_retrieved_titles = [title for title_list in retrieved_titles.values() for title in title_list]
            overall_recall = metrics.title_recall(all_retrieved_titles, g_title)
            baseline_recall = mb_logs.get(question, {"recall": 0.60})["recall"]
        
        if args.data.dataset_name == "hotpot" and args.search.rm == "wiki5M":
            # Get minimum baseline from previous runs (if available)
            meets_baseline = overall_recall >= baseline_recall
            optimal_search_count = _find_optimal_search_count(retrieved_titles, g_title) if meets_baseline else -1
        
        elif args.data.dataset_name in ["bcplus"]:
            meets_baseline = overall_recall >= baseline_recall
            optimal_search_count = _find_optimal_search_count(passages, g_title) if meets_baseline else -1
            all_retrieved_titles = None
            
        elif args.data.dataset_name in ["musique", "2wiki"] or args.search.rm != "wiki5M":
            retrieved_text = get_text(passages)
            retrieved_passages = [p for p_list in retrieved_text.values() for p in p_list]
            
            g_sent = [s for sent in g_sent.values() for s in sent]
            overall_f1 = _get_support_f1(retrieved_passages, g_sent)
            
            baseline_f1 = mb_logs.get(question, {"support_f1": 0.60})["support_f1"]
            meets_baseline_f1 = overall_f1 >= baseline_f1
            meets_baseline_recall = overall_recall >= baseline_recall
            
            meets_baseline = meets_baseline_f1 or meets_baseline_recall
            
            optimal_search_count = _find_optimal_search_count_hybrid(retrieved_text, g_sent, retrieved_titles, g_title) if meets_baseline else -1
            
        else:
            raise(NotImplementedError)
        
        # Calculate reward based on search efficiency
        reward = _calculate_search_efficiency_reward(predicted_searches, optimal_search_count, max_iterations)
        
        rewards.append(reward)
        
        print('--'*16)
        print(f"Reward: {reward:.4f}")
        print(f"Predicted searches: {predicted_searches}, Num Hops: {len([k for k in filtered_traj.keys() if k.startswith('thought')])} Optimal searches: {optimal_search_count}")
        if all_retrieved_titles:
            print(f"Retrieved titles {all_retrieved_titles}")
        elif args.data.dataset_name=="bcplus" and isinstance(retrieved_docids[0], int):
            print(f"Retrieved DocIDs {retrieved_docids}")
            
        print(f"Gold titles {set(g_title)}")
        print(f"Overall recall: {overall_recall:.4f}, Max Recall: {baseline_recall:.4f}")
        print('--'*16)
        
    
    print("##"*32)
    
    return rewards


def _find_optimal_search_count_hybrid(
    retrieved_text: Dict[int, List[str]], 
    gold_sentences: List[str], 
    retrieved_titles: Dict[int, List[str]], 
    gold_titles: List[str], 
):
    """Find the earliest search step that achieves target recall/f1."""
    seen_titles, seen_passages = [], []
    best_recall, best_supp = 0.0, 0.0
    optimal_count = -1
    
    for search_count, titles in retrieved_titles.items():
        new_titles = [title for title in titles if title not in seen_titles]
        seen_titles.extend(new_titles)
        current_recall = metrics.title_recall(seen_titles, gold_titles)
        
        passages = retrieved_text[search_count]
        new_passages = [p for p in passages if p not in seen_passages]
        seen_passages.extend(new_passages)
        
        # compute support F1 with all passages retrieved so far
        current_supp = _get_coverage(seen_passages, gold_sentences)
        
        # this is a stronger signal, but if there are duplicate titles we fallback to support f1
        if current_recall > best_recall or (current_recall == best_recall and current_supp > best_supp):
            best_recall = current_recall
            best_supp = current_supp
            optimal_count = search_count
        
    return optimal_count

def _find_optimal_search_count(
    retrieved_titles: Dict[int, List[str]], 
    gold_titles: List[str], 
) -> int:
    """Find the earliest search step that achieves target recall."""
    seen_titles = []
    best_recall = 0.0
    optimal_count = -1
    
    for search_count, titles in retrieved_titles.items():
        # Add new titles (avoid duplicates)
        new_titles = [title for title in titles if title not in seen_titles]
        seen_titles.extend(new_titles)
        
        # Calculate cumulative recall
        if args.data.dataset_name == "bcplus":
            current_recall = metrics.recall_ret(seen_titles, gold_titles)
        else:
            current_recall = metrics.title_recall(seen_titles, gold_titles)
        
        if current_recall > best_recall:
            best_recall = current_recall
            optimal_count = search_count
    
    return optimal_count


def _calculate_search_efficiency_reward(
    predicted_searches: int, 
    optimal_searches: int, 
    max_iterations: int
) -> float:
    """Calculate reward based on search efficiency."""
    if optimal_searches == -1:
        # No good solution found, should have searched more
        delta = predicted_searches - max_iterations
        normalized_diff = abs(delta) / max_iterations
        clamped_diff = min(max(normalized_diff, 1e-6), 1 - 1e-6)
        reward = math.log((1 - clamped_diff) / clamped_diff)
        return max(-args.grpo.max_reward, min(reward, 0.0))
    
    else:
        # Good solution exists
        search_delta = predicted_searches - optimal_searches
        
        if search_delta > 0:
            # Over-searched - penalize
            normalized_diff = abs(search_delta) / max_iterations
            clamped_diff = min(max(normalized_diff, 1e-6), 1 - 1e-6)
            reward = math.log((1 - clamped_diff) / clamped_diff)
            return max(-args.grpo.max_reward, min(reward, args.grpo.max_reward - 0.1))
        
        elif search_delta == 0:
            # Perfect timing - reward with hardness bonus
            hardness_bonus = (optimal_searches / max_iterations) * args.grpo.zeta
            return args.grpo.max_reward + hardness_bonus
        
        else:
            # Under-searched (shouldn't happen in normal cases)
            print(f"[WARNING] Under-searched scenario detected: {search_delta}")
            return 0.0


def compute_format_reward(
    questions: List[str], 
    trajectories: List[Dict], 
    gold_titles: List[List[str]], 
    answers: List[str], 
    gold_sent: List[Dict[str, List[str]]]
) -> List[float]:
    """
    Compute reward based on format compliance and search success rate.
    
    This reward encourages the model to:
    - Use proper search format (not start with "Failed to execute")
    - Successfully execute search operations
    
    Args:
        questions: List of input questions (unused but kept for interface)
        trajectories: List of model trajectories
        gold_titles: List of gold standard titles (unused but kept for interface)
        answers: List of expected answers (unused but kept for interface)
        gold_sent: Unused param
        
    Returns:
        List of format compliance rewards
    """
    rewards = []
    
    for trajectory in trajectories:
        format_scores = []
        filtered_traj, _, _ = filter_trajectory(trajectory)
        
        # Check format compliance for each search step
        for key, value in filtered_traj.items():
            if (key.startswith("tool_name") and value != "finish"):
                # Reward successful searches, penalize failed ones
                if value.startswith("Failed to execute"):
                    format_scores.append(-args.grpo.format_coeff)
                else:
                    format_scores.append(args.grpo.format_coeff)
        
        # Calculate average format reward
        if format_scores:
            avg_reward = sum(format_scores) / len(format_scores)
        else:
            avg_reward = 0.0
        
        rewards.append(avg_reward)
        
    print(f"Format Rewards: {rewards}")
    
    return rewards


# Legacy function aliases for backward compatibility
def v1_reward_func(questions, trajectories, gold_titles, answers, gold_sent):
    """Alias for compute_retrieval_reward."""
    return compute_retrieval_reward(questions, trajectories, gold_titles, answers, gold_sent)


def gt_reward(questions, trajectories, gold_titles, answers, gold_sent):
    """Alias for compute_retrieval_reward."""
    return compute_retrieval_reward(questions, trajectories, gold_titles, answers, gold_sent)


def format_reward(questions, trajectories, gold_titles, answers, gold_sent):
    """Alias for compute_format_reward."""
    return compute_format_reward(questions, trajectories, gold_titles, answers, gold_sent)

def setup_wandb(args):
    """Initialize Weights & Biases logging."""
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    os.system("export WANDB_BASE_URL='https://api.wandb.ai'")
    
    if local_rank == 0:  # Only initialize on main process
        wandb_dir = os.path.join(args.paths.output_path, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        
        wandb.init(
            project=args.grpo.project_name,
            dir=wandb_dir,
            name=args.grpo.run_name,
            config=vars(args)
        )
        print(f"[INFO] Wandb initialized: project={args.grpo.project_name}, run={args.grpo.run_name}")


def load_baseline_logs(log_path: str) -> Dict[str, Dict]:
    """Load minimum baseline logs for reward computation."""
    if not log_path or not os.path.exists(log_path):
        print("[WARNING] No baseline logs provided or file not found")
        raise()
    
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
        
        # Convert to question -> metrics mapping
        baseline_logs = {
            example["question"]: example["metrics"] 
            for example in logs 
            if "question" in example and "metrics" in example
        }
        
        print(f"[INFO] Loaded baseline logs for {len(baseline_logs)} questions")
        return baseline_logs
        
    except Exception as e:
        print(f"[ERROR] Failed to load baseline logs from {log_path}: {e}")
        return {}


def main(args):
    """Main function."""    
    # Setup distributed training environment
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    # Initialize logging
    setup_wandb(args)
    
    # Initialize and run trainer
    trainer = CustomTrainer(args)
    trainer.train()
    
    print("[INFO] GRPO training completed successfully!")


if __name__ == "__main__":
    # globals
    args = get_args("grpo")
    mb_logs = load_baseline_logs(args.paths.mb_logs)
    
    main(args)
