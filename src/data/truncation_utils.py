import re
import math
import json
from typing import Dict, Any, Optional
import dspy

def format(trajectory: dict[str, Any]):
    adapter = dspy.settings.adapter or dspy.ChatAdapter()
    trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
    return adapter.format_fields(trajectory_signature, trajectory, role="user")


# -------------------------
# Helpers: token estimation & clipping
# -------------------------
def estimate_tokens(text: str, tokenizer=None) -> int:
    """Estimate tokens. If tokenizer provided, use it, else fallback to words * 1.3."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            toks = tokenizer.encode if hasattr(tokenizer, "encode") else tokenizer.__call__
            return len(toks(text))
        except Exception:
            pass

    return max(1, int(len(text.split()) * 1.3))

def estimate_prompt_overhead(question: str = "", tokenizer=None) -> int:
    """
    Estimate the token overhead from prompts, instructions, and formatting.
    This includes the ReAct prompting structure, tool descriptions, etc.
    """
    # Base overhead for ReAct prompting structure
    base_overhead = 5000
    
    # Add question tokens
    question_tokens = estimate_tokens(question, tokenizer) if question else 100
    
    # Add some buffer for tool descriptions, formatting, etc.
    formatting_overhead = 1000
    
    return base_overhead + question_tokens + formatting_overhead

def get_adaptive_token_budget(question: str = "", max_seq_length: int = 20000, tokenizer=None) -> int:
    """
    Calculate the available token budget for trajectory content after accounting for prompts.
    """
    prompt_overhead = estimate_prompt_overhead(question, tokenizer)
    available_budget = max_seq_length - prompt_overhead
    
    # Ensure we have at least some reasonable budget
    return max(available_budget, 10000)

def basic_truncate_trajectory(trajectory, last_n=4):
    print("[INFO] Reducing input context length manually...")
    # Keep only the last observations, replace others with corresponding thought content
    observation_keys = [k for k in trajectory.keys() if k.startswith('observation_') and not k.endswith('_init')]
    observation_keys.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else -1)
    
    # Replace all but the last observation with their corresponding thought content
    if len(observation_keys) > last_n:
        for key in observation_keys[:-last_n]:
            # Extract the index from observation key (e.g., 'observation_4' -> '4')
            idx_str = key.split('_')[1]
            
            observation = trajectory[key]
            for i, passage_dict in enumerate(observation["passages_this_hop"]):
                passage_dict["long_text"] = "\n".join(passage_dict["long_text"].split("\n")[:2]) + "\n..."
                observation["passages_this_hop"][i] = passage_dict

            trajectory[key] = observation
    
    return trajectory

def compute_trajectory_tokens(trajectory: dict, tokenizer):
    trajectory_string = format(trajectory)
    return estimate_tokens(trajectory_string, tokenizer)

def smart_truncate(trajectory: dict, tokenizer=None, max_seq_length: int = 20000, question: str = "", last_n: int = 4) -> dict:
    """
    Adaptively truncate trajectory to stay within token limits while preserving essential information.
    
    Args:
        trajectory: The trajectory dictionary to truncate
        tokenizer: Optional tokenizer for accurate token counting
        max_seq_length: Maximum sequence length (default 35k)
        question: The original question (used for prompt overhead calculation)
        last_n: Number of most recent observations to preserve fully
        
    Returns:
        Truncated trajectory dictionary
    """
    # Calculate available token budget
    max_tokens = get_adaptive_token_budget(question, max_seq_length, tokenizer)
    
    # Make a copy to avoid modifying the original
    trajectory = trajectory.copy()
    
    # Initial token count
    current_tokens = compute_trajectory_tokens(trajectory, tokenizer)
    
    if current_tokens <= max_tokens:
        return trajectory
    
    print(f"[INFO] Smart truncation: {current_tokens} tokens -> target: {max_tokens} (budget after {estimate_prompt_overhead(question, tokenizer)} prompt overhead)")
    
    # Get all observation keys and sort by index
    observation_keys = [k for k in trajectory.keys() if k.startswith('observation_') and not k.endswith('_init')]
    observation_keys.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else -1)
    
    # Strategy 1: Truncate passages in older observations (keep last_n fully intact)
    if len(observation_keys) > last_n:
        for key in observation_keys[:-last_n]:
            if key in trajectory and "passages_this_hop" in trajectory[key]:
                passages = trajectory[key].get("passages_this_hop", []) if isinstance(trajectory[key], dict) else []
                
                # Progressively truncate passages
                for i, passage in enumerate(passages):
                    if "long_text" in passage:
                        # Keep first 2-3 sentences + ellipsis
                        sentences = passage["long_text"].split(". ")
                        if len(sentences) > 3:
                            truncated_text = ". ".join(sentences[:3]) + "... [truncated]"
                            trajectory[key]["passages_this_hop"][i]["long_text"] = truncated_text
                
                # Check if we're within limits now
                current_tokens = compute_trajectory_tokens(trajectory, tokenizer)
                if current_tokens <= max_tokens:
                    print(f"[INFO] Truncation successful after passage reduction: {current_tokens} tokens")
                    return trajectory
    
    # Strategy 2: Reduce number of passages in older observations
    if current_tokens > max_tokens and len(observation_keys) > last_n:
        for key in observation_keys[:-last_n]:
            if key in trajectory and "passages_this_hop" in trajectory[key]:
                passages = trajectory[key].get("passages_this_hop", []) if isinstance(trajectory[key], dict) else []
                
                # Keep only top 2-3 passages (by score if available)
                if len(passages) > 3:
                    # Sort by score if available, otherwise keep first few
                    if all("score" in p for p in passages):
                        passages.sort(key=lambda x: x.get("score", 0), reverse=True)
                    trajectory[key]["passages_this_hop"] = passages[:3]
                
                # Check if we're within limits now
                current_tokens = compute_trajectory_tokens(trajectory, tokenizer)
                if current_tokens <= max_tokens:
                    print(f"[INFO] Truncation successful after passage count reduction: {current_tokens} tokens")
                    return trajectory
    
    # Strategy 3: More aggressive truncation - remove older observations entirely
    if current_tokens > max_tokens:
        # Calculate how many observations to keep
        target_observations = max(last_n, 2)  # Keep at least 2 observations
        
        while len(observation_keys) > target_observations and current_tokens > max_tokens:
            # Remove the oldest observation (and related thought/tool entries)
            oldest_key = observation_keys.pop(0)
            idx = oldest_key.split('_')[1]
            
            # Remove all entries for this index
            keys_to_remove = [k for k in trajectory.keys() if k.endswith(f'_{idx}')]
            for k in keys_to_remove:
                if k in trajectory:
                    del trajectory[k]
            
            current_tokens = compute_trajectory_tokens(trajectory, tokenizer)
            print(f"[INFO] Removed observation {idx}, current tokens: {current_tokens}")
    
    # Strategy 4: Final aggressive truncation of remaining passages if still over limit
    if current_tokens > max_tokens:
        for key in trajectory.keys():
            if key.startswith('observation_') and "passages_this_hop" in trajectory[key]:
                passages = trajectory[key].get("passages_this_hop", []) if isinstance(trajectory[key], dict) else []
                
                for i, passage in enumerate(passages):
                    if "long_text" in passage:
                        # Very aggressive truncation - keep only first sentence
                        first_sentence = passage["long_text"].split(".")[0] + "... [heavily truncated]"
                        trajectory[key]["passages_this_hop"][i]["long_text"] = first_sentence
                
                current_tokens = compute_trajectory_tokens(trajectory, tokenizer)
                if current_tokens <= max_tokens:
                    break
    
    if current_tokens > max_tokens:        
        # Get all keys sorted by their index (oldest first)
        all_indexed_keys = []
        for key in trajectory.keys():
            if '_' in key and key.split('_')[-1].isdigit():
                idx = int(key.split('_')[-1])
                all_indexed_keys.append((idx, key))
        
        # Sort by index (oldest first)
        all_indexed_keys.sort(key=lambda x: x[0])
        
        # Remove keys starting from oldest until we're within budget
        # But always keep at least the most recent observation
        min_keys_to_keep = 2
        
        while current_tokens > max_tokens and len(all_indexed_keys) > min_keys_to_keep:
            # Remove the oldest indexed key
            oldest_idx, oldest_key = all_indexed_keys.pop(0)
            
            if oldest_key in trajectory:
                del trajectory[oldest_key]
                # print(f"[INFO] Removed key: {oldest_key}")
            
            current_tokens = compute_trajectory_tokens(trajectory, tokenizer)
            print(f"[INFO] After removing {oldest_key}: {current_tokens} tokens")
            
            if current_tokens <= max_tokens:
                break
    
    final_tokens = compute_trajectory_tokens(trajectory, tokenizer)
    print(f"[INFO] Smart truncation completed: {final_tokens} tokens (target: {max_tokens})")
    
    assert final_tokens <= max_seq_length
    
    return trajectory

def adaptive_truncate_for_react(trajectory: dict, question: str = "", tokenizer=None, max_seq_length: int = 20000, preserve_recent: int = 4) -> dict:
    """
    Convenience function for ReAct model integration.
    
    Args:
        trajectory: The trajectory dictionary from ReAct
        question: Original question for prompt overhead estimation
        tokenizer: Tokenizer for accurate token counting
        max_seq_length: Maximum sequence length allowed
        preserve_recent: Number of recent observations to preserve fully
        
    Returns:
        Truncated trajectory ready for use in ReAct model
    """
    return smart_truncate(
        trajectory=trajectory,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        question=question,
        last_n=preserve_recent
    )
    