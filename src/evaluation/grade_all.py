
import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm

from src.prompts.generation_prompt import create_judge_prompt, parse_judge_response
from vllm import LLM, SamplingParams

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_grader():
    """
    Grader model is loaded with vLLM for faster inference
    """
    llm = LLM(
        model="Qwen/Qwen3-32B",
        tensor_parallel_size=8,
        trust_remote_code=True,
        max_model_len=4096*4,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0, 
        top_p=1.0,
        max_tokens=4096,
        stop=None,
    )
    
    return llm, sampling_params

def find_log_files(root_directory: str) -> List[str]:
    """
    Recursively find all log files:
    - logs.json
    - logs_*.json (e.g., logs_basic.json, logs_fs.json, logs_corag.json)
    - intermediate_data.json
    """
    log_files = []
    
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            # Match logs.json, logs_something.json, or intermediate_data.json
            if (file == 'logs.json' or 
                (file.startswith('logs_') and file.endswith('.json')) or 
                file == 'intermediate_data.json'):
                if "logs_doc" in file or file=="logs.json":
                    log_files.append(os.path.join(root, file))
                elif "intermediate" in file:
                    log_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(log_files)} log files")
    return log_files

def determine_judge_filename(log_file: str) -> str:
    """
    Determine the output filename for judge results based on log file name
    """
    log_dir = os.path.dirname(log_file)
    log_filename = os.path.basename(log_file)
    
    if log_filename.startswith('logs_'):
        # Extract suffix from logs_something.json -> judge_something.json
        suffix = log_filename[5:]  # Remove 'logs_' prefix
        if suffix.endswith('.json'):
            suffix = suffix[:-5]  # Remove '.json' extension
        judge_filename = f"judge_{suffix}.json" if suffix else "judge.json"
    else:
        # Default to judge.json
        judge_filename = "judge.json"
    
    return os.path.join(log_dir, judge_filename)

def load_log_data(log_file: str) -> List[Dict[str, Any]]:
    """
    Load data from log file
    """
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.warning(f"Expected list in {log_file}, got {type(data)}")
            return []
        
        return data
    except Exception as e:
        logger.error(f"Error loading {log_file}: {e}")
        return []

def extract_question_answer_pred(item: Dict[str, Any]) -> Optional[tuple]:
    """
    Extract question, correct answer, and prediction from a log item
    """
    try:
        # Extract question
        question = item.get('question', '')
        if not question:
            return None
        
        # Extract prediction
        if item.get("output"):
            pred = item["output"].get("pred", '')
        else:
            pred = item.get('pred', '')
        if not pred:
            return None
        
        # Extract correct answer - try multiple possible keys
        correct_answer = None
        if 'answer' in item:
            correct_answer = item['answer']
        elif 'gold_answers' in item:
            correct_answer = item['gold_answers']
        elif 'golden_answers' in item:
            correct_answer = item['golden_answers']
        
        if correct_answer is None:
            return None
        
        # Ensure correct_answer is a string for grading
        if isinstance(correct_answer, list):
            # Join list answers or take first one
            if len(correct_answer) > 0:
                correct_answer = correct_answer[0] if len(correct_answer) == 1 else ' | '.join(str(x) for x in correct_answer)
            else:
                return None
        
        return question, str(correct_answer), str(pred)
    
    except Exception as e:
        logger.debug(f"Error extracting data from item: {e}")
        return None

def grade_batch(llm, sampling_params, batch_data: List[tuple]) -> List[Dict[str, Any]]:
    """
    Grade a batch of questions using the LLM
    """    
    prompts = []
    for question, correct_answer, pred in batch_data:
        prompt = create_judge_prompt(question, pred, correct_answer)
        prompts.append(prompt)

    messages_list = [[{"role": "user", "content": p}] for p in prompts]

    try:
        outputs = llm.chat(messages_list, sampling_params)
        results = []
        
        for i, output in enumerate(outputs):
            question, correct_answer, pred = batch_data[i]
            judge_response = output.outputs[0].text if output.outputs else ""
                
            parsed_result = parse_judge_response(judge_response)
            
            # Add original data to result
            result = {
                'question': question,
                'correct_answer': correct_answer,
                'pred': pred,
                'judge_response': judge_response,
                **parsed_result
            }
            results.append(result)
        
        return results
    
    except Exception as e:
        logger.error(f"Error in batch grading: {e}")
        return []

def process_log_file(log_file: str, llm, sampling_params, batch_size: int = 8) -> None:
    """
    Process a single log file and save judge results
    """
    logger.info(f"Processing {log_file}")
    
    # Determine output file
    judge_file = determine_judge_filename(log_file)
    
    # Skip if judge file already exists
    if os.path.exists(judge_file):
        logger.info(f"Judge file {judge_file} already exists, skipping")
        return
    
    # Load log data
    log_data = load_log_data(log_file)
    if not log_data:
        logger.warning(f"No data found in {log_file}")
        return
    
    # Extract questions, answers, and predictions
    valid_items = []
    for item in log_data:
        extracted = extract_question_answer_pred(item)
        if extracted:
            valid_items.append(extracted)
    
    if not valid_items:
        logger.warning(f"No valid items found in {log_file}")
        return
    
    logger.info(f"Found {len(valid_items)} valid items to grade in {log_file}")
    
    if len(valid_items) > 1000:
        valid_items = valid_items[:1000]
    
    # Process in batches
    all_results = []
    for i in tqdm(range(0, len(valid_items), batch_size), desc=f"Grading {os.path.basename(log_file)}"):
        batch = valid_items[i:i + batch_size]
        batch_results = grade_batch(llm, sampling_params, batch)
        all_results.extend(batch_results)
    
    # Calculate summary statistics
    total_items = len(all_results)
    correct_items = sum(1 for result in all_results if result.get('correct', False))
    parse_errors = sum(1 for result in all_results if result.get('parse_error', False))
    
    summary = {
        'total_items': total_items,
        'correct_items': correct_items,
        'accuracy': correct_items / total_items if total_items > 0 else 0.0,
        'parse_errors': parse_errors,
        'source_log_file': log_file
    }
    
    # Save results
    output_data = {
        'summary': summary,
        'results': all_results
    }
    
    try:
        with open(judge_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved judge results to {judge_file} (Accuracy: {summary['accuracy']:.3f})")
    except Exception as e:
        logger.error(f"Error saving judge results to {judge_file}: {e}")

def main(root_directories: List[str], batch_size: int = 8, max_workers: int = 2):
    """
    Main function to process all log files
    """
    # Initialize grader
    logger.info("Initializing grader model...")
    llm, sampling_params = setup_grader()
    
    # Find all log files
    all_log_files = []
    for root_dir in root_directories:
        if os.path.exists(root_dir):
            log_files = find_log_files(root_dir)
            all_log_files.extend(log_files)
            logger.info(f"Found {len(log_files)} log files in {root_dir}")
        else:
            logger.warning(f"Directory {root_dir} does not exist")
    
    if not all_log_files:
        logger.error("No log files found")
        return
    
    logger.info(f"Total log files to process: {len(all_log_files)}")
    
    # Process log files
    if max_workers == 1:
        # Sequential processing
        for log_file in all_log_files:
            process_log_file(log_file, llm, sampling_params, batch_size)
    else:
        # Note: With vLLM, parallel processing might not be beneficial due to GPU memory constraints
        # Processing sequentially for now, but keeping the structure for potential future improvements
        logger.info("Processing files sequentially due to GPU memory constraints")
        for log_file in all_log_files:
            process_log_file(log_file, llm, sampling_params, batch_size)

if __name__ == "__main__":
    root_directories = [
        # "/mnt/ddn/alta01/t-abjava/data/frugalrag/eval/grpo/e5/",
        "/mnt/ddn/alta01/t-abjava/data/frugalrag/eval/grpo/colbert/hotpot/llama8b",
        "/mnt/ddn/alta01/t-abjava/data/frugalrag/eval/grpo/colbert/2wiki/llama8b",
        "/mnt/ddn/alta01/t-abjava/data/frugalrag/eval/grpo/colbert/musique/llama8b",
        "../data/frugalrag/eval/base/colbert/hotpot/qwen7b/"
        # "/mnt/ddn/alta01/t-abjava/data/frugalrag/eval/sft/colbert/hotpot/qwen7b/m5_0.90",
        # "/mnt/ddn/alta01/t-abjava/data/frugalrag/eval/sft/colbert/musique/qwen7b/m5_0.90",
        # "/mnt/ddn/alta01/t-abjava/data/frugalrag/eval/sft/colbert/",
        # "/mnt/ddn/alta01/t-abjava/flashrag_outputs"
    ]
    
    main(root_directories, batch_size=256, max_workers=1)