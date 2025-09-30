import os
import json
import time
import warnings
import threading
import traceback
from typing import Dict, List, Optional, Any, Union

import torch
import dspy
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from src.data.data_utils import get_eval_dataset, split_into_sentences
from src.evaluation import metrics
from src.model import Hopper
from src.parser_ import get_args
from openai import OpenAI
from src.prompts.generation_prompt import *


warnings.filterwarnings("ignore")


class Evaluator:
    def __init__(self, args):
        """
        Initialize the evaluator with configuration arguments.
        
        Args:
            args: Configuration object containing model and evaluation parameters
        """
        self.args = args
        self.data, self.gold_titles, self.gold_sent = get_eval_dataset(args)
        if args.data.num_test > -1:
            self.data = self.data[:args.data.num_test]
        self.model = Hopper(args, custom=True)

        if args.paths.prompt_path:
            self.model.update_react_prompt(args.paths.prompt_path)

    def run_model(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Run the model on a single question with error handling.
        
        Args:
            question: The input question string
            
        Returns:
            Model output dictionary or None if an error occurs
        """
        try:
            if hasattr(self.args.model, 'answer_model') and self.args.model.answer_model:
                # Global variables lm_answer and lm_query should be passed as parameters
                # This is a design issue that should be addressed later
                return self.model(question=question, lm_answer=answer_generator, lm_query=reasoner, get_original_traj=True if self.args.model.max_iters > 5 else None)
            else:
                return self.model(question=question, get_original_traj=True if self.args.model.max_iters > 5 else None)
        except Exception as e:
            print(f"Exception occurred while processing question: {question}")
            traceback.print_exc()
            return None

    def run_model_with_timeout(
        self, 
        question: str, 
        answer_generator: Optional[Any] = None, 
        reasoner: Optional[Any] = None, 
        timeout: int = 240
    ) -> Optional[Dict[str, Any]]:
        """
        Run model with timeout mechanism to prevent hanging.
        
        Args:
            question: Input question string
            answer_generator: Language model for answer generation
            reasoner: Language model for sub-query generation ()
            timeout: Timeout in seconds (default: 240)
            
        Returns:
            Model output dictionary or None if timeout/error occurs
        """
        result = {}

        def target():
            """Target function to run in separate thread."""
            try:
                if hasattr(self.args.model, 'answer_model') and self.args.model.answer_model:
                    result["output"] = self.model(
                        question=question, lm_answer=answer_generator, lm_query=reasoner, get_original_traj=True if self.args.model.max_iters > 5 else None
                    )
                else:
                    result["output"] = self.model(question=question)
            except Exception as e:
                print("Model exception (will retry next batch):")
                traceback.print_exc()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            print(f"Timeout after {timeout}s on question: {question!r}")
            return None
            
        if "error" in result:
            raise result["error"]

        return result.get("output")

    def _initialize_metrics(self) -> Dict[str, List]:
        """Initialize empty metrics dictionary."""
        return {
            "match": [],
            "em": [],
            "f1": [],
            "recall": [],
            "support_f1": [],
            "recall_flashrag": [],
            "precision_flashrag": [],
            "num_hops": [],
            "num_searches": [],
        }

    def _load_existing_logs(self) -> tuple[List[Dict], List[str]]:
        """
        Load existing logs from file if they exist.
        
        Returns:
            Tuple of (logs_list, solved_questions_list)
        """
        path = "logs.json"
            
        logs_path = os.path.join(self.args.paths.output_path, path)
        if not os.path.exists(logs_path):
            return [], []
            
        try:
            with open(logs_path) as f:
                curr_logs = json.load(f)
                
            # Filter logs with valid predictions
            logs = [log for log in tqdm(curr_logs) if len(log.get("pred_titles", [])) > 0]
            solved_questions = [log["question"] for log in logs]
            
            print(f"Loaded {len(logs)} existing results")
            return logs, solved_questions
            
        except Exception as e:
            print(f"Error loading existing logs: {e}")
            return [], []

    def _update_metrics_from_logs(self, metric_dict: Dict[str, List], logs: List[Dict]) -> None:
        """Update metric dictionary with values from existing logs."""
        print(f"Updating metrics from {len(logs)} logs...")
        for log in logs:
            metrics_data = log.get("metrics", {})
            for key in metric_dict:
                if key in metrics_data:
                    metric_dict[key].append(metrics_data[key])
        
        # Print current metric counts
        for key, values in metric_dict.items():
            if values:
                print(f"  {key}: {len(values)} values, avg: {sum(values)/len(values):.4f}")

    def _prepare_evaluation_pairs(self, solved_questions: List[str]) -> List[tuple]:
        """
        Prepare pairs for parallel processing.
        
        Args:
            solved_questions: List of already processed questions to skip
            
        Returns:
            List of (function, args) tuples for parallel execution
        """
        return [
            (
                self.process_example,
                {
                    "question": example["question"],
                    "_id": example["_id"],
                    "answer": example["answer"],
                },
            )
            for example in self.data
            if example["question"] not in solved_questions
        ]

    def _calculate_average_metrics(self, metric_dict: Dict[str, List]) -> Dict[str, float]:
        """Calculate average metrics from accumulated values."""
        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in metric_dict.items()
        }

    def _save_results(self, metric_dict: Dict[str, List], logs: List[Dict]) -> None:
        """Save average metrics and logs to files."""
        # Save average metrics
        avg_metrics = self._calculate_average_metrics(metric_dict)
        avg_metrics_path = os.path.join(self.args.paths.output_path, f"avg_metrics_{self.args.model.prompt_version}.json") if args.model.answer_only else os.path.join(self.args.paths.output_path, "avg_metrics.json")
        with open(avg_metrics_path, "w") as f:
            json.dump(avg_metrics, f, indent=2)

        # Save detailed logs
        if self.args.model.answer_only:
            path = f"logs_{self.args.model.prompt_version}.json"
        else:
            path = "logs.json"
            
        logs_path = os.path.join(self.args.paths.output_path, path)
        with open(logs_path, "w") as f:
            json.dump(logs, f, indent=2)

    def _process_results_batch(
        self, 
        results: List[Optional[Dict]], 
        metric_dict: Dict[str, List], 
        logs: List[Dict]
    ) -> None:
        """Process a batch of results and update metrics and logs."""
        for log_dict in results:
            if log_dict is not None:
                # Update metrics
                metrics_data = log_dict.get("metrics", {})
                for key in metric_dict:
                    if key in metrics_data:
                        metric_dict[key].append(metrics_data[key])
                
                logs.append(log_dict)

    def run_loop(self, answer_only=False) -> None:
        """Main evaluation loop with parallel processing and progress tracking."""
        # Initialize
        metric_dict = self._initialize_metrics()
        logs, solved_questions = self._load_existing_logs()
        logs_to_process = None
        if answer_only:
            # Setup vLLM for answer generation
            llm, sampling_params = setup_answer_generator(args)

            prompts = []
            logs_to_process = logs[:args.data.num_test] if args.data.num_test > 0 else logs
            for log in logs_to_process:
                traj = log["trajectory"]
                thoughts = [v for k, v in traj.items() if "thought" in k]
                documents = [
                    p["long_text"]
                    for k, v in traj.items()
                    if "observation" in k and isinstance(v, dict) and v.get("passages_this_hop")
                    for p in v["passages_this_hop"]
                ]
                prompts.append(get_prompt(
                    args, log["question"], log["search_queries"], thoughts, log["rationale"], documents
                ))

            # Generate responses using vLLM (processes all prompts efficiently)
            print(f"Generating responses for {len(prompts)} prompts using vLLM...")
            messages_list = [[{"role": "user", "content": p}] for p in prompts]
            outputs = llm.chat(messages_list, sampling_params)
            if self.args.model.prompt_version in ["doc_cot", "v1_cot", "rag_prompt_oai", "v1_oai"]:
                decoded_all = [extract_chout(output.outputs[0].text.strip()) for output in outputs]            
            else:
                decoded_all = [output.outputs[0].text.strip() for output in outputs]
            
            # Assign back and recalculate metrics
            print(f"Processing {len(logs_to_process)} logs for answer-only evaluation...")
            for i, (log, pred) in enumerate(zip(logs_to_process, decoded_all)):
                log["pred"] = pred.strip()
                
                # eval
                answer = log["answer"]
                answer_list = [answer] if isinstance(answer, str) else answer
                predicted_answer = log["pred"]

                # Calculate answer metrics only
                f1_scores = [metrics.f1(predicted_answer, a) for a in answer_list]
                em_scores = [metrics.em(predicted_answer, a) for a in answer_list]
                match_scores = [metrics.match(predicted_answer, a) for a in answer_list]
                
                _, pred_passages, _, _ = self._extract_trajectory_info(log["trajectory"])
                
                recall_flash = metrics.recall_flashrag(pred_passages, answer_list)
                precision_flash = metrics.precision_flashrag(pred_passages, answer_list)
                
                # Update metrics in log
                log["metrics"]["f1"] = max(f1_scores) if f1_scores else 0.0
                log["metrics"]["em"] = max(em_scores) if em_scores else 0.0
                log["metrics"]["match"] = max(match_scores) if match_scores else 0.0
                
                log["metrics"]["recall_flashrag"] = recall_flash
                log["metrics"]["precision_flashrag"] = precision_flash
                
                if i % 50 == 0:  # Print progress every 50 items
                    print(f"Processed {i+1}/{len(logs_to_process)} - F1: {log['metrics']['f1']:.3f}, EM: {log['metrics']['em']:.3f}")

        logs_to_process = logs if logs_to_process is None else logs_to_process
        # Update metrics from logs (including updated answer_only metrics)
        self._update_metrics_from_logs(metric_dict, logs_to_process)
        
        print(f"Already solved: {len(solved_questions)} questions")

        # Prepare evaluation pairs
        pairs = self._prepare_evaluation_pairs(solved_questions)
        
        if args.model.answer_only:
            print("Answer-only mode: saving updated results...")
            self._save_results(metric_dict, logs_to_process)
            
            # Print summary
            final_metrics = self._calculate_average_metrics(metric_dict)
            print(f"\nEvaluation completed! Processed {len(logs_to_process)} questions.")
            print("Final metrics:")
            for key, value in final_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            return
        
        if not pairs:
            print("All questions already processed!")
            return

        # Process with multiple threads
        print(f"Processing {len(pairs)} remaining questions...")
        executor = dspy.parallel.Parallel(
            num_threads=min(32, len(pairs)), 
            max_errors=10,
            provide_traceback=True
        )
        results = executor.forward(pairs)
        self._process_results_batch(results, metric_dict, logs)

        # Save intermediate results
        self._save_results(metric_dict, logs)

        # Retry failed cases with single thread
        successful_questions = {log["question"] for log in logs}
        retry_pairs = [
            pair for pair in pairs 
            if pair[1]["question"] not in successful_questions
        ]
        
        if retry_pairs:
            print(f"Retrying {len(retry_pairs)} failed questions with single thread...")
            single_executor = dspy.parallel.Parallel(
                num_threads=1, 
                max_errors=10, 
                provide_traceback=True
            )
            retry_results = single_executor.forward(retry_pairs)
            self._process_results_batch(retry_results, metric_dict, logs)

        # Save final results
        self._save_results(metric_dict, logs)
        
        # Print summary
        final_metrics = self._calculate_average_metrics(metric_dict)
        print(f"\nEvaluation completed! Processed {len(logs)} questions.")
        print("Final metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")

    def _extract_trajectory_info(self, trajectory: Dict[str, Any]) -> tuple[List[str], List[str], List[str], int]:
        """
        Extract information from model trajectory.
        
        Args:
            trajectory: Model trajectory dictionary
            
        Returns:
            Tuple of (pred_titles, pred_passages, search_queries, num_hops)
        """
        pred_titles, pred_passages, search_queries, max_hops = [], [], [], []
        
        for key, value in trajectory.items():
            # Calculate hop number
            if "init" in key:
                max_hops.append(0)
            else:
                max_hops.append(int(key.split("_")[-1]) + 2)

            # Extract search information
            if key.startswith("observation") and isinstance(value, dict):
                search_queries.append(value.get("search_query", ""))
                
                passages = value.get("passages_this_hop", [])
                for passage in passages:
                    if isinstance(passage, dict):
                        if self.args.search.rm in ["wiki5M", "wiki20M"]:
                            title = passage.get("long_text", "").split("|")[0].strip()
                            content = passage.get("long_text", "").split("|")[1].strip() if "|" in passage.get("long_text", "") else ""
                        elif self.args.search.rm in ["e5-large", "e5-base"]:
                            title = str(passage.get("long_text", "")).split("\n")[0].strip()
                            content = str(passage.get("long_text", ""))
                        elif "qwen" in self.args.search.rm and self.args.data.dataset_name == "bcplus":
                            title = int(passage["doc_id"])
                            content = str(passage.get("long_text", "")) 
                    
                    pred_titles.append(title)
                    pred_passages.append(content)

        num_hops = max(max_hops) if max_hops else 0
        return pred_titles, pred_passages, search_queries, num_hops

    def _calculate_support_f1(self, pred_passages: List[str], gold_sent: List[str]) -> float:
        """Calculate support F1 score between predicted passages and gold sentences."""
        if not gold_sent:
            return 0.0
            
        support_f1_scores = []
        for sent in gold_sent:
            max_f1 = 0.0
            for pred_passage in pred_passages:
                if self.args.data.dataset_name in ["2wiki", "hotpot"]:
                    for pred_sent in split_into_sentences(pred_passage):
                        f1_score = metrics.f1(pred_sent.strip(), sent)
                        max_f1 = max(max_f1, f1_score)
                elif self.args.data.dataset_name in ["musique"]:
                    f1_score = metrics.f1(pred_passage.strip(), sent)
                    max_f1 = max(max_f1, f1_score)
                elif self.args.data.dataset_name == "bcplus":
                    max_f1 = -1
                    f1_score = -1
                else:
                    raise(NotImplementedError)
                        
            support_f1_scores.append(max_f1)
        
        return sum(support_f1_scores) / len(support_f1_scores)

    def process_example(self, question: str, answer: Union[str, List[str]], _id: str) -> Optional[Dict[str, Any]]:
        """
        Process a single evaluation example.
        
        Args:
            question: Input question
            answer: Gold answer(s) 
            _id: Example identifier
            
        Returns:
            Log dictionary with results and metrics, or None if processing fails
        """
        gold_titles = self.gold_titles.get(_id, [])
        gold_sent = self.gold_sent.get(_id, [])

        # Run model with timeout
        timeout = 2000 if self.args.model.max_iters > 10 else 240
        output = self.run_model_with_timeout(question, answer_generator, reasoner, timeout=timeout)
        if output is None:
            return None

        if self.args.model.max_iters > 5:
            output, trajectory = output
        else:
            trajectory = output.get("trajectory", {})
        
        rationale = output.get("reasoning", [])

        # Check for API errors in trajectory
        for key, value in trajectory.items():
            if isinstance(value, str) and "litellm" in value.lower():
                print(f"API error detected in trajectory for question: {question}")
                return None

        # Extract trajectory information
        pred_titles, pred_passages, search_queries, num_hops = self._extract_trajectory_info(trajectory)

        # Prepare answer for comparison
        answer_list = [answer] if isinstance(answer, str) else answer
        predicted_answer = output.get("answer", "")

        # Calculate metrics
        f1_scores = [metrics.f1(predicted_answer, a) for a in answer_list]
        em_scores = [metrics.em(predicted_answer, a) for a in answer_list]
        match_scores = [metrics.match(predicted_answer, a) for a in answer_list]
        
        if self.args.data.dataset_name in ["2wiki", "musique", "hotpot"]:
            support_f1 = self._calculate_support_f1(pred_passages, gold_sent)
            recall = metrics.title_recall(pred_titles, gold_titles)
        elif self.args.data.dataset_name in ["bcplus"]: # in bcplus, gold_titles refers to evidence passages
            support_f1 = -1
            # this is document level recall in bcplus
            recall = metrics.recall_ret(pred_titles, gold_titles)
        
        recall_flash = metrics.recall_flashrag(pred_passages, answer_list)
        precision_flash = metrics.precision_flashrag(pred_passages, answer_list)

        # Build result dictionary
        log_dict = {
            "question": question,
            "answer": answer,
            "pred": predicted_answer,
            "gold_titles": gold_titles,
            "pred_titles": pred_titles,
            "search_queries": search_queries,
            "num_searches": len(search_queries),
            "num_hops": num_hops,
            "trajectory": trajectory,
            "rationale": rationale,
            "metrics": {
                "f1": max(f1_scores) if f1_scores else 0.0,
                "em": max(em_scores) if em_scores else 0.0,
                "match": max(match_scores) if match_scores else 0.0,
                "recall": recall,
                "recall_flashrag": recall_flash,
                "precision_flashrag": precision_flash,
                "support_f1": support_f1,
                "num_searches": len(search_queries),
                "num_hops": num_hops,
            }
        }

        print(f"[INFO] metrics: f1={max(f1_scores)}, recall={recall}")
        return log_dict

def setup_answer_generator(args):
    """
    Answer model is loaded with vLLM for faster inference
    """
    llm = LLM(
        model=args.model.model_name_or_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=4096*4,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0, 
        top_p=1.0,
        max_tokens=1024,
        stop=None,
    )
    
    return llm, sampling_params

def setup_language_models(args) -> tuple[Any, Optional[Any], Optional[Any]]:
    """
    Setup language models for evaluation.
    
    Args:
        args: Configuration arguments
        
    Returns:
        Tuple of (main_lm, answer_lm, query_lm)
    """
    # Setup main language model
    main_lm = dspy.LM(
        model=args.model.model_name_or_path,
        api_base=f"http://0.0.0.0:{args.search.port[0]}/v1/",
        model_type="chat",
        api_key="EMPTY",
        custom_llm_provider="openai",
        provider="openai",
        cache=False,
    )

    # Setup separate answer model if specified
    answer_lm = None
    if hasattr(args.model, 'answer_model') and args.model.answer_model:
        if len(args.search.port) <= 1:
            raise ValueError("Separate answer model requires at least 2 ports")
        
        print("Setting up separate answer model...")
        answer_lm = dspy.LM(
            model=args.model.answer_model,
            api_base=f"http://0.0.0.0:{args.search.port[1]}/v1/",
            model_type="chat",
            api_key="EMPTY",
            custom_llm_provider="openai",
            provider="openai",
            cache=False,
        )
    else:
        print("Using single model for both query and answer generation")

    return main_lm, answer_lm, None  # query_lm not used in current implementation

def get_prompt(args, query, past_subqueries, past_subanswers, final_reasoning, documents):
    return doc_cot(query, past_subqueries, past_subanswers, final_reasoning, documents)

if __name__ == "__main__":
    # Parse arguments and setup
    args = get_args()
    os.makedirs(args.paths.output_path, exist_ok=True)
    
    if not args.model.answer_only:        
        # Setup language models, globals used everywhere
        main_lm, answer_generator, reasoner = setup_language_models(args)
    
        # Configure DSPy
        dspy.settings.configure(lm=main_lm)
        dspy.configure(lm=main_lm)

    # Run evaluation
    print("Starting evaluation...")
    evaluator = Evaluator(args)
    evaluator.run_loop(args.model.answer_only)
    print("Evaluation completed!")
