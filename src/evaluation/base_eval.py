import os
from tqdm import tqdm
import json
from ..parser_ import get_args
from . import metrics
from datasets import load_dataset
from tqdm import tqdm
import warnings
import dspy
import threading
from src.data.data_utils import format_input_context, load_corpus, load_wiki18_corpus, get_eval_dataset, split_into_sentences
from src.search.search_utils import search_by_http
import requests


warnings.filterwarnings("ignore")


class BaselineEvaluator:
    def __init__(self, args):
        self.args = args
        self.data, self.gold_titles, self.gold_sent = get_eval_dataset(args)
        print("[INFO] eval examples loaded")
        # load model

        lm = dspy.LM(
            deployment_name=self.args.model.model_name_or_path,
            model=self.args.model.model_name_or_path,
            api_base=f"http://0.0.0.0:{args.search.port[0]}/v1/",
            model_type="chat",
            api_key="EMPTY",
            custom_llm_provider="openai",
            provider="openai",
            cache=False,
        )

        dspy.configure(lm=lm)
        print("configured dspy model for inference")

        if self.args.model.baseline in ["rag"]:
            self.model = dspy.Predict("context, question -> answer")

        elif self.args.model.baseline in ["naive"]:
            self.model = dspy.Predict("question -> answer")

        if self.args.model.baseline in ["naive-cot"]:
            self.model = dspy.ChainOfThought("question -> answer")
            print("COT model initialized")

        if self.args.model.baseline in ["rag", "rag-cot"]:
            # search using http
            if args.search.rm in ["wiki5M", "wiki20M"]:
                self.api_url = f"http://127.0.0.1:{args.search.search_port}".rstrip("/")
            elif args.search.rm == "e5-large":
                self.corpus = load_corpus()
            elif args.search.rm == "e5-base":
                self.corpus = load_wiki18_corpus()
            elif args.search.rm == "qwen8b" and args.data.dataset_name == "bcplus":
                self.api_url = f"http://0.0.0.0:{args.search.search_port}".rstrip("/")

            if self.args.model.baseline == "rag-cot":
                self.model = dspy.ChainOfThought("context, question -> answer")
                print("COT model initialized")

    def get_documents(self, search_query):
        """does naive search"""
        try:
            if self.args.search.rm in ["wiki5M", "wiki20M"]:
                response = requests.get(
                    f"{self.api_url}/api/search",
                    params={"query": search_query, "k": self.args.search.ndocs},
                )
                
                response.raise_for_status()
                results = response.json()["topk"]
                passages = [{"long_text": item["text"], "score": item["score"]} for item in results[:self.args.search.ndocs]]
            
            elif self.args.search.rm in ["e5-large", "e5-base"]:
                retriever_results = search_by_http(search_query, port=self.args.search.search_port)
                passages = [{"long_text": format_input_context(self.corpus[int(res["doc_id"])]), "score": res["score"]} for res in retriever_results][:self.args.search.ndocs]
                passages = passages[::-1] if self.args.search.rm == "e5-large" else passages
                
            elif self.args.search.rm == "qwen8b" and self.args.data.dataset_name == "bcplus":
                response = requests.get(
                        f"{self.api_url}/api/search",
                        params={"query": search_query, "k": self.args.search.ndocs},
                )
                response.raise_for_status()
                results = response.json()
                passages = [{"long_text": res["snippet"], "score": res["score"], "doc_id": int(res["docid"])} for res in results]
            
            return passages
            
        except Exception as e:
            print(f"[RemoteSearchModule API Error] {e}")
            passages = []

    def run_model(self, question):
        if self.args.model.baseline in ["rag-cot", "rag"]:
            passages = self.get_documents(question)
            context = "\n".join([p["long_text"] for p in passages]) + "\n" + question
                        
            output = self.model(context=context, question=question)
            return {"passages": passages, "answer": output.answer}

        elif self.args.model.baseline in ["naive-cot", "naive"]:
            output = self.model(question=question)
            return {"answer": output.answer}

    def run_model_with_timeout(
        self, question, lm_answer=None, lm_query=None, timeout=1000
    ):
        """Run self.model(question=…) in a daemon thread, time out after `timeout` seconds."""
        result = {}

        def target():
            result["out"] = self.run_model(question=question)

        # leave result['out'] unset so we return None
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            print(f"Timeout after {timeout}s on question: {question!r}")
            return None
        if "err" in result:
            raise result["err"]

        return result.get("out", None)

    def run_loop(self):
        metric_dict = {
            "match": [],
            "em": [],
            "f1": [],
            "recall": [],
            "recall_flashrag": [],
            "precision_flashrag": [],
            "support_f1": [],
            "num_hops": [],
            "num_searches": [],
        }
        logs = []

        if os.path.exists(os.path.join(self.args.paths.output_path, "logs.json")):
            with open(os.path.join(self.args.paths.output_path, "logs.json")) as f:
                curr_logs = json.load(f)

            for l in tqdm(curr_logs):
                if len(l["pred_titles"]) > 0:
                    logs.append(l)

            solved_questions = [l["question"] for l in logs]
            for log in logs:
                for k, v in log["metrics"].items():
                    metric_dict[k].append(v)
        else:
            solved_questions = []

        print(f"already solved {len(solved_questions)}")

        pairs = [
            (
                self.process_example,
                {
                    "question": example["question"],
                    "_id": example["_id"],
                    "answer": example["answer"],
                },
            )
            for idx, example in enumerate(self.data)
            if example["question"] not in solved_questions
        ]

        # use multiprocessing instead of dspy.parallel for baselines
        if len(pairs) == 0:
            print("already done, skipping")
        else:
            executor = dspy.parallel.Parallel(
                num_threads=min(16, len(pairs)), max_errors=10, provide_traceback=True
            )
            results = executor.forward(pairs)

            for idx, log_dict in enumerate(results):
                if log_dict is not None:
                    for k, v in log_dict["metrics"].items():
                        metric_dict[k].append(v)
                    
                    logs.append(log_dict)

            with open(
                os.path.join(self.args.paths.output_path, "avg_metrics.json"), "w"
            ) as f:
                avg_metrics = { k: sum(v) / len(v) for k, v in metric_dict.items()}
                json.dump(avg_metrics, f, indent=1)

            with open(os.path.join(self.args.paths.output_path, "logs.json"), "w") as f:
                json.dump(logs, f, indent=1)
    
    def _calculate_support_f1(self, pred_passages, gold_sent) -> float:
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

    def process_example(self, question, answer, _id):
        gold_titles, gold_sent = self.gold_titles[_id], self.gold_sent[_id]

        output = self.run_model_with_timeout(question)

        if output is None:
            return None

        if args.model.baseline == "naive":
            pred_titles, search_queries, num_hops, trajectory, rationale = (
                [],
                [],
                0,
                {},
                "",
            )
            pred_answer = output["answer"]
            pred_passages = []

        elif args.model.baseline == "naive-cot":
            pred_titles, search_queries, num_hops, trajectory, rationale = (
                [],
                [],
                0,
                {},
                "",
            )
            pred_answer = output.get("answer", "")
            pred_passages = []

        elif args.model.baseline in ["rag", "rag-cot"]:
            pred_answer = output["answer"]
            passages = output["passages"]
            pred_titles = [
                passage["long_text"].split("|")[0].strip() for passage in passages
            ]
            pred_passages = [passage["long_text"] for passage in passages]
            search_queries = [question]
            num_hops = 1
            trajectory, rationale = {}, ""

        log_dict = {}
        log_dict["question"] = question
        log_dict["answer"] = answer
        log_dict["pred"] = pred_answer
        log_dict["gold_titles"] = gold_titles
        log_dict["pred_titles"] = pred_titles
        log_dict["search_queries"] = search_queries
        log_dict["num_searches"] = len(search_queries)
        log_dict["num_hops"] = num_hops
        log_dict["trajectory"] = trajectory
        log_dict["rationale"] = rationale

        if self.args.data.dataset_name in ["2wiki", "musique", "hotpot"]:
            support_f1 = self._calculate_support_f1(pred_passages, gold_sent)
            recall = metrics.title_recall(pred_titles, gold_titles)
        elif self.args.data.dataset_name in ["bcplus"]:
            support_f1 = -1
            recall = metrics.recall_ret(pred_titles, gold_titles)
            
        # to match with aliases
        answer = [answer] if isinstance(answer, str) else answer

        log_dict["metrics"] = {
            "f1": max([metrics.f1(pred_answer, a) for a in answer]),
            "em": max([metrics.em(pred_answer, a) for a in answer]),
            "match": max([metrics.match(pred_answer, a) for a in answer]),
            "recall": recall,
            "recall_flashrag": metrics.recall_flashrag(pred_passages, answer),
            "precision_flashrag": metrics.precision_flashrag(pred_passages, answer),
            "support_f1": support_f1,
            "num_searches": len(search_queries),
            "num_hops": num_hops,
        }

        return log_dict


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.paths.output_path, exist_ok=True)

    eval_ = BaselineEvaluator(args)
    eval_.run_loop()
