import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import json
from ..parser_ import get_args
from . import metrics
from datasets import load_dataset
import time
from tqdm import tqdm
import warnings
from ..model import Hopper
import traceback
import dspy
import threading


warnings.filterwarnings("ignore")


class BaselineEvaluator:
    def __init__(self, args):
        self.args = args
        self.data, self.gold_titles, self.gold_sent = self.load_files()
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

        if self.args.baseline in ["rag"]:
            self.model = dspy.Predict("context, question -> answer")

        elif self.args.baseline in ["naive"]:
            self.model = dspy.Predict("question -> answer")

        if self.args.baseline in ["naive-cot"]:
            self.model = dspy.ChainOfThought("question -> answer")
            print("COT model initialized")

        if self.args.baseline in ["rag", "rag-cot"]:
            from colbert import Searcher

            self.retrieve = Searcher(
                index=args.index,
                index_root=args.index_root,
                collection=args.collection_path,
                checkpoint="/data/leret/colbertv2.0",
            )

            if self.args.baseline == "rag-cot":
                self.model = dspy.ChainOfThought("context, question -> answer")
                print("COT model initialized")

    def get_documents(self, search_query):
        doc_ids, _, scores = self.retrieve.search(search_query, self.args.ndocs)
        documents = [self.retrieve.collection.data[id_] for id_ in doc_ids]
        passages = [
            {"long_text": doc, "score": score} for doc, score in zip(documents, scores)
        ]

        return passages

    def load_files(self):
        if self.args.data.dataset_name in ["hotpot", "hopo"]:
            with open(
                f"{self.args.root}/data/eval-v2/hopo/hotpot_dev_distractor_v1.json"
            ) as f:
                dataset = json.load(f)

            gold_titles, gold_sent = {}, {}
            for example in dataset:
                supporting_titles = [t[0] for t in example["supporting_facts"]]
                supporting_sent = [s[1] for s in example["supporting_facts"]]
                curr_gold_sent, curr_gold_titles = [], []

                for context in example["context"]:
                    title = context[0]
                    if title in supporting_titles:
                        sent_idx = supporting_sent[supporting_titles.index(title)]
                        curr_gold_sent.append(context[1][sent_idx])
                        curr_gold_titles.append(title)

                gold_sent[example["_id"]], gold_titles[example["_id"]] = (
                    curr_gold_sent,
                    curr_gold_titles,
                )

        elif self.args.data.dataset_name in ["musique", "dgslibisey/MuSiQue"]:
            from datasets import load_dataset
            import random

            random.seed(663)
            orig_dataset = load_dataset("dgslibisey/MuSiQue")
            orig_dataset = orig_dataset["validation"]
            orig_dataset = [d for d in orig_dataset if d["answerable"]]  # extra check

            # compute the evidence F1 with the answerable GT, no need to create mapping
            dataset, gold_titles, gold_sent = [], {}, {}

            for example in orig_dataset:
                curr_gold_sent, curr_gold_titles = [], []
                for paragraph in example["paragraphs"]:
                    if paragraph["is_supporting"]:
                        curr_gold_titles.append(paragraph["title"])
                        curr_gold_sent.append(paragraph["paragraph_text"])

                gold_sent[example["id"]], gold_titles[example["id"]] = (
                    curr_gold_sent,
                    curr_gold_titles,
                )
                dataset.append(
                    {
                        "_id": example["id"],
                        "question": example["question"],
                        "answer": example["answer_aliases"] + [example["answer"]],
                    }
                )

            # # for quick test
            # dataset = random.sample(dataset, 500)

        elif self.args.data.dataset_name in ["2wiki"]:
            import random

            random.seed(663)
            with open(f"{self.args.root}/2wikimhqa/data/dev.json") as f:
                orig_dataset = json.load(f)

            # compute the evidence F1 with the answerable GT, no need to create mapping
            dataset, gold_titles, gold_sent = [], {}, {}

            for example in orig_dataset:
                curr_gold_titles, curr_gold_sent = [], []

                supporting_titles = [t[0] for t in example["supporting_facts"]]
                supporting_sent = [s[1] for s in example["supporting_facts"]]

                for context in example["context"]:
                    title = context[0]
                    if title in supporting_titles:
                        sent_idx = supporting_sent[supporting_titles.index(title)]
                        curr_gold_sent.append(context[1][sent_idx])
                        curr_gold_titles.append(title)

                gold_sent[example["_id"]], gold_titles[example["_id"]] = (
                    curr_gold_sent,
                    curr_gold_titles,
                )
                dataset.append(
                    {
                        "_id": example["_id"],
                        "question": example["question"],
                        "answer": example["answer"],
                    }
                )

            # # for quick test
            # dataset = random.sample(dataset, 500)

        return dataset, gold_titles, gold_sent

    def run_model(self, question):
        if self.args.baseline in ["rag-cot", "rag"]:
            passages = self.get_documents(question)
            context = "\n".join([p["long_text"] for p in passages]) + "\n" + question
            output = self.model(context=context, question=question)
            return {"passages": passages, "answer": output.answer}

        elif self.args.baseline in ["naive-cot", "naive"]:
            output = self.model(question=question)
            return {"answer": output.answer}

    def run_model_with_timeout(
        self, question, lm_answer=None, lm_query=None, timeout=240
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
                metric_dict["match"].append(log["metrics"]["match"])
                metric_dict["em"].append(log["metrics"]["em"])
                metric_dict["f1"].append(log["metrics"]["f1"])
                metric_dict["recall"].append(log["metrics"]["recall"])
                metric_dict["support_f1"].append(log["metrics"]["support_f1"])
                metric_dict["num_hops"].append(log["metrics"]["num_hops"])
                metric_dict["num_searches"].append(log["metrics"]["num_searches"])
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
                    metric_dict["match"].append(log_dict["metrics"]["match"])
                    metric_dict["em"].append(log_dict["metrics"]["em"])
                    metric_dict["f1"].append(log_dict["metrics"]["f1"])
                    metric_dict["recall"].append(log_dict["metrics"]["recall"])
                    metric_dict["support_f1"].append(log_dict["metrics"]["support_f1"])
                    metric_dict["num_hops"].append(log_dict["metrics"]["num_hops"])
                    metric_dict["num_searches"].append(
                        log_dict["metrics"]["num_searches"]
                    )

                    logs.append(log_dict)

            with open(
                os.path.join(self.args.paths.output_path, "avg_metrics.json"), "w"
            ) as f:
                avg_metrics = {
                    "f1": sum(metric_dict["f1"]) / len(metric_dict["f1"]),
                    "match": sum(metric_dict["match"]) / len(metric_dict["match"]),
                    "em": sum(metric_dict["em"]) / len(metric_dict["em"]),
                    "recall": sum(metric_dict["recall"]) / len(metric_dict["recall"]),
                    "support_f1": sum(metric_dict["support_f1"])
                    / len(metric_dict["support_f1"]),
                    "num_hops": sum(metric_dict["num_hops"])
                    / len(metric_dict["num_hops"]),
                    "num_searches": sum(metric_dict["num_searches"])
                    / len(metric_dict["num_searches"]),
                }
                json.dump(avg_metrics, f, indent=1)

            with open(os.path.join(self.args.paths.output_path, "logs.json"), "w") as f:
                json.dump(logs, f, indent=1)

    def process_example(self, question, answer, _id):
        gold_titles, gold_sent = self.gold_titles[_id], self.gold_sent[_id]

        output = self.run_model_with_timeout(question)

        if output is None:
            return None

        if args.baseline == "naive":
            pred_titles, search_queries, num_hops, trajectory, rationale = (
                [],
                [],
                0,
                {},
                "",
            )
            pred_answer = output["answer"]
            pred_passages = []

        elif args.baseline == "naive-cot":
            pred_titles, search_queries, num_hops, trajectory, rationale = (
                [],
                [],
                0,
                {},
                "",
            )
            pred_answer = output.get("answer", "")
            pred_passages = []

        elif args.baseline in ["rag", "rag-cot"]:
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

        support_f1 = []
        for sent in gold_sent:
            f1s = []
            for pred_sent in pred_passages:
                for ps in pred_sent.split("."):
                    f1s.append(metrics.f1(ps, sent))

            support_f1.append(max(f1s) if len(f1s) > 0 else 0)

        # to match with aliases
        answer = [answer] if isinstance(answer, str) else answer

        log_dict["metrics"] = {
            "f1": max([metrics.f1(pred_answer, a) for a in answer]),
            "em": max([metrics.em(pred_answer, a) for a in answer]),
            "match": max([metrics.match(pred_answer, a) for a in answer]),
            "recall": metrics.title_recall(pred_titles, gold_titles),
            "support_f1": sum(support_f1) / len(support_f1),
            "num_searches": len(search_queries),
            "num_hops": num_hops,
        }

        return log_dict


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)

    eval_ = BaselineEvaluator(args)
    eval_.run_loop()
