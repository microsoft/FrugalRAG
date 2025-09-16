import random
random.seed(633)

import json
import os
import dspy
from typing import List, Dict, Optional
from datasets import load_dataset, Dataset
from openai.types.chat import ChatCompletion
from contextlib import nullcontext
from transformers import PreTrainedTokenizerFast
from threading import Lock
from dspy.datasets import HotPotQA

from src.logger_config import logger


def load_corpus() -> Dataset:
    corpus: Dataset = load_dataset("corag/kilt-corpus", split="train")
    logger.info(f"Loaded {len(corpus)} passages from corag/kilt-corpus")
    return corpus

def format_input_context(doc: Dict[str, str]) -> str:
    title: str = doc.get("title", "")
    contents: str = doc["contents"]
    if contents.startswith(title + "\n"):
        contents = contents[len(title) + 1 :]

    return f"{title}\n{contents}".strip()


def get_prompt_optimization_dataset(args):
    if args.data.input_file.endswith(".json"):
        with open(args.data.input_file) as f:
            orig_data = json.load(f)
            sampled_questions = [d["question"] for d in orig_data]

    if args.data.dataset_name == "hotpot":
        all_examples = [
            x.with_inputs("question")
            for x in HotPotQA(train_seed=2024, train_size=args.data.num_prompt_examples*2).train if x.question not in sampled_questions
        ]
        trainset = random.sample(list(all_examples), args.data.num_prompt_examples)
        valset = [example for example in all_examples if example not in trainset]
            
    elif args.data.dataset_name in ["musique", "dgslibisey/MuSiQue"]:
        from datasets import load_dataset

        input_data = load_dataset("dgslibisey/MuSiQue")
        input_data = [d for d in input_data["train"] if d["question"] not in sampled_questions]

        train_idx = random.sample(list(range(args.data.num_prompt_examples*2)), args.data.num_prompt_examples)
        val_idx = [idx for idx in range(args.data.num_prompt_examples*2) if idx not in train_idx]

        all_examples = [
            dspy.Example(
                question=input_data[i]["question"], answer=input_data[i]["answer"]
            ).with_inputs("question")
            for i in range(args.data.num_prompt_examples*2)
        ]

        trainset = [item for idx, item in enumerate(all_examples) if idx in train_idx]
        valset = [item for idx, item in enumerate(all_examples) if idx in val_idx]

        with open(os.path.join(args.paths.prompt_path, "train_questions.json"), "w") as f:
            json.dump([d.question for d in all_examples], f, indent=2)

    elif args.data.dataset_name in ["2wiki"]:
        from datasets import load_dataset
        input_data = load_dataset("framolfese/2WikiMultihopQA")["train"]
        input_data = [d for d in input_data if d["question"] not in sampled_questions]

        train_idx = random.sample(list(range(args.data.num_prompt_examples*2)), args.data.num_prompt_examples)
        val_idx = [idx for idx in range(args.data.num_prompt_examples*2) if idx not in train_idx]

        all_examples = [
            dspy.Example(
                question=input_data[i]["question"], answer=input_data[i]["answer"]
            ).with_inputs("question")
            for i in range(args.data.num_prompt_examples*2)
        ]

        trainset = [item for idx, item in enumerate(all_examples) if idx in train_idx]
        valset = [item for idx, item in enumerate(all_examples) if idx in val_idx]

        with open(os.path.join(args.paths.prompt_path, "train_questions.json"), "w") as f:
            json.dump([d.question for d in all_examples], f, indent=2)
            
    elif args.data.dataset_name == "bcplus":
        with open(args.data.input_file) as f:
            input_data = [json.loads(i) for i in f.readlines()]
        
        # sample 50 for prompt optimization, 50 for validation
        train_idx = random.sample(list(range(args.data.num_prompt_examples*2)), args.data.num_prompt_examples)
        val_idx = [idx for idx in range(args.data.num_prompt_examples*2) if idx not in train_idx]
        
        all_examples = [
            dspy.Example(
                question=input_data[i]["query"], answer=input_data[i]["answer"]
            ).with_inputs("question")
            for i in range(args.data.num_prompt_examples*2)
        ]
        
        trainset = [item for idx, item in enumerate(all_examples) if idx in train_idx]
        valset = [item for idx, item in enumerate(all_examples) if idx in val_idx]

        with open(os.path.join(args.paths.prompt_path, "train_questions.json"), "w") as f:
            json.dump([d.question for d in all_examples], f, indent=2)
    
    elif args.data.dataset_name == "researchy":
        from datasets import load_dataset
        input_data = load_dataset("corbyrosset/researchy_questions")

        train_idx = random.sample(list(range(args.data.num_prompt_examples*2)), args.data.num_prompt_examples)
        val_idx = [idx for idx in range(args.data.num_prompt_examples*2) if idx not in train_idx]
        
        all_examples = [
            dspy.Example(
                question=input_data[i]["question"], answer=[j['Url '] for j in input_data[i]["DocStream"]]
            ).with_inputs("question")
            for i in range(args.data.num_prompt_examples*2)
        ]
        
        trainset = [item for idx, item in enumerate(all_examples) if idx in train_idx]
        valset = [item for idx, item in enumerate(all_examples) if idx in val_idx]

        with open(os.path.join(args.paths.prompt_path, "train_questions.json"), "w") as f:
            json.dump([d.question for d in all_examples], f, indent=2)

    else:
        raise (NotImplementedError)

    return trainset, valset


def get_original_dataset(args):
    if args.data.input_file.endswith(".json"):
        with open(args.data.input_file) as f:
            orig_dataset = json.load(f)
    elif args.data.input_file.endswith(".jsonl"):
        with open(args.data.input_file) as f:
            raw = f.readlines()
            orig_dataset = [json.loads(d) for d in raw]
        
    dataset, gold_titles, gold_sent = [], {}, {}
        
    if args.data.dataset_name in ["hotpot", "hopo"]:
        for example in orig_dataset:
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
            
            dataset.append(example)
        
        return dataset, gold_titles, gold_sent 

    elif args.data.dataset_name in ["musique", "dgslibisey/MuSiQue"]:
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
        
        return dataset, gold_titles, gold_sent 

    elif args.data.dataset_name in ["2wiki"]:
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
        
        return dataset, gold_titles, gold_sent

    elif args.data.dataset_name == "bcplus":
        for example in orig_dataset:
            # evidence_docs = [d["text"] for d in example["evidence_docs"]]
            evidence_docids = [int(d["docid"]) for d in example["evidence_docs"]]
            # gold_docs = [d["text"] for d in example["gold_docs"]]
            gold_docids = [int(d["docid"]) for d in example["gold_docs"]]
            
            gold_titles[example["query_id"]] = evidence_docids
            gold_sent[example["query_id"]] = gold_docids
            
            dataset.append(
                {
                    "_id": example["query_id"],
                    "question": example["query"],
                    "answer": example["answer"]
                }
            )
            
        return dataset, gold_titles, gold_sent  
    
    else:
        raise(NotImplementedError)
    

def get_eval_dataset(args):
    dataset, gold_titles, gold_sent = [], {}, {}
    
    if os.path.exists(args.data.input_file):
        print(f"[INFO] Loading data from {args.data.input_file}")
        if args.data.input_file.endswith(".json"):
            with open(args.data.input_file) as f:
                raw_dataset = json.load(f)
        elif args.data.input_file.endswith(".jsonl"):
            with open(args.data.input_file) as f:
                raw_dataset = [json.loads(i) for i in f.readlines()]
    else:
        try:
            from datasets import load_dataset
            print("[INFO] Loading validation data from HF")
            raw_dataset = load_dataset(args.data.input_file)["validation"]
        except:
            print("[ERROR] The input_file argument does not contain a valid path to a json file, neither is it a valid HF datasets pointer. Please check the arguments carefully!")
            raise(NotImplementedError)


    if args.data.dataset_name in ["hotpot", "hopo"]:
        for example in raw_dataset:
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
            
            dataset.append(example)
        
        return dataset, gold_titles, gold_sent

    elif args.data.dataset_name in ["musique", "dgslibisey/MuSiQue"]:
        for example in raw_dataset:
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
        
        return dataset, gold_titles, gold_sent

    elif args.data.dataset_name in ["2wiki"]:
        for example in raw_dataset:
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

        return dataset, gold_titles, gold_sent

    elif args.data.dataset_name == "bcplus":
        for example in raw_dataset:
            # evidence_docs = [d["text"] for d in example["evidence_docs"]]
            evidence_docids = [int(d["docid"]) for d in example["evidence_docs"]]
            # gold_docs = [d["text"] for d in example["gold_docs"]]
            gold_docids = [int(d["docid"]) for d in example["gold_docs"]]
            
            gold_titles[example["query_id"]] = evidence_docids
            gold_sent[example["query_id"]] = gold_docids
            
            
            dataset.append(
                {
                    "_id": example["query_id"],
                    "question": example["query"],
                    "answer": example["answer"]
                }
            )
        
        # here gold titles, sents are evidence documents and gold documents
        return dataset, gold_titles, gold_sent
                
    else:
        print("Not Supported")
        raise(NotImplementedError)

# found a cool sentence breaker on stack overflow, useful for 2wiki train/eval
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mt|Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences