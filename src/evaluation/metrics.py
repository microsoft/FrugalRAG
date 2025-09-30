import logging
import string
from collections import Counter
from typing import Callable

import numpy as np
import regex

logger = logging.getLogger(__name__)


# Normalization and score functions from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em(prediction, ground_truth, normalize_fn=normalize_answer):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))


def match(prediction, ground_truth, normalize_fn=normalize_answer):
    return float(normalize_fn(ground_truth) in normalize_fn(prediction))


def f1(prediction, ground_truth, normalize_fn=normalize_answer):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def recall_ret(pred, ground_truth):
    common = Counter(pred) & Counter(ground_truth)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    recall = 1.0 * num_same / len(ground_truth)
    return recall


# leret style
def title_recall(pred, ground_truth):
    recall = 0
    pred = [normalize_answer(p) for p in pred]
    for gt in ground_truth:
        if normalize_answer(gt) in pred:
            recall += 1

    return 1.0 * recall / len(ground_truth)

def recall_flashrag(passages, golden_answers):
    """minimal reproduction of https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/evaluator/metrics.py"""
    hit_list = []
    for passage in passages:
        for gold_answer in golden_answers:
            if normalize_answer(gold_answer) in normalize_answer(passage):
                hit_list.append(True)
                break
            else:
                hit_list.append(False)
    
    score = 1 if any(hit_list) else 0
    
    return score

def precision_flashrag(passages, golden_answers):
    """minimal reproduction of https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/evaluator/metrics.py"""
    hit_list = []
    for passage in passages:
        for gold_answer in golden_answers:
            if normalize_answer(gold_answer) in normalize_answer(passage):
                hit_list.append(True)
                break
            else:
                hit_list.append(False)
    
    score = sum(hit_list) / len(hit_list) if hit_list else 0
    
    return score