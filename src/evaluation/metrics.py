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


def rouge_wrapper(prediction, ground_truth):
    from rouge import Rouge

    rouge = Rouge()
    try:
        result = rouge.get_scores(prediction, ground_truth, avg=True)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except:
        return 0.0, 0.0, 0.0


def rouge_score(prediction, ground_truths):
    ground_truths = [x for x in ground_truths if len(x) > 0]
    if (
        len(prediction) == 0 or len(ground_truths) == 0
    ):  # check if empty prediction or if there is no hypothesis with len > 0
        return 0.0, 0.0, 0.0
    scores = [rouge_wrapper(prediction, gt) for gt in ground_truths]
    rouge1 = max(s[0] for s in scores)
    rouge2 = max(s[1] for s in scores)
    rougel = max(s[2] for s in scores)
    return rouge1, rouge2, rougel
