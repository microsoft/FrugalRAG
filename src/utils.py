import os
import json
import torch
import threading

from torch import Tensor
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding
from typing import List, Union, Mapping, Dict, Any

from src.logger_config import logger


def move_to_device(sample, device: Union[int, torch.device]):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device, non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_device(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_device(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)(
                {k: _move_to_device(v) for k, v in maybe_tensor.items()}
            )
        else:
            return maybe_tensor

    return _move_to_device(sample)


def create_batch_dict(
    tokenizer: PreTrainedTokenizerFast, input_texts: List[str], max_length: int = 512
) -> BatchEncoding:
    return tokenizer(
        input_texts,
        max_length=max_length,
        padding=True,
        pad_to_multiple_of=8,
        return_token_type_ids=False,
        truncation=True,
        return_tensors="pt",
    )


def get_task_def_by_task_name(task_name: str) -> str:
    task_name_to_instruct: Dict[str, str] = {
        # KILT eval instructions
        "aidayago2": "Given a mention of named entity with surrounding context, retrieve the Wikipedia page that links to the entity",
        "cweb": "Given a mention of named entity with surrounding context, retrieve the Wikipedia page that links to the entity",
        "eli5": "Provided a user question, retrieve Wikipedia passages that can help answer the question",
        "fever": "Given a claim, retrieve documents that support or refute the claim",
        "hotpotqa": "Given a multi-hop question, retrieve documents that can help answer the question",
        "structured_zeroshot": "Given a head entity and a relation, retrieve the Wikipedia page that contains the tail entity",
        "trex": "Given a head entity and a relation, retrieve the Wikipedia page that contains the tail entity",
        "triviaqa": "Given a question, retrieve Wikipedia passages that answer the question",
        "wned": "Given a mention of named entity with surrounding context, retrieve the Wikipedia page that links to the entity",
        "wow": "Given a conversation history, retrieve a Wikipedia passage that contains relevant information to continue the conversation",
        "blink": "Given a mention of named entity with surrounding context, retrieve the Wikipedia page that links to the entity",
    }

    if task_name in task_name_to_instruct:
        return task_name_to_instruct[task_name]

    logger.warning(
        f"Task name {task_name} not found in task_name_to_instruct, will use default instruct"
    )
    return "Given a search query, retrieve relevant documents that can help answer the query"


def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ""

    return "Instruct: {}\nQuery: ".format(task_description)


def pool(last_hidden_states: Tensor, attention_mask: Tensor, pool_type: str) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif (
        pool_type == "weightedavg"
    ):  # position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
        attention_mask *= attention_mask.cumsum(dim=1)  # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
        s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        emb = s / d
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[
                torch.arange(batch_size, device=last_hidden.device), sequence_lengths
            ]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb