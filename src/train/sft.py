import os
import json
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import random
from typing import Any, Callable, Literal, get_origin, get_type_hints
import dspy
from datasets import Dataset
import torch
import pandas as pd
from argparse import ArgumentParser
from dspy.adapters.chat_adapter import format_turn
from dspy.predict.react import Tool
from ..model import CustomReAct
from ..evaluation import metrics
from ..parser_ import get_args

# Disable multi-node distributed mode
os.environ.pop("MASTER_ADDR", None)
os.environ.pop("MASTER_PORT", None)
os.environ.pop("WORLD_SIZE", None)
os.environ.pop("RANK", None)
os.environ.pop("LOCAL_RANK", None)


def format_trajectory(trajectory: dict[str, Any]):
    adapter = dspy.settings.adapter or dspy.ChatAdapter()
    trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
    return adapter.format_fields(trajectory_signature, trajectory, role="user")


def dummy_func():
    pass

class CustomTrainer:
    def __init__(self, args):
        self.args = args

        tool_desc = (
            "Searches documents using a search query.\n"
            "Arguments:\n"
            '- "search_query": a string search_query to search.\n'
            'IMPORTANT: YOU MUST always PROVIDE "search_query" in the arguments!'
        )

        query_generator = Tool(func=dummy_func, name="AdvancedSearch", desc=tool_desc)
        self.react_model = CustomReAct(
            "question -> answer", tools=[query_generator], args=args
        )
        self.adapter = dspy.settings.adapter or dspy.ChatAdapter()

        if args.paths.prompt_path != "":
            self.react_model.load(args.paths.prompt_path)
            
        self.model, self.tokenizer = self.load_full_model()

        os.makedirs(args.paths.output_path, exist_ok=True)
        # with open(os.path.join(args.paths.output_path, "args.json"), "w") as f:
        #     json.dump(vars(args), f, indent=1)

    def preprocess_example(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def load_full_model(self, ckpt=""):
        local_rank = os.getenv("LOCAL_RANK", None)
        if local_rank is None:
            device_string = "auto"
        else:
            device_string = "cuda:" + str(local_rank)

        if ckpt != "":
            model = AutoModelForCausalLM.from_pretrained(
                ckpt, device_map=device_string, torch_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(ckpt)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model.model_name_or_path,
                device_map=device_string,
                torch_dtype=torch.bfloat16,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.args.model.model_name_or_path)

        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
            model.config.pad_token_id = tokenizer.pad_token_id  # updating model config

        tokenizer.padding_side = "left"

        return model, tokenizer

    def apply_template(self, example, add_generation_prompt=False):
        question, context = example["question"], example["input"]
        react_input = self.adapter.format(
            signature=self.react_model.react.signature,
            demos=self.react_model.react.demos,
            inputs={"question": question, "trajectory": format_trajectory(context)},
        )
        best_completion = example["output"]
        react_dict = {}
        for k, v in best_completion.items():
            if (
                k.startswith("observation")
                and not isinstance(v, dict)
                and v.startswith("Failed")
            ):
                print(v, "skipped")
                react_dict = {}
                break

            if k.startswith("thought"):
                react_dict["next_thought"] = v
            elif k.startswith("tool_args"):
                react_dict["next_tool_args"] = v
            elif k.startswith("tool_name"):
                react_dict["next_tool_name"] = v

        if react_dict != {}:
            react_output = format_turn(
                values=react_dict,
                role="assistant",
                signature=self.react_model.react.signature,
            )
            react_input.append(react_output)
        else:
            react_input = None

        return react_input

    def get_dataset(self, do_sft=True):
        train_filepath = self.args.data.input_file
        assert os.path.exists(train_filepath) == True

        print(train_filepath)

        with open(train_filepath) as f:
            train_data = json.load(f)

        sft_train_data = []
        for example in train_data:
            react_messages = self.apply_template(example)
            if react_messages is not None:
                sft_train_data.append({"messages": react_messages})

        sft_eval_data = []
        print(len(sft_train_data))
        return sft_train_data, sft_eval_data

    def train(self):
        train_dataset, eval_dataset = self.get_dataset()

        # convert to hf format
        train_dataset = Dataset.from_list([item for item in train_dataset])

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            args=SFTConfig(
                per_device_train_batch_size=args.sft.batch_size,
                gradient_accumulation_steps=args.sft.grad_acc,
                warmup_steps=self.args.sft.warmup,
                num_train_epochs=self.args.sft.epochs,
                # max_steps=10, # debug
                learning_rate=self.args.sft.lr,
                fp16=False,
                bf16=True,
                logging_steps=1,
                optim="adamw_torch",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=663,
                output_dir=self.args.paths.output_path,
                dataloader_num_workers=8,
                dataloader_pin_memory=True,
                save_strategy="no",
                max_seq_length=self.args.sft.max_seq_len,
                dataset_num_proc=2,
                report_to="none",
            ),
        )
        trainer.train()

        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        if local_rank == 0:
            # Save the model and tokenizer
            self.model.save_pretrained(
                os.path.join(
                    self.args.paths.output_path,
                    f"{self.args.model.model_name_or_path.replace('/', '_')}",
                )
            )
            self.tokenizer.save_pretrained(
                os.path.join(
                    self.args.paths.output_path,
                    f"{self.args.model.model_name_or_path.replace('/', '_')}",
                )
            )
            df = pd.DataFrame(trainer.state.log_history)
            df.to_csv(
                os.path.join(
                    self.args.paths.output_path,
                    f"{self.args.model.model_name_or_path.replace('/', '_')}.csv",
                ),
                sep="\t",
            )


if __name__ == "__main__":
    args = get_args("sft")
    multihop_trainer = CustomTrainer(args)
    multihop_trainer.train()
