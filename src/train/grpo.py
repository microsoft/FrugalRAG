import json
from trl import GRPOTrainer
from src.evaluation import metrics
import torch
from typing import (
    Any,
    Callable,
    Literal,
    get_origin,
    get_type_hints,
    Union,
    Optional,
    TYPE_CHECKING,
    Dict,
    List,
    Tuple,
    Type,
)
import dspy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
import torch.nn as nn
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from dspy.primitives.prediction import Prediction
import re

# from trl.extras.profiling import profiling_context, profiling_decorator
from trl.trainer.grpo_trainer import *
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from transformers.trainer import *
import random


def split_tensor_list(
    all_tensor_dict: list[dict[str, Optional[torch.Tensor]]], num_chunks: int
) -> list[list[dict[str, Optional[torch.Tensor]]]]:
    outputs = []
    for tensor_dict in all_tensor_dict:

        first_tensor = next(
            tensor for tensor in tensor_dict.values() if tensor is not None
        )
        chunk_size = first_tensor.shape[0] // num_chunks
        outputs.append(
            [
                {
                    key: (
                        tensor[i * chunk_size : (i + 1) * chunk_size]
                        if tensor is not None
                        else None
                    )
                    for key, tensor in tensor_dict.items()
                }
                for i in range(num_chunks)
            ]
        )

    return outputs


def format_trajectory(trajectory: dict[str, Any]):
    adapter = dspy.settings.adapter or dspy.ChatAdapter()
    trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
    return adapter.format_fields(trajectory_signature, trajectory, role="user")


def get_passages(observation: dict) -> list[str]:
    if isinstance(observation, dict):
        passages = observation["passages_this_hop"]
    elif observation == "Completed.":
        return 0
    else:
        return None

    output = [
        passage["long_text"] if isinstance(passage, dict) else passage
        for passage in passages
    ]
    return output


class GRPOReActTrainer(GRPOTrainer):
    def __init__(self, **kwargs):
        self.custom_args = kwargs.pop("custom_args")
        self.react_model = kwargs.pop("react_model")
        self.adapter = dspy.settings.adapter or dspy.ChatAdapter()

        super().__init__(**kwargs)
        self.max_iters = self.custom_args.model.max_iters

    # with trl>=1.70.0dev0
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"

        trajectories = [{} for _ in range(len(inputs))]
        finish_locations = {}
        all_completion_ids = []
        all_completion_masks = []
        all_completion_texts = []
        all_prompt_ids, all_prompt_mask, all_prompt_text = [], [], []
        all_ref_per_token_logps, all_old_per_token_logps = [], []
        questions = [x["prompt"] for x in inputs]

        for idx in range(len(inputs)):
            trajectory_id = f"traj_{idx}"
            # Reset for each trajectory
            self.react_model.tools["AdvancedSearch"].func.reset_retrieved_docs(trajectory_id)
            
            init_trajectory = {}
            # build trajectory
            init_trajectory["tool_name_init"] = "AdvancedSearch"
            init_trajectory["tool_args_init"] = {"search_query": questions[idx]}
            init_trajectory["thought_init"] = (
                "I will start by searching for relevant documents with the original question as the search query."
            )

            # Pass trajectory_id to ensure unique docs per trajectory
            init_trajectory["observation_init"] = self.react_model.tools["AdvancedSearch"](questions[idx], trajectory_id)
            
            for k, v in init_trajectory.items():
                trajectories[idx][k] = v

        for hop in range(self.max_iters):
            react_inputs = [
                {
                    "prompt": self.adapter.format(
                        signature=self.react_model.react.signature,
                        demos=self.react_model.react.demos,
                        inputs={
                            "question": inputs[idx]["prompt"],
                            "trajectory": format_trajectory(trajectories[idx]),
                        },
                    )
                }
                for idx in range(len(inputs))
            ]
            prompts = [x["prompt"] for x in react_inputs]
            prompts_text = [
                maybe_apply_chat_template(example, self.processing_class)["prompt"]
                for example in react_inputs
            ]

            prompt_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )

            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            prompt_ids, prompt_mask = (
                prompt_inputs["input_ids"],
                prompt_inputs["attention_mask"],
            )

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length :]
                prompt_mask = prompt_mask[:, -self.max_prompt_length :]

            all_prompt_ids.append(prompt_ids)
            all_prompt_mask.append(prompt_mask)
            all_prompt_text.append(prompts_text)
            # Generate completions using either vLLM or regular generation
            if self.args.use_vllm:  # and self.vllm_mode == "server":
                # First, have main process load weights if needed
                if self.state.global_step != self._last_loaded_step:
                    self._move_model_to_vllm()
                    self._last_loaded_step = self.state.global_step

                # Generate completions using vLLM
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)

                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]
                completion_ids = [
                    torch.tensor(ids, device=device) for ids in completion_ids
                ]
                completion_ids = pad(
                    completion_ids, padding_value=self.processing_class.pad_token_id
                )
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

            else:
                # Regular generation path
                with unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model:
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids,
                        attention_mask=prompt_mask,
                        generation_config=self.generation_config,
                    )

                prompt_length = prompt_ids.size(1)
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full(
                (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
            )
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
                is_eos.size(0), -1
            )
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            if finish_locations:
                # mask the entire sequence if the model predicted "finish" previously
                should_mask = torch.zeros(
                    completion_mask.size(0), dtype=torch.bool, device=device
                )
                for idx, finish_loc in finish_locations.items():
                    if finish_loc < hop:
                        should_mask[idx] = True

                # Whenever you index a 2-D tensor with a 1-D boolean mask on the first axis, PyTorch zeros out entire rows
                completion_mask[should_mask] = 0

            # Append completion_ids and completion_mask for this hop
            all_completion_ids.append(completion_ids)
            all_completion_masks.append(completion_mask)

            # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
            if self.mask_truncated_completions:
                truncated_completions = ~is_eos.any(dim=1)
                completion_mask = (
                    completion_mask * (~truncated_completions).unsqueeze(1).int()
                )

            # Concatenate prompt_mask with completion_mask for logit computation
            attention_mask = torch.cat(
                [prompt_mask, completion_mask], dim=1
            )  # (B, P+C)

            logits_to_keep = completion_ids.size(
                1
            )  # we only need to compute the logits for the completion tokens
            batch_size = (
                self.args.per_device_train_batch_size
                if mode == "train"
                else self.args.per_device_eval_batch_size
            )

            with torch.no_grad():
                # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
                # computation here, and use per_token_logps.detach() instead.
                if self.num_iterations > 1:
                    old_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                    )

                else:
                    old_per_token_logps = None

                if self.beta == 0.0:
                    ref_per_token_logps = None
                elif self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size,
                        )

            all_ref_per_token_logps.append(ref_per_token_logps)
            all_old_per_token_logps.append(old_per_token_logps)
            # Decode the generated completions
            completions_text = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )

            all_completion_texts.append(completions_text)

            # trajectories updated for next hop
            for idx, curr_completion in enumerate(completions_text):
                trajectory_id = f"traj_{idx}"
                try:
                    hop_pred = self.adapter.parse(
                        self.react_model.react.signature, curr_completion
                    )
                    trajectories[idx][f"thought_{hop}"] = hop_pred["next_thought"]
                    trajectories[idx][f"tool_name_{hop}"] = hop_pred["next_tool_name"]
                    trajectories[idx][f"tool_args_{hop}"] = hop_pred["next_tool_args"]
                    try:
                        trajectories[idx][f"observation_{hop}"] = (
                            self.react_model.tools[hop_pred["next_tool_name"]](
                                hop_pred["next_tool_args"]["search_query"], trajectory_id
                            )
                        )
                        passages_this_hop = get_passages(
                            trajectories[idx][f"observation_{hop}"]
                        )

                        # signals finish
                        if passages_this_hop == 0:
                            finish_locations[idx] = (
                                hop
                                if finish_locations.get(idx) is None
                                else finish_locations[idx]
                            )

                    except Exception as e:
                        trajectories[idx][
                            f"observation_{hop}"
                        ] = f"Failed to execute {hop_pred['next_tool_name']}: {str(e)}"

                except Exception as ep:
                    trajectories[idx][f"thought_{hop}"] = "Failed to execute."
                    trajectories[idx][f"tool_name_{hop}"] = "Failed to execute."
                    trajectories[idx][f"tool_args_{hop}"] = "Failed to execute."
                    trajectories[idx][f"observation_{hop}"] = (
                        str(ep) + ". Use the correct format."
                    )

            
        # log the exact completion text instead of formatted ones
        completions_text = [t[0] for t in all_completion_texts]
        prompts_text = [t[0] for t in all_prompt_text]

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = (
                    f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                )
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c} for p, c in zip(prompts, completions)
                        ]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                            :, 0
                        ]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [
                        key for key in inputs[0] if key not in ["prompt", "completion"]
                    ]
                    reward_kwargs = {
                        key: [example[key] for example in inputs] for key in keys
                    }
                    output_reward_func = reward_func(
                        questions=questions, trajectories=trajectories, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]
                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max --> it will be different per hop
        for i, hop_completion_mask in enumerate(all_completion_masks):
            agg_completion_mask = self.accelerator.gather_for_metrics(
                hop_completion_mask.sum(1)
            )
            self._metrics[mode][f"completions_{i}/mean_length"].append(
                agg_completion_mask.float().mean().item()
            )
            self._metrics[mode][f"completions_{i}/min_length"].append(
                agg_completion_mask.float().min().item()
            )
            self._metrics[mode][f"completions_{i}/max_length"].append(
                agg_completion_mask.float().max().item()
            )

            # identify sequences that terminated with EOS and log their lengths
            agg_terminated_with_eos = self.accelerator.gather_for_metrics(
                is_eos.any(dim=1)
            )
            term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
            clipped_completions_ratio = 1 - len(term_completion_mask) / len(
                agg_completion_mask
            )
            self._metrics[mode][f"completions_{i}/clipped_ratio"].append(
                clipped_completions_ratio
            )
            if len(term_completion_mask) == 0:
                # edge case where no completed sequences are found
                term_completion_mask = torch.zeros(1, device=device)
            self._metrics[mode][f"completions_{i}/mean_terminated_length"].append(
                term_completion_mask.float().mean().item()
            )
            self._metrics[mode][f"completions_{i}/min_terminated_length"].append(
                term_completion_mask.float().min().item()
            )
            self._metrics[mode][f"completions_{i}/max_terminated_length"].append(
                term_completion_mask.float().max().item()
            )

        # Get the names of the reward functions
        reward_func_names = []
        for reward_func in self.reward_funcs:
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            reward_func_names.append(reward_func_name)

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # # Log prompt and completion texts
        # self._textual_logs["prompt"].extend(gather_object(prompts_text))
        # self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        final_outputs = [
            {
                "prompt_ids": pid,
                "prompt_mask": pm,
                "completion_ids": cid,
                "completion_mask": cm,
                "advantages": advantages,  # constant for each hop
                "old_per_token_logps": old_tlp,
                "ref_per_token_logps": ref_tlp,
            }
            for pid, pm, cid, cm, old_tlp, ref_tlp in zip(
                all_prompt_ids,
                all_prompt_mask,
                all_completion_ids,
                all_completion_masks,
                all_old_per_token_logps,
                all_ref_per_token_logps,
            )
        ]

        # apply loss on a random hop
        # return random.sample(final_outputs, 1)[0]
        return final_outputs

    @profiling_decorator
    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[list, torch.Tensor]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the accumulated local batch (Per-GPU batch size × Gradient accumulation steps)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire accumulated batch and splits it into smaller batches
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every gradient_accumulation_steps * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            generate_every = self.args.gradient_accumulation_steps * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                accumulated_local_batch = self._generate_and_score_completions(
                    accumulated_local_batch
                )
                _all_buffered_inputs = split_tensor_list(
                    accumulated_local_batch, self.args.gradient_accumulation_steps
                )
            inputs = []
            for buffered_inputs in _all_buffered_inputs:
                inputs.append(
                    buffered_inputs[self._step % self.args.gradient_accumulation_steps]
                )

            self._step += 1
        else:
            # In evaluation, there is neither gradient accumulation, nor multiple iterations
            inputs = self._generate_and_score_completions(accumulated_local_batch)

        return inputs

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            loss = self.compute_liger_loss(model, inputs)
        else:
            loss = self._compute_loss(model, inputs)

        return loss

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        losses = []
        for hop_input in inputs:
            # TODO: sagemaker compatibility patch
            # if is_sagemaker_mp_enabled():
            #     loss_mb = smp_forward_backward(model, hop_inputs, self.args.gradient_accumulation_steps)
            #     return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                if self.model_accepts_loss_kwargs:
                    loss = self.compute_loss(model, hop_input)
                else:
                    loss = self.compute_loss(
                        model, hop_input, num_items_in_batch=num_items_in_batch
                    )

            del hop_input

            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learnign rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # Finally we need to normalize the loss for reporting
                if num_items_in_batch is None:
                    loss = loss / self.args.gradient_accumulation_steps

                self.accelerator.backward(loss, **kwargs)

                losses.append(loss.detach())

        return sum(losses) / len(losses)
