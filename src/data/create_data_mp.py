import dspy
import os
import json
from tqdm import tqdm
import traceback
from src.evaluation import metrics
from src.parser_ import get_args
from src.model import Hopper
from src.data.data_utils import get_original_dataset, split_into_sentences

class DataGenerator:
    def __init__(self, args):
        self.args = args
        self.model = Hopper(args, custom=True)
    
    def _get_coverage(self, retrieved_passages, flattened_gold_sentences):
        retrieved_units = []
        if self.args.data.dataset_name in "2wiki":
            for para in retrieved_passages:
                retrieved_units.extend(split_into_sentences(para))
        elif self.args.data.dataset_name == "musique":
            retrieved_units = retrieved_passages
        else:
            raise NotImplementedError
        
        # compute coverage of gold sentences
        covered = 0
        matched_retrieved = set()
        for gold in flattened_gold_sentences:
            best_match = 0.0
            best_idx = None
            for i, pred in enumerate(retrieved_units):
                f1_val = metrics.f1(pred, gold)
                if f1_val > best_match:
                    best_match = f1_val
                    best_idx = i
            if best_match > 0.7:  # threshold for "covered"
                covered += 1
                matched_retrieved.add(best_idx)

        recall = covered / len(flattened_gold_sentences)
        return recall

    def get_rewards_coverage(self, gold_paras, trajectories, hop, curr_seen):
        if self.args.search.rm in ["wiki5M", "wiki20M"]:
            curr_seen = [p.split("|")[1].strip() for p in curr_seen]
        elif self.args.search.rm in ["e5-large", "e5-base"]:
            curr_seen = ["\n".join(p.split("\n")[1:]) for p in curr_seen]
        
        rewards = [-1] * len(trajectories)
        for idx, trajectory in enumerate(trajectories):
            observation = trajectory.get(f"observation_{hop}", None)
            if observation and isinstance(observation, dict):

                for passage in observation["passages_this_hop"]:
                    if self.args.search.rm in ["wiki20M", "wiki5M"]:
                        curr_seen.append(passage["long_text"].split("|")[1].strip())
                    elif self.args.search.rm in ["e5-large", "e5-base"]:
                        curr_seen.append("\n".join(passage["long_text"].split("\n")[1:]))
                        
                f1_reward = self._get_coverage(curr_seen, gold_paras)
                rewards[idx] = float(f1_reward * 100)
                
        return rewards
    
    def get_rewards_recall(self, gold_titles, trajectories, hop, curr_seen):
        if self.args.data.dataset_name == "hotpot" and self.args.search.rm in ["wiki5M", "wiki20M"]:
            curr_seen = [p.split("|")[0].strip() for p in curr_seen]
        elif self.args.data.dataset_name == "hotpot" and self.args.search.rm in ["e5-large", "e5-base"]:
            curr_seen = [p.split("\n")[0].strip() for p in curr_seen]
        
        rewards = [-1] * len(trajectories)
        for idx, trajectory in enumerate(trajectories):
            observation = trajectory.get(f"observation_{hop}", None)
            if observation and isinstance(observation, dict):
                for passage in observation["passages_this_hop"]:
                    if self.args.search.rm in ["wiki20M", "wiki5M"]:
                        curr_seen.append(passage["long_text"].split("|")[0].strip())                        
                    elif self.args.search.rm in ["e5-large", "e5-base"]:
                        curr_seen.append(passage["long_text"].split("\n")[0].strip())
                    elif self.args.data.dataset_name == "bcplus" and "qwen" in self.args.search.rm:
                        curr_seen.append(int(passage["doc_id"]))

                
                if self.args.data.dataset_name == "bcplus" and "qwen" in self.args.search.rm:
                    recall_reward = metrics.recall_ret(curr_seen, gold_titles)
                else:
                    recall_reward = metrics.title_recall(curr_seen, gold_titles)
                    
                rewards[idx] = float(recall_reward * 100)
                
        return rewards

    def process_example(self, example, gold, gold_titles, example_idx):
        main_trajectory = {}
        main_trajectory_rewards, sft_data = [], []
        main_trajectory_retrieved_docs = set()
        seen_passages = []

        main_trajectory["tool_name_init"] = "AdvancedSearch"
        main_trajectory["tool_args_init"] = {"search_query": example["question"]}
        main_trajectory["thought_init"] = (
            "I will start by searching for relevant documents with the original question as the search query."
        )
        
        self.model.orchestrator.tools["AdvancedSearch"].func.reset_retrieved_docs()
        
        observation_init_dict = self.model.orchestrator.tools["AdvancedSearch"](
            example["question"]
        )
        passages = observation_init_dict["passages_this_hop"]

        main_trajectory["observation_init"] = observation_init_dict
        
        # Main trajectory document tracker
        for passage in passages:
            main_trajectory_retrieved_docs.add(hash(passage["long_text"]))
            if self.args.data.dataset_name == "bcplus":
                seen_passages.append(int(passage["doc_id"]))
            else:
                seen_passages.append(passage["long_text"])
            
        for hop in range(self.args.model.max_iters):
            tasks = [
                {
                    "question": example["question"],
                    "main_trajectory": main_trajectory,
                    "hop": hop,
                    "ensemble_filepath": ensemble_filepaths[i],
                    "candidate_idx": i,
                    "main_trajectory_retrieved_docs": main_trajectory_retrieved_docs.copy()
                }
                for i in range(self.args.data.candidates + 1)
            ]
            curr_trajectories = [None] * (self.args.data.candidates + 1)

            exec_pairs = [(self.process_candidate, task) for task in tasks]
            candidate_executor = dspy.parallel.Parallel(
                num_threads=min(self.args.data.candidates + 1, len(exec_pairs)),
                # num_threads=1,
                max_errors=10,
                provide_traceback=True,
            )

            results = candidate_executor.forward(exec_pairs)
            
            for traj_idx, hop_output in results:
                curr_trajectories[traj_idx] = hop_output

            # all nth hop trajectories
            if self.args.data.dataset_name in ["musique", "2wiki"]:
                rewards = self.get_rewards_coverage(
                    gold, [t for t in curr_trajectories if t], hop=hop, curr_seen=seen_passages[:]
                )
            elif self.args.data.dataset_name in ["hotpot", "bcplus"]:
                rewards = self.get_rewards_recall(
                    gold_titles, [t for t in curr_trajectories if t], hop=hop, curr_seen=seen_passages[:]
                )
            
            # common main graph, add observation
            best_reward = max(rewards)
            selected_idx = rewards.index(best_reward)
            main_trajectory_rewards.append(best_reward)

            sft_data.append(
                {
                    "input": main_trajectory.copy(),
                    "question": example["question"],
                    "output": curr_trajectories[selected_idx],
                }
            )

            main_trajectory.update(curr_trajectories[selected_idx])
            
            
            # Update main trajectory retrieved docs with the selected documents
            if curr_trajectories[selected_idx] and f"observation_{hop}" in curr_trajectories[selected_idx]:
                observation = curr_trajectories[selected_idx][f"observation_{hop}"]
                if isinstance(observation, dict) and "passages_this_hop" in observation:
                    for passage in observation["passages_this_hop"]:
                        main_trajectory_retrieved_docs.add(hash(passage["long_text"]))
                        if self.args.data.dataset_name == "bcplus":
                            seen_passages.append(int(passage["doc_id"]))
                        else:
                            seen_passages.append(passage["long_text"])
            

            if main_trajectory[f"tool_name_{hop}"] == "finish":
                main_trajectory[f"tool_args_{hop}"] = {}
                break

        output = {
            "example_idx": example_idx,
            "sft_data": sft_data,
            "main_trajectory": main_trajectory,
            "main_trajectory_rewards": main_trajectory_rewards,
        }

        with open(
            os.path.join(self.args.paths.output_path, "examples", f"{example_idx}.json"), "w"
        ) as f:
            json.dump(output, f, indent=1)

        return output

    def make_dataset(self, ensemble_filepaths):
        sft_data = []

        os.makedirs(self.args.paths.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.args.paths.output_path, "examples"), exist_ok=True)
        # with open(os.path.join(self.args.paths.output_path, "args.json"), "w") as f:
        #     json.dump(vars(self.args), f, indent=1)

        train_dataset, gold_titles, gold_sent = get_original_dataset(self.args)
        print(f"[INFO] The dataset contains {len(train_dataset)} examples")
        ignore_list = []
        for file_ in os.listdir(os.path.join(self.args.paths.output_path, "examples")):
            try:
                with open(os.path.join(self.args.paths.output_path, "examples", file_)) as f:
                    check_none = json.load(f)
                if check_none:
                    ignore_list.append(file_.split(".")[0])
            except:
                print("[WARNING] corrupted json, skipping")

        pairs = [
            (
                self.process_example,
                {
                    "example": example,
                    "gold": gold_sent[example["_id"]],
                    "gold_titles": gold_titles[example["_id"]],
                    "example_idx": example["_id"],
                },
            )
            for example in train_dataset
            if example["_id"] not in ignore_list
        ]
        executor = dspy.parallel.Parallel(
            num_threads=min(16, len(pairs) + 1), max_errors=10, provide_traceback=True
            # num_threads=1, max_errors=10, provide_traceback=True
        )
        print(f"[INFO] Total number of examples: {len(pairs)}")

        results = executor.forward(pairs)

        for ignore_example in ignore_list:
            with open(
                os.path.join(
                    self.args.paths.output_path, "examples", str(ignore_example) + ".json"
                )
            ) as f:
                processed_output = json.load(f)

            results.append(processed_output)

        mapping = {td["_id"]: idx for idx, td in enumerate(train_dataset)}

        for result_dict in results:
            if result_dict:
                try:
                    example_idx = result_dict["example_idx"]
                    example = train_dataset[mapping[example_idx]]
                except Exception as e:
                    print(e)
                    continue

                sft_data.extend(result_dict["sft_data"])

                main_trajectory = result_dict["main_trajectory"]
                main_trajectory_rewards = result_dict["main_trajectory_rewards"]

        with open(os.path.join(self.args.paths.output_path, f"train_sft.json"), "w") as f:
            json.dump(sft_data, f, indent=1)

    def process_candidate(
        self, question, main_trajectory, hop, ensemble_filepath, candidate_idx, main_trajectory_retrieved_docs
    ):
        
        self.model.orchestrator.tools["AdvancedSearch"].func.reset_retrieved_docs()
        self.model.orchestrator.tools["AdvancedSearch"].func.retrieved_docs = main_trajectory_retrieved_docs.copy()
        
        self.model.update_react_prompt(ensemble_filepath)
        try:
            hop_output = self.model.next_hop(
                question, trajectory=main_trajectory, hop=hop
            )
            return candidate_idx, hop_output
        except Exception as e:
            traceback.print_exc()  # Prints full traceback
            print(f"[WARNING] hop failed, skipping {candidate_idx}")
            return candidate_idx, None


if __name__ == "__main__":
    args = get_args()
    
    lm = dspy.LM(
        model=args.model.model_name_or_path,
        deployment_name=args.model.model_name_or_path,
        api_base=f"http://0.0.0.0:{args.search.port[0]}/v1/",
        model_type="chat",
        api_key="EMPTY",
        custom_llm_provider="openai",
        provider="openai",
        cache=False,
        temperature=0.70,
    )

    dspy.settings.configure(lm=lm)
    dspy.configure(lm=lm)

    ensemble_filepaths = []
    for filename in os.listdir(args.paths.prompt_path):
        if filename.startswith("bootstrapped"):
            ensemble_filepaths.append(os.path.join(args.paths.prompt_path, filename))

    generator = DataGenerator(args)
    generator.make_dataset(ensemble_filepaths)
