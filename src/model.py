import dspy
from dspy.predict.react import Tool
from typing import (
    Any,
    Callable,
    Literal,
    List,
    Dict,
)
from transformers import AutoTokenizer
from dspy.signatures.signature import ensure_signature
import requests
from .search.search_utils import search_by_http
from .search.search_fineweb import query_fineweb
from .data.data_utils import format_input_context, load_corpus, load_wiki18_corpus
from .data.truncation_utils import adaptive_truncate_for_react, format, compute_trajectory_tokens

# reduce this to 12000 for bcplus prompt optimization, else 20000
MODEL_MAX_LEN = 12000

class SearchModule:
    """
    A search module for retrieving documents from different corpora.
    
    Supports wiki5M/wiki20M and KILT-based E5 retrieval methods.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.retrieved_docs = set()  # Track retrieved document IDs/texts
        self.trajectory_retrieved_docs = {}  # Track per-trajectory retrieved docs

        if args.search.rm in ["wiki5M", "wiki20M"]:
            self.api_url = f"http://127.0.0.1:{args.search.search_port}".rstrip("/")
        elif args.search.rm == "e5-large":
            self.corpus = load_corpus()
        elif args.search.rm == "e5-base":
            self.corpus = load_wiki18_corpus()
        elif args.search.rm == "qwen8b" and args.data.dataset_name == "bcplus":
            self.api_url = f"http://0.0.0.0:{args.search.search_port}".rstrip("/")

    def __call__(self, search_query: str, trajectory_id: str = None) -> Dict[str, Any]:
        return self._search_unique(search_query, trajectory_id)

    def _search_unique(self, search_query: str, trajectory_id: str = None) -> Dict[str, Any]:
        """Search for unique documents, expanding search if duplicates are found"""
        
        # Use trajectory-specific retrieved docs if trajectory_id is provided
        if trajectory_id is not None:
            if trajectory_id not in self.trajectory_retrieved_docs:
                self.trajectory_retrieved_docs[trajectory_id] = set()
            retrieved_docs = self.trajectory_retrieved_docs[trajectory_id]
        else:
            retrieved_docs = self.retrieved_docs
            
        if self.args.search.rm in ["wiki5M", "wiki20M"]:
            try:
                passages: List[Dict[str, Any]] = []
                max_fetch_k = self.args.search.ndocs * 50
                
                response = requests.get(
                    f"{self.api_url}/api/search",
                    params={"query": search_query, "k": max_fetch_k},
                )
                response.raise_for_status()
                results = response.json()["topk"]

                new_passages = []
                for item in results:
                    doc_hash = hash(item["text"])
                    if doc_hash not in retrieved_docs:
                        new_passages.append({"long_text": item["text"], "score": item["score"]})
                        retrieved_docs.add(doc_hash)
                        if len(passages) + len(new_passages) >= self.args.search.ndocs:
                            break
                
                passages.extend(new_passages)
                passages = passages[:self.args.search.ndocs]

            except Exception as e:
                print(f"[RemoteSearchModule API Error] {e}")
                passages = []

        elif self.args.search.rm in ["e5-large", "e5-base"]:
            # retrieves 250 documents by default, picks top 5 unique
            retriever_results: List[Dict] = search_by_http(search_query, port=self.args.search.search_port)
            
            passages: List[str] = []
            for res in retriever_results:
                doc_id = res["doc_id"]
                if doc_id not in retrieved_docs:
                    passage = format_input_context(self.corpus[int(doc_id)])
                    passages.append({"long_text": passage, "score": res["score"]})
                    retrieved_docs.add(doc_id)
                    if len(passages) >= self.args.search.ndocs:
                        break
            
            # following KILT implementation
            passages = passages[::-1] if self.args.search.rm == "e5-large" else passages

        elif self.args.data.dataset_name == "bcplus":
            try:
                passages: List[Dict[str, Any]] = []
                max_fetch_k = self.args.search.ndocs * 10
                
                response = requests.get(
                    f"{self.api_url}/api/search",
                    params={"query": search_query, "k": max_fetch_k},
                )
                response.raise_for_status()
                results = response.json()
                new_passages = []
                for item in results:
                    doc_id = int(item["docid"])
                    if doc_id not in retrieved_docs:
                        new_passages.append({"long_text": item["snippet"], "score": item["score"], "doc_id": doc_id})
                        retrieved_docs.add(doc_id)
                        if len(passages) + len(new_passages) >= self.args.search.ndocs:
                            break
                
                passages.extend(new_passages)
                passages = passages[:self.args.search.ndocs]

            except Exception as e:
                print(f"[RemoteSearchModule API Error] {e}")
                passages = []
        
        elif self.args.data.dataset_name == "researchy":
            try:
                passages: List[Dict[str, Any]] = []
                max_fetch_k = self.args.search.ndocs * 10
                
                results = query_fineweb(search_query, max_fetch_k)
                new_passages = []
                for item in results:
                    doc_hash = hash(item["text"])
                    if doc_id not in retrieved_docs:
                        new_passages.append({"long_text": item["text"]})
                        retrieved_docs.add(doc_hash)
                        if len(passages) + len(new_passages) >= self.args.search.ndocs:
                            break
                
                passages.extend(new_passages)
                passages = passages[:self.args.search.ndocs]

            except Exception as e:
                print(f"[RemoteSearchModule API Error] {e}")
                passages = []
        
        else:
            raise (NotImplementedError)

        return {
            "search_query": search_query,
            "passages_this_hop": passages,
        }

    def reset_retrieved_docs(self, trajectory_id: str = None):
        """Reset the tracking of retrieved documents"""
        if trajectory_id is not None:
            if trajectory_id in self.trajectory_retrieved_docs:
                self.trajectory_retrieved_docs[trajectory_id].clear()
        else:
            self.retrieved_docs.clear()
            self.trajectory_retrieved_docs.clear()

class CustomReAct(dspy.ReAct):
    def __init__(self, signature, tools: list[Callable], args):
        """
        signature: A DSPy signature defining the input/output interface for the agent.
            This specifies what inputs the agent expects and what outputs it should produce.
        tools (list[Callable]): A list of tools available to the agent. Each tool can be:
            - A function
            - A callable class
            - A dspy.Tool instance
            Tools are automatically wrapped in Tool instances if they aren't already.
        args: Configuration object containing agent parameters:
            - max_iters: Maximum number of reasoning iterations
            - max_retries: Maximum number of retry attempts for failed actions
            - no_finish: If True, agent won't automatically add a 'finish' tool
        """

        self.args = args
        self.max_iters = args.model.max_iters
        self.max_retries = args.model.max_retries
        self.no_finish = args.model.no_finish
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model.model_name_or_path)
        
        self.signature = signature = ensure_signature(signature)

        tools = [
            t if isinstance(t, Tool) or hasattr(t, "input_variable") else Tool(t)
            for t in tools
        ]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        if not self.no_finish:
            instr.extend(
                [
                    f"You will be given {inputs} and your goal is to finish with {outputs}.\n",
                    "To do this, you will interleave Thought, Tool Name, and Tool Args, and receive a resulting Observation.\n",
                    "Thought can reason about the current situation, and Tool Name can be the following types:\n",
                ]
            )
            finish_desc = f"Signals that the final outputs, i.e. {outputs}, are now available and marks the task as complete."
            finish_args = (
                {}
            )  # k: v.annotation for k, v in signature.output_fields.items()}
            tools["finish"] = Tool(
                func=lambda **kwargs: "Completed.",
                name="finish",
                desc=finish_desc,
                args=finish_args,
            )

        else:
            instr.extend(
                [
                    f"You will be given {inputs}.\n",
                    "You must decide what to do next by providing:\n",
                    "- Thought: reasoning about the current step\n",
                    "- Tool Name: one of the available tools\n",
                    "- Tool Args: A valid JSON with the necessary keys.\n",
                    "Repeat this for a few steps to gather information.\n",
                    "You do not need to produce a final answer — just use the tools iteratively.",
                ]
            )

            instr.append(
                "Make sure to include all required Tool Args as a valid JSON object.\n"
            )

        for idx, tool in enumerate(tools.values()):
            args = (
                tool.args if hasattr(tool, "args") else str({tool.input_variable: str})
            )
            desc = (
                f", whose description is <desc>{tool.desc}</desc>."
                if tool.desc
                else "."
            ).replace("\n", "  ")
            desc += f" It takes arguments {args} in JSON format."
            instr.append(f"({idx+1}) {tool.name}{desc}")

        if tools.keys() == 1:
            args = (
                tool.args if hasattr(tool, "args") else str({tool.input_variable: str})
            )
            desc = (
                f", whose description is <desc>{tool.desc}</desc>."
                if tool.desc
                else "."
            ).replace("\n", "  ")
            desc += f" It takes arguments {args} in JSON format."
            instr.append(f"({idx+1}) {tool.name.lower()}{desc}")

        # add fallback aliases
        tool_names = (
            tools.keys()
            if len(tools.keys()) > 1
            else list(tools.keys()) + [tool.name.lower()]
        )

        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append(
                "next_tool_name", dspy.OutputField(), type_=Literal[tuple(tool_names)]
            )
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools = tools
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(
            fallback_signature
        )

    def get_next_hop(self, **input_args):
        trajectory = input_args.pop("trajectory", {})
        lm_query = input_args.pop("lm_query", None)
        idx = input_args.pop("hop")
        tries = 0

        try:
            max_retries = self.max_retries if self.max_retries is not None else 3
            while True:
                if  (self.args.model.max_iters > 5 or self.args.data.dataset_name in ["bcplus", "researchy"]) and compute_trajectory_tokens(trajectory, self.tokenizer) > MODEL_MAX_LEN:
                    print("[INFO] Trajectory too long, reducing length")
                    # copies trajectory internally
                    trajectory = adaptive_truncate_for_react(
                        trajectory, 
                        tokenizer=self.tokenizer,
                        preserve_recent=2, 
                        max_seq_length=MODEL_MAX_LEN
                    )
                        
                pred = self.react(**input_args, trajectory=format(trajectory))
                tool_name = pred.next_tool_name

                if "finish" in tool_name and not self.no_finish:
                    tool_name = "finish"
                elif "AdvancedSearch" in tool_name or "advancedsearch" in tool_name:
                    tool_name = "AdvancedSearch"

                try:
                    if "trajectory_id" in pred.next_tool_args.keys():
                        pred.next_tool_args.pop("trajectory_id")
                    # make sure to not use model's trajectory ID argument
                    if lm_query is not None:
                        with dspy.context(lm_query=lm_query):
                            observation = self.tools[tool_name](**pred.next_tool_args)
                    else:
                        observation = self.tools[tool_name](**pred.next_tool_args)
                    break
                except Exception as e1:
                    observation = f"Failed to execute {tool_name}: {str(e1)}. Retry with correct arguments."
                    trajectory[f"observation_{tries+idx+1}"] = observation
                    trajectory[f"thought_{tries+idx+1}"] = pred.next_thought
                    trajectory[f"tool_name_{tries+idx+1}"] = tool_name
                    trajectory[f"tool_args_{tries+idx+1}"] = pred.next_tool_args

                tries += 1
                if tries > max_retries:
                    break

        except Exception as e2:
            print(f"Exception: {e2}")
            return {
                f"thought_{idx}": "Failed to execute.",
                f"tool_name_{idx}": "Failed to execute.",
                f"tool_args_{idx}": "Failed to execute.",
                f"observation_{idx}": f"Tool call failed with error: {str(e2)}",
            }

        output = {}

        output[f"thought_{idx}"] = pred.next_thought
        output[f"tool_name_{idx}"] = tool_name
        output[f"tool_args_{idx}"] = pred.next_tool_args
        output[f"observation_{idx}"] = observation

        return output

    def forward(self, **input_args):
        trajectory = {}

        trajectory["tool_name_init"] = "AdvancedSearch"
        trajectory["tool_args_init"] = {"search_query": input_args["question"]}
        trajectory["thought_init"] = (
            "I will start by searching for relevant documents with the original question as the search query."
        )

        trajectory["observation_init"] = self.tools["AdvancedSearch"](input_args["question"], trajectory_id=None)

        lm_answer = input_args.pop("lm_answer", None)
        lm_query = input_args.pop("lm_query", None)
        
        for idx in range(self.max_iters):
            output = self.get_next_hop(
                question=input_args["question"],
                trajectory=trajectory.copy(),
                hop=idx,
                lm_query=lm_query,
            )

            trajectory.update(output)

            if not self.no_finish and trajectory[f"tool_name_{idx}"] == "finish":
                break
        
        if (self.args.model.max_iters > 5 or self.args.data.dataset_name in ["bcplus", "researchy"]) and compute_trajectory_tokens(trajectory, self.tokenizer) > MODEL_MAX_LEN:
            print("[INFO] Trajectory too long, reducing length")
            # copies trajectory internally
            trunc_trajectory = adaptive_truncate_for_react(
                trajectory, 
                tokenizer=self.tokenizer,
                preserve_recent=2, 
                max_seq_length=MODEL_MAX_LEN
            )
        else:
            trunc_trajectory = trajectory
            
        
        if lm_answer is not None:
            with dspy.context(lm=lm_answer):
                extract = self.extract(**input_args, trajectory=format(trunc_trajectory))
        else:
            extract = self.extract(**input_args, trajectory=format(trunc_trajectory))

        # reset tracker
        self.tools["AdvancedSearch"].func.reset_retrieved_docs()
        
        if input_args.get("get_original_traj") or self.args.model.max_iters > 5:
            return dspy.Prediction(trajectory=trunc_trajectory, **extract), trajectory

        else:
            return dspy.Prediction(trajectory=trunc_trajectory, **extract)

class Hopper(dspy.Module):
    def __init__(self, args, debug=False, custom=True):
        super().__init__()

        desc = (
            "Searches documents using a search_query.\n"
            "Arguments:\n"
            '- "search_query": an optimized search query for dense passage retrieval.\n'
            'IMPORTANT: YOU MUST always PROVIDE "search_query" in the arguments!'
        )

        self.debug = debug
        self.custom = custom

        query_generator = Tool(
            func=SearchModule(args), name="AdvancedSearch", desc=desc
        )
        print(query_generator.args)

        if custom:  # default
            self.orchestrator = CustomReAct(
                "question -> answer", tools=[query_generator], args=args
            )
            print("[INFO] Initialized custom model")

        else:
            self.orchestrator = dspy.ReAct(
                "question -> answer", tools=[query_generator], max_iters=args.model.max_iters
            )
            print("[INFO] Initialized react basic model")

    def forward(self, question, lm_answer=None, lm_query=None, **kwargs):
        output = (
            self.orchestrator(question=question, lm_answer=lm_answer, lm_query=lm_query, **kwargs)
            if self.custom is True
            else self.orchestrator(question=question, **kwargs)
        )

        return output

    def next_hop(self, question, hop, lm=None, trajectory={}):
        # DO NOT UPDATE THE DICT IN PLACE!
        pred = self.orchestrator.get_next_hop(
            question=question, trajectory=trajectory.copy(), hop=hop
        )
        return pred.copy()

    def update_react_prompt(self, prompt_path):
        self.orchestrator.load(prompt_path)
