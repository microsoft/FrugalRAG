import dspy
import os
import json
import random
import dsp
from src.data.data_utils import get_prompt_optimization_dataset
from src.parser_ import get_args
from src.model import Hopper


random.seed(663)
import multiprocessing as mp


def generate_fewshot(args, lm):
    if not (os.path.exists(args.paths.prompt_path) and len(os.listdir(args.paths.prompt_path)) > 3):
            
        finish_prefix = "nofinish" if args.model.no_finish else "finish"
        out_template = os.path.join(args.paths.prompt_save_path, "react/bootstrapped")
        os.makedirs(os.path.dirname(out_template), exist_ok=True)
        
        candidates = args.data.candidates
        # to prevent memory errors
        if args.data.dataset_name == "bcplus":
            num_candidate_programs = 3 * candidates
            max_bootstrapped_demos = 1
        elif args.model.max_iters > 5 and args.model.no_finish:
            num_candidate_programs = 5 * candidates
            max_bootstrapped_demos = 1
        else:
            num_candidate_programs = 5 * candidates
            max_bootstrapped_demos = 3

        pipeline = Hopper(args, custom=True, debug=False)
        config = {
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "num_threads": 32,
            "num_candidate_programs": num_candidate_programs,
        }

        train_dataset, val_dataset = get_prompt_optimization_dataset(args)
        # save trainset, valset
        train_questions = [t["question"] for t in train_dataset]
        val_questions = [t["question"] for t in val_dataset]

        with open(os.path.join(args.paths.prompt_save_path, "optimp_questions.json"), "w") as f:
            json.dump(train_questions + val_questions, f, indent=1)
        
        # with open(os.path.join(args.paths.prompt_save_path, "args.json"), "w") as f:
        #     json.dump(vars(args), f, indent=1)

        tp = dspy.BootstrapFewShotWithRandomSearch(
            metric=custom_metric_supportf1, **config
        )

        basicmh_bs = tp.compile(pipeline, trainset=train_dataset, valset=val_dataset)

        all_ensembles = [prog for *_, prog in basicmh_bs.candidate_programs]
        ensemble = []
        for i in all_ensembles:
            if "Example" not in str(i.dump_state()):
                continue
            ensemble.append(i)

        # sorted from best to worst
        ensemble = ensemble[:candidates]
        ensemble.insert(0, pipeline)

        for idx, prog in enumerate(ensemble):
            out_path = f"{out_template}_{idx}.json"
            prog.orchestrator.save(out_path)

        print(f"saved models to {os.path.dirname(out_path)}")
    else:
        print("already done")

def custom_metric_supportf1(example, pred, trace=None):
    assert type(example.answer) is str or type(example.answer) is list
    
    return (
        dsp.answer_match(pred.answer, example.answer)
        if isinstance(example.answer, list)
        else dsp.answer_match(pred.answer, [example.answer])
    )


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

    try:
        generate_fewshot(args, lm)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Exception occurred: {e}")
    finally:
        # Clean up multiprocessing pools, if any
        import gc
        gc.collect()
