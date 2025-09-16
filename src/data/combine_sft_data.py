import json
import random
import os
from argparse import ArgumentParser


def main(args):
    random.seed(args.seed)

    finish = args.finish_path
    no_finish = args.nofinish_path

    with open(no_finish) as f:
        data_nofinish = json.load(f)

    with open(finish) as f:
        data_finish = json.load(f)

    questions = set(d["question"] for d in data_finish)
    questions = list(questions)

    nofinish_questions = random.sample(
        questions, int(args.split_ratio * len(questions))
    )
    finish_questions = [q for q in questions if q not in nofinish_questions]

    output_data = [ex for ex in data_nofinish if ex["question"] in nofinish_questions]
    output_data += [ex for ex in data_finish if ex["question"] in finish_questions]

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "train_sft.json"), "w") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--finish_path", type=str, required=True)
    parser.add_argument("--nofinish_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--split_ratio", type=float, default=0.9, help="Ratio of no-finish samples"
    )
    parser.add_argument("--seed", type=int, default=663, help="Random seed")

    args = parser.parse_args()

    main(args)
