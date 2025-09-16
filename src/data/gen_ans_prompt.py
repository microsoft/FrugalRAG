import json
from pathlib import Path


def clear_demos_all_prompts(prompt_root: str):
    """
    Walks through all folders in `prompt_root`, finds any folder named `react`,
    clears the `react.demos` list in every JSON inside, and writes results
    to a parallel folder where `react` is replaced with `ans`.

    Args:
        prompt_root (str): Root folder for prompts (e.g., 'prompts')
    """
    root_path = Path(prompt_root)

    for react_dir in root_path.rglob("react"):
        if not react_dir.is_dir():
            continue

        for json_file in react_dir.rglob("*.json"):
            try:
                # Read original JSON
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Modify if keys exist
                if "react" in data and isinstance(data["react"], dict):
                    data["react"]["demos"] = []

                # Compute equivalent 'ans' path
                relative_path = json_file.relative_to(react_dir)
                ans_dir = react_dir.with_name("ans")
                output_path = ans_dir / relative_path

                # Ensure target folder exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Write modified JSON
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                print(f"Processed: {json_file} → {output_path}")

            except Exception as e:
                print(f"Error processing {json_file}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_root", type=str, required=True,
                        help="Root folder containing prompt subfolders (e.g., 'prompts')")
    args = parser.parse_args()

    clear_demos_all_prompts(args.prompt_root)
