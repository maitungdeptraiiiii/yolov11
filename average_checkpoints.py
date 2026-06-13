import argparse
import copy
from collections import OrderedDict
from pathlib import Path
import glob

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Average compatible YOLO checkpoints into a model soup.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", default="./models/model_soup.pth")
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    args = parse_args()
    checkpoint_paths = []
    for pattern in args.checkpoints:
        matches = sorted(glob.glob(pattern))
        checkpoint_paths.extend(matches if matches else [pattern])
    checkpoints = [load_checkpoint(path) for path in checkpoint_paths]
    states = [checkpoint["model"] for checkpoint in checkpoints]
    keys = list(states[0].keys())
    if any(list(state.keys()) != keys for state in states[1:]):
        raise ValueError("Checkpoints do not have identical model parameter keys.")

    averaged = OrderedDict()
    for key in keys:
        values = [state[key] for state in states]
        if values[0].dtype.is_floating_point:
            averaged[key] = torch.stack([value.float() for value in values]).mean(dim=0).to(values[0].dtype)
        else:
            averaged[key] = values[0].clone()

    best_index = max(
        range(len(checkpoints)),
        key=lambda idx: float(checkpoints[idx].get("score") or -1e9),
    )
    output = copy.deepcopy(checkpoints[best_index])
    output["model"] = averaged
    output["source_checkpoints"] = [str(Path(path)) for path in checkpoint_paths]
    output["source_scores"] = [checkpoint.get("score") for checkpoint in checkpoints]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Saved averaged checkpoint from {len(checkpoints)} sources to {output_path}")


if __name__ == "__main__":
    main()
