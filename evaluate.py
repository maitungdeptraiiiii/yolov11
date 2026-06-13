import argparse

from utils.metrics import run_external_evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection predictions with COCO-style AP interpolation.")
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = run_external_evaluator(args.ground_truth, args.predictions, args.output)
    print(f"mAP50={metrics['mAP@0.5']:.6f} mAP50-95={metrics['mAP@0.5:0.95']:.6f}")


if __name__ == "__main__":
    main()
