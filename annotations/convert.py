import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert prediction JSON to Kaggle submission CSV.")
    parser.add_argument("--input", required=True, help="Path to predictions.json from predict.py")
    parser.add_argument("--output", required=True, help="Path to output submission.csv")
    return parser.parse_args()


def convert_box(box):
    x_min, y_min, x_max, y_max = box["bbox"]
    return {
        "x_min": float(x_min),
        "y_min": float(y_min),
        "x_max": float(x_max),
        "y_max": float(y_max),
        "class": box["class"],
        "confidence": float(box["confidence"]),
    }


def main():
    args = parse_args()
    predictions = json.loads(Path(args.input).read_text(encoding="utf-8"))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "bounding_boxes"])
        writer.writeheader()
        for item in predictions:
            boxes = [convert_box(box) for box in item.get("boxes", [])]
            writer.writerow(
                {
                    "image_id": item["image_id"],
                    "bounding_boxes": json.dumps(boxes, ensure_ascii=False),
                }
            )


if __name__ == "__main__":
    main()
