import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from utils.boxes import box_iou
from utils.metrics import run_external_evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse prediction JSON files with Weighted Box Fusion.")
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="", help="Comma-separated model weights; defaults to equal weights.")
    parser.add_argument("--iou_threshold", type=float, default=0.55)
    parser.add_argument("--skip_box_threshold", type=float, default=0.001)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--ground_truth", default=None)
    parser.add_argument("--score_output", default=None)
    return parser.parse_args()


def load_predictions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(item["image_id"]): item.get("boxes", []) for item in data}


def fuse_cluster(cluster, total_model_weight):
    coordinate_weights = torch.tensor(
        [item["score"] * item["model_weight"] for item in cluster], dtype=torch.float32
    )
    boxes = torch.tensor([item["box"] for item in cluster], dtype=torch.float32)
    fused_box = (boxes * coordinate_weights[:, None]).sum(dim=0) / coordinate_weights.sum().clamp(min=1e-9)
    best_score_by_model = {}
    for item in cluster:
        weighted_score = item["score"] * item["model_weight"]
        best_score_by_model[item["model_index"]] = max(
            best_score_by_model.get(item["model_index"], 0.0), weighted_score
        )
    score = sum(best_score_by_model.values()) / max(1e-9, total_model_weight)
    return fused_box, min(1.0, score)


def weighted_box_fusion(model_predictions, model_weights, iou_threshold, skip_box_threshold, max_det):
    image_ids = sorted({image_id for predictions in model_predictions for image_id in predictions})
    total_model_weight = sum(model_weights)
    output = []

    for image_id in image_ids:
        by_class = defaultdict(list)
        for model_index, predictions in enumerate(model_predictions):
            for prediction in predictions.get(image_id, []):
                score = float(prediction.get("confidence", 0.0))
                if score < skip_box_threshold:
                    continue
                by_class[prediction["class"]].append(
                    {
                        "box": [float(value) for value in prediction["bbox"]],
                        "score": score,
                        "model_weight": model_weights[model_index],
                        "model_index": model_index,
                    }
                )

        fused_predictions = []
        for class_name, candidates in by_class.items():
            clusters = []
            for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
                best_cluster = None
                best_iou = -1.0
                candidate_box = torch.tensor(candidate["box"], dtype=torch.float32).view(1, 4)
                for cluster_index, cluster in enumerate(clusters):
                    fused_box, _ = fuse_cluster(cluster, total_model_weight)
                    iou = float(box_iou(candidate_box, fused_box.view(1, 4))[0, 0])
                    if iou >= iou_threshold and iou > best_iou:
                        best_cluster = cluster_index
                        best_iou = iou
                if best_cluster is None:
                    clusters.append([candidate])
                else:
                    clusters[best_cluster].append(candidate)

            for cluster in clusters:
                fused_box, fused_score = fuse_cluster(cluster, total_model_weight)
                x1, y1, x2, y2 = [round(float(value), 2) for value in fused_box]
                if x2 <= x1 or y2 <= y1:
                    continue
                fused_predictions.append(
                    {
                        "class": class_name,
                        "confidence": round(fused_score, 6),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        fused_predictions.sort(key=lambda item: item["confidence"], reverse=True)
        output.append({"image_id": image_id, "boxes": fused_predictions[:max_det]})
    return output


def main():
    args = parse_args()
    model_predictions = [load_predictions(path) for path in args.predictions]
    if args.weights:
        model_weights = [float(value.strip()) for value in args.weights.split(",") if value.strip()]
    else:
        model_weights = [1.0] * len(model_predictions)
    if len(model_weights) != len(model_predictions):
        raise ValueError("--weights must contain one value per prediction file.")
    if any(weight <= 0 for weight in model_weights):
        raise ValueError("All model weights must be positive.")

    fused = weighted_box_fusion(
        model_predictions,
        model_weights,
        args.iou_threshold,
        args.skip_box_threshold,
        args.max_det,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fused, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved WBF predictions from {len(model_predictions)} models to {output_path}")

    if args.ground_truth:
        score_output = args.score_output or str(output_path.with_name(output_path.stem + "_score.json"))
        metrics = run_external_evaluator(args.ground_truth, output_path, score_output)
        print(
            f"mAP50={metrics['mAP@0.5']:.6f} "
            f"mAP50-95={metrics['mAP@0.5:0.95']:.6f}"
        )


if __name__ == "__main__":
    main()
