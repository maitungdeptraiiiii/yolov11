import json
from collections import defaultdict
from pathlib import Path

import torch

from .boxes import box_iou


def _load_json(value):
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text(encoding="utf-8"))
    return value


def _average_precision(recalls, precisions):
    if not recalls:
        return 0.0
    recall = torch.tensor(recalls, dtype=torch.float64)
    precision = torch.tensor(precisions, dtype=torch.float64)
    precision = torch.flip(torch.cummax(torch.flip(precision, dims=[0]), dim=0).values, dims=[0])
    recall_points = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    values = []
    for point in recall_points:
        mask = recall >= point
        values.append(precision[mask].max() if mask.any() else torch.tensor(0.0, dtype=torch.float64))
    return float(torch.stack(values).mean())


def _evaluate_class(gt_by_image, predictions, iou_threshold):
    num_ground_truth = sum(len(boxes) for boxes in gt_by_image.values())
    matched = {image_id: torch.zeros(len(boxes), dtype=torch.bool) for image_id, boxes in gt_by_image.items()}
    true_positives = []
    false_positives = []

    for prediction in sorted(predictions, key=lambda item: item[0], reverse=True):
        _, image_id, box = prediction
        gt_boxes = gt_by_image.get(image_id, [])
        if not gt_boxes:
            true_positives.append(0.0)
            false_positives.append(1.0)
            continue
        gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
        pred_tensor = torch.tensor(box, dtype=torch.float32).view(1, 4)
        ious = box_iou(pred_tensor, gt_tensor).view(-1)
        ious[matched[image_id]] = -1.0
        best_iou, best_index = ious.max(dim=0)
        if float(best_iou) >= iou_threshold:
            matched[image_id][best_index] = True
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)

    if not true_positives:
        return 0.0, 0, 0, 0.0, 0.0
    tp = torch.tensor(true_positives).cumsum(dim=0)
    fp = torch.tensor(false_positives).cumsum(dim=0)
    recalls = (tp / max(1, num_ground_truth)).tolist()
    precisions = (tp / (tp + fp).clamp(min=1e-9)).tolist()
    ap = _average_precision(recalls, precisions) if num_ground_truth > 0 else 0.0
    final_tp = int(tp[-1])
    final_fp = int(fp[-1])
    recall = final_tp / max(1, num_ground_truth)
    precision = final_tp / max(1, final_tp + final_fp)
    return ap, final_tp, final_fp, recall, precision


def evaluate_predictions(ground_truth, predictions):
    ground_truth = _load_json(ground_truth)
    predictions = _load_json(predictions)
    classes = ground_truth.get("classes", [])
    gt = {name: defaultdict(list) for name in classes}
    pred = {name: [] for name in classes}
    image_aliases = {}
    for image in ground_truth.get("images", []):
        image_id = str(image.get("id"))
        image_aliases[image_id] = image_id
        if image.get("file_name"):
            image_aliases[str(image["file_name"])] = image_id
            image_aliases[Path(str(image["file_name"])).name] = image_id

    for annotation in ground_truth.get("annotations", []):
        class_name = annotation.get("class")
        if class_name in gt:
            image_id = image_aliases.get(str(annotation["image_id"]), str(annotation["image_id"]))
            gt[class_name][image_id].append(annotation["bbox"])

    for image_prediction in predictions:
        raw_image_id = str(image_prediction.get("image_id"))
        image_id = image_aliases.get(raw_image_id, image_aliases.get(Path(raw_image_id).name, raw_image_id))
        for box_prediction in image_prediction.get("boxes", []):
            class_name = box_prediction.get("class")
            if class_name in pred:
                pred[class_name].append(
                    (float(box_prediction.get("confidence", 0.0)), image_id, box_prediction["bbox"])
                )

    iou_thresholds = [round(0.50 + 0.05 * index, 2) for index in range(10)]
    per_class = {}
    ap50_values = []
    ap5095_values = []
    total_tp = 0
    total_fp = 0
    total_gt = 0
    total_predictions = 0

    for class_name in classes:
        num_gt = sum(len(boxes) for boxes in gt[class_name].values())
        aps = []
        stats50 = None
        for threshold in iou_thresholds:
            result = _evaluate_class(gt[class_name], pred[class_name], threshold)
            aps.append(result[0])
            if threshold == 0.50:
                stats50 = result
        ap50, tp50, fp50, recall50, precision50 = stats50
        ap5095 = sum(aps) / len(aps)
        if num_gt > 0:
            ap50_values.append(ap50)
            ap5095_values.append(ap5095)
        total_tp += tp50
        total_fp += fp50
        total_gt += num_gt
        total_predictions += len(pred[class_name])
        per_class[class_name] = {
            "ap": round(ap50, 6),
            "ap50": round(ap50, 6),
            "ap50_95": round(ap5095, 6),
            "num_ground_truth": num_gt,
            "num_predictions": len(pred[class_name]),
            "true_positives": tp50,
            "false_positives": fp50,
            "recall": round(recall50, 6),
            "precision": round(precision50, 6),
        }

    map50 = sum(ap50_values) / max(1, len(ap50_values))
    map5095 = sum(ap5095_values) / max(1, len(ap5095_values))
    return {
        "mAP@0.5": round(map50, 6),
        "mAP@0.5:0.95": round(map5095, 6),
        "iou_threshold": 0.5,
        "num_ground_truth_boxes": total_gt,
        "num_predictions": total_predictions,
        "micro_precision": round(total_tp / max(1, total_tp + total_fp), 6),
        "micro_recall": round(total_tp / max(1, total_gt), 6),
        "per_class": per_class,
    }


def run_external_evaluator(ground_truth, predictions, output):
    metrics = evaluate_predictions(ground_truth, predictions)
    Path(output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
