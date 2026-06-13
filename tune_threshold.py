import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from models import ConvNeXtYOLO, DarknetYOLOv8
from utils.boxes import clip_boxes, nms
from utils.data import DEFAULT_CLASSES, image_to_tensor, load_annotation_file, resize_letterbox, undo_letterbox
from utils.inference import decode_outputs, save_predictions
from utils.metrics import run_external_evaluator


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Tune confidence and NMS IoU thresholds on validation data.")
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--checkpoint", default="./models/best.pth")
    parser.add_argument("--output", default="./models/threshold_tuning.json")
    parser.add_argument("--pred_dir", default="./models/threshold_tuning")
    parser.add_argument("--img_size", type=int, default=960)
    parser.add_argument("--conf_values", default="0.001,0.003,0.005,0.01,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--iou_values", default="0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--max_det", type=int, default=100)
    parser.add_argument("--backbone", choices=["convnext_tiny", "darknet"], default="convnext_tiny")
    parser.add_argument("--neck_width", type=float, default=1.0)
    parser.add_argument("--neck_depth", type=float, default=0.34)
    parser.add_argument("--p2_head", action="store_true")
    parser.add_argument("--scale", choices=["n", "s", "m", "l", "x", "none"], default="s")
    parser.add_argument("--width", type=float, default=0.50)
    parser.add_argument("--depth", type=float, default=0.34)
    parser.add_argument("--reg_max", type=int, default=16)
    parser.add_argument("--imagenet_normalize", action="store_true", help="Force ImageNet normalization during tuning.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(args):
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    classes = checkpoint.get("classes", DEFAULT_CLASSES)
    saved_args = checkpoint.get("args", {})
    architecture = checkpoint.get("architecture", "legacy")
    args.imagenet_normalize = bool(saved_args.get("imagenet_normalize", False) or args.imagenet_normalize)
    width = float(saved_args.get("width", args.width))
    depth = float(saved_args.get("depth", args.depth))
    reg_max = int(saved_args.get("reg_max", args.reg_max))
    backbone = saved_args.get("backbone")
    if backbone is None:
        state_keys = checkpoint.get("model", {}).keys()
        backbone = "convnext_tiny" if any(k.startswith("backbone.features.") for k in state_keys) else "darknet"
    if backbone == "convnext_tiny":
        neck_width = float(saved_args.get("neck_width", args.neck_width))
        neck_depth = float(saved_args.get("neck_depth", args.neck_depth))
        p2_head = bool(saved_args.get("p2_head", args.p2_head))
        model = ConvNeXtYOLO(
            num_classes=len(classes),
            reg_max=reg_max,
            pretrained_backbone=False,
            neck_width=neck_width,
            neck_depth=neck_depth,
            p2_head=p2_head,
        ).to(args.device)
    else:
        scale = saved_args.get("scale", args.scale)
        scale = None if scale in (None, "none") else scale
        model = DarknetYOLOv8(num_classes=len(classes), width=width, depth=depth, reg_max=reg_max, scale=scale).to(args.device)
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as exc:
        if backbone == "convnext_tiny" and architecture == "legacy":
            raise RuntimeError(
                "This checkpoint uses the previous ConvNeXt + C2f architecture and is not compatible with "
                "the current ConvNeXt + YOLO11 model. Train a new checkpoint with the updated architecture."
            ) from exc
        raise
    model.eval()
    return model, classes


@torch.no_grad()
def cache_raw_predictions(model, classes, items, image_dir, args, min_conf):
    cache = []
    image_dir = Path(image_dir)
    for item in tqdm(items, desc="cache val predictions"):
        info = item["info"]
        image_path = image_dir / Path(info["file_name"]).name
        image = Image.open(image_path).convert("RGB")
        padded, _, meta = resize_letterbox(image, torch.empty((0, 4)), args.img_size)
        tensor = image_to_tensor(padded, args.imagenet_normalize).unsqueeze(0).to(args.device)
        outputs = model(tensor)
        boxes, scores, labels = decode_outputs(
            outputs, args.img_size, model.strides, len(classes), conf_threshold=min_conf
        )
        boxes = undo_letterbox(boxes.cpu(), meta)
        boxes = clip_boxes(boxes, meta["orig_w"], meta["orig_h"])
        cache.append(
            {
                "image_id": Path(image_path).name,
                "boxes": boxes.cpu(),
                "scores": scores.cpu(),
                "labels": labels.cpu(),
            }
        )
    return cache


def make_predictions(cache, classes, conf_threshold, iou_threshold, max_det):
    predictions = []
    for item in cache:
        boxes = item["boxes"]
        scores = item["scores"]
        labels = item["labels"]
        score_mask = scores >= conf_threshold
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        labels = labels[score_mask]

        keep_all = []
        for cls_idx in range(len(classes)):
            mask = labels == cls_idx
            if mask.any():
                keep = nms(boxes[mask], scores[mask], iou_threshold)
                keep_all.append(torch.where(mask)[0][keep])

        result_boxes = []
        if keep_all:
            keep_idx = torch.cat(keep_all)
            keep_idx = keep_idx[scores[keep_idx].argsort(descending=True)][:max_det]
            for box, score, label in zip(boxes[keep_idx], scores[keep_idx], labels[keep_idx]):
                x1, y1, x2, y2 = box.tolist()
                x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)
                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                    continue
                result_boxes.append(
                    {
                        "class": classes[int(label)],
                        "confidence": round(float(score), 6),
                        "bbox": [x1, y1, x2, y2],
                    }
                )
        predictions.append({"image_id": item["image_id"], "boxes": result_boxes})
    return predictions


def metric_value(metrics):
    if not isinstance(metrics, dict):
        return None
    for key in ("mAP@0.5", "map50", "mAP", "score"):
        if key in metrics:
            return float(metrics[key])
    return None


def main():
    args = parse_args()
    _, items = load_annotation_file(args.val_data)
    conf_values = parse_float_list(args.conf_values)
    iou_values = parse_float_list(args.iou_values)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model, classes = load_model(args)
    cache = cache_raw_predictions(model, classes, items, args.val_image_dir, args, min(conf_values))

    records = []
    best = {"score": -1e9, "conf_threshold": None, "iou_threshold": None, "metrics": None}
    for conf in conf_values:
        for iou in iou_values:
            tag = f"conf{conf:.2f}_iou{iou:.2f}".replace(".", "p")
            pred_path = pred_dir / f"{tag}.json"
            score_path = pred_dir / f"{tag}_score.json"
            predictions = make_predictions(cache, classes, conf, iou, args.max_det)
            save_predictions(predictions, pred_path)
            metrics = run_external_evaluator(args.val_data, pred_path, score_path)
            score = metric_value(metrics)
            record = {"conf_threshold": conf, "iou_threshold": iou, "score": score, "metrics": metrics}
            records.append(record)
            print(f"conf={conf:.2f} iou={iou:.2f} score={score}")
            if score is not None and score > best["score"]:
                best = {"score": score, "conf_threshold": conf, "iou_threshold": iou, "metrics": metrics}

    result = {"best": best, "records": records}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best: conf={best['conf_threshold']} iou={best['iou_threshold']} score={best['score']}")
    print(f"saved tuning report to {args.output}")


if __name__ == "__main__":
    main()
