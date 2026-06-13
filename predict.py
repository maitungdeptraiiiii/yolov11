import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from models import ConvNeXtYOLO, DarknetYOLOv8
from utils.data import DEFAULT_CLASSES
from utils.inference import predict_image, save_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the custom YOLO-style object detector.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="./models/best.pth")
    parser.add_argument("--img_size", type=int, default=960)
    parser.add_argument("--conf_threshold", type=float, default=None)
    parser.add_argument("--iou_threshold", type=float, default=None)
    parser.add_argument("--max_det", type=int, default=100)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta_scales", default=None, help="Comma-separated TTA scales, e.g. 0.85,1.15")
    parser.add_argument("--imagenet_normalize", action="store_true", help="Force ImageNet normalization at inference.")
    parser.add_argument("--backbone", choices=["convnext_tiny", "darknet"], default="convnext_tiny")
    parser.add_argument("--no_backbone_pretrained", action="store_true")
    parser.add_argument("--neck_width", type=float, default=1.0, help="Legacy option retained for CLI compatibility.")
    parser.add_argument("--neck_depth", type=float, default=0.34, help="Depth multiplier for YOLO11 C3k2/C2PSA blocks.")
    parser.add_argument("--p2_head", action="store_true")
    parser.add_argument("--scale", choices=["n", "s", "m", "l", "x", "none"], default="s")
    parser.add_argument("--width", type=float, default=0.50)
    parser.add_argument("--depth", type=float, default=0.34)
    parser.add_argument("--reg_max", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(args):
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    classes = checkpoint.get("classes", DEFAULT_CLASSES)
    saved_args = checkpoint.get("args", {})
    architecture = checkpoint.get("architecture", "legacy")
    thresholds = checkpoint.get("best_thresholds", {})
    imagenet_normalize = bool(saved_args.get("imagenet_normalize", False) or args.imagenet_normalize)
    backbone = saved_args.get("backbone")
    if backbone is None:
        state_keys = checkpoint.get("model", {}).keys()
        backbone = "convnext_tiny" if any(k.startswith("backbone.features.") for k in state_keys) else "darknet"
    width = float(saved_args.get("width", args.width))
    depth = float(saved_args.get("depth", args.depth))
    reg_max = int(saved_args.get("reg_max", args.reg_max))
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
    conf_threshold = args.conf_threshold
    iou_threshold = args.iou_threshold
    if conf_threshold is None:
        class_thresholds = thresholds.get("class_conf_thresholds")
        if class_thresholds is not None:
            conf_threshold = [float(x) for x in class_thresholds]
        else:
            conf_threshold = float(thresholds.get("conf_threshold", 0.25))
    if iou_threshold is None:
        iou_threshold = float(thresholds.get("iou_threshold", 0.50))
    return model, classes, conf_threshold, iou_threshold, imagenet_normalize


def main():
    args = parse_args()
    model, classes, conf_threshold, iou_threshold, imagenet_normalize = load_model(args)
    image_dir = Path(args.image_dir)
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    )
    tta_scales = None
    if args.tta_scales:
        tta_scales = [float(x.strip()) for x in args.tta_scales.split(",") if x.strip()]
    predictions = []
    for image_path in tqdm(image_paths, desc="predict"):
        predictions.append(
            predict_image(
                model,
                image_path,
                classes,
                img_size=args.img_size,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_det=args.max_det,
                device=args.device,
                tta=args.tta,
                tta_scales=tta_scales,
                imagenet_normalize=imagenet_normalize,
            )
        )
    save_predictions(predictions, args.output)


if __name__ == "__main__":
    main()
