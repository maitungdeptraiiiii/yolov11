import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models import ConvNeXtYOLO, DarknetYOLOv8
from utils.boxes import clip_boxes, nms
from utils.data import DetectionDataset, detection_collate, image_to_tensor, resize_letterbox, undo_letterbox
from utils.inference import decode_outputs, predict_image, save_predictions
from utils.loss import YOLOLoss
from utils.metrics import run_external_evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom YOLO-style object detector.")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--checkpoint_dir", default="./models/")
    parser.add_argument("--img_size", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.10)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--backbone", choices=["convnext_tiny", "darknet"], default="convnext_tiny")
    parser.add_argument("--no_backbone_pretrained", action="store_true")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    parser.add_argument("--neck_width", type=float, default=1.0, help="Legacy option retained for CLI compatibility.")
    parser.add_argument("--neck_depth", type=float, default=0.34, help="Depth multiplier for YOLO11 C3k2/C2PSA blocks.")
    parser.add_argument("--p2_head", action="store_true", help="Add a stride-4 detection level for small objects.")
    parser.add_argument("--scale", choices=["n", "s", "m", "l", "x", "none"], default="s")
    parser.add_argument("--width", type=float, default=0.50)
    parser.add_argument("--depth", type=float, default=0.34)
    parser.add_argument("--reg_max", type=int, default=20)
    parser.add_argument("--tal_topk", type=int, default=13)
    parser.add_argument("--tal_alpha", type=float, default=0.5)
    parser.add_argument("--tal_beta", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--iou_threshold", type=float, default=0.50)
    parser.add_argument("--max_det", type=int, default=100)
    parser.add_argument("--mosaic_prob", type=float, default=0.15)
    parser.add_argument("--mixup_prob", type=float, default=0.03)
    parser.add_argument("--affine_prob", type=float, default=1.0)
    parser.add_argument("--translate", type=float, default=0.10)
    parser.add_argument("--scale_gain", type=float, default=0.20)
    parser.add_argument("--shear", type=float, default=0.10)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--perspective_prob", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.50)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--multi_scale_sizes", default="", help="Comma-separated train sizes, e.g. 896,960,1024,1088")
    parser.add_argument("--multi_scale_min", type=float, default=0.80)
    parser.add_argument("--multi_scale_max", type=float, default=1.20)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--warmup_epochs", type=float, default=5.0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--close_mosaic_epochs", type=int, default=20)
    parser.add_argument("--close_multiscale_epochs", type=int, default=20)
    parser.add_argument("--imagenet_normalize", action="store_true")
    parser.add_argument("--save_top_k", type=int, default=5)
    parser.add_argument("--no_map50_sweep", action="store_true")
    parser.add_argument("--no_classwise_conf_sweep", action="store_true")
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop after this many validations without mAP50 improvement; 0 disables early stopping.",
    )
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--val_tta", action="store_true")
    parser.add_argument("--val_conf_values", default="0.001,0.003,0.005,0.01,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--val_iou_values", default="0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--no_class_balanced_sampler", action="store_true")
    parser.add_argument("--no_class_loss_weights", action="store_true")
    parser.add_argument("--class_balance_power", type=float, default=0.5)
    parser.add_argument("--max_class_weight", type=float, default=3.0)
    parser.add_argument("--empty_sample_weight", type=float, default=0.25)
    parser.add_argument("--focused_class", default="chair")
    parser.add_argument("--focused_oversample", type=float, default=1.75)
    parser.add_argument("--focused_copypaste_prob", type=float, default=0.25)
    parser.add_argument("--focused_mosaic_donor_prob", type=float, default=0.35)
    parser.add_argument("--focused_min_scale", type=float, default=0.90)
    parser.add_argument("--box_weight", type=float, default=7.5)
    parser.add_argument("--dfl_weight", type=float, default=1.5)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--copypaste_prob", type=float, default=0.15)
    parser.add_argument("--debug_train", action="store_true")
    parser.add_argument("--debug_interval", type=int, default=50)
    parser.add_argument("--verbose", action="store_true", help="Print extra validation/checkpoint details.")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars and keep one summary line per epoch.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_training_device(device):
    text = str(device).strip()
    if "," not in text:
        if text.isdigit():
            return f"cuda:{text}", [int(text)]
        return text, []

    if not torch.cuda.is_available():
        raise RuntimeError(f"Multi-GPU device '{device}' was requested, but CUDA is not available.")

    device_ids = []
    for part in text.split(","):
        part = part.strip()
        if part.startswith("cuda:"):
            part = part.split(":", 1)[1]
        if not part.isdigit():
            raise ValueError(f"Invalid CUDA device id in --device '{device}'. Use values like 0,1 or cuda:0,cuda:1.")
        device_ids.append(int(part))

    available = torch.cuda.device_count()
    missing = [idx for idx in device_ids if idx >= available]
    if missing:
        raise RuntimeError(f"Requested CUDA devices {missing}, but only {available} CUDA device(s) are visible.")
    return f"cuda:{device_ids[0]}", device_ids


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_int_list(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def metric_value(metrics):
    if not isinstance(metrics, dict):
        return None
    for key in ("mAP@0.5", "map50", "mAP50", "mAP", "score"):
        if key in metrics:
            return float(metrics[key])
    return None


class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(unwrap_model(model)).eval()
        self.decay = decay
        self.updates = 0
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / 2000))
        model_state = unwrap_model(model).state_dict()
        for key, ema_value in self.ema.state_dict().items():
            model_value = model_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_value, alpha=1 - decay)
            else:
                ema_value.copy_(model_value)


def make_optimizer(model, lr, weight_decay, backbone_lr_scale=1.0):
    groups = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean_name = name.removeprefix("module.")
        is_backbone = clean_name.startswith("backbone.")
        if param.ndim == 1 or name.endswith(".bias") or ".bn." in name:
            groups["backbone_no_decay" if is_backbone else "head_no_decay"].append(param)
        else:
            groups["backbone_decay" if is_backbone else "head_decay"].append(param)

    param_groups = []
    for key, params in groups.items():
        if not params:
            continue
        is_backbone = key.startswith("backbone")
        is_no_decay = key.endswith("no_decay")
        param_groups.append(
            {
                "params": params,
                "lr": lr * (backbone_lr_scale if is_backbone else 1.0),
                "weight_decay": 0.0 if is_no_decay else weight_decay,
                "name": key,
            }
        )
    return torch.optim.AdamW(param_groups, lr=lr)


def make_scheduler(optimizer, epochs, warmup_epochs, min_lr_ratio):
    def lr_lambda(epoch_idx):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return max(0.01, (epoch_idx + 1) / warmup_epochs)
        denom = max(1.0, epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (epoch_idx - warmup_epochs + 1) / denom))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def set_backbone_trainable(model, trainable):
    raw_model = unwrap_model(model)
    backbone = getattr(raw_model, "backbone", None)
    if backbone is None:
        return False
    for param in backbone.parameters():
        param.requires_grad_(trainable)
    return True


def build_model(args, num_classes):
    if args.backbone == "convnext_tiny":
        return ConvNeXtYOLO(
            num_classes=num_classes,
            reg_max=args.reg_max,
            pretrained_backbone=not args.no_backbone_pretrained,
            neck_width=args.neck_width,
            neck_depth=args.neck_depth,
            p2_head=args.p2_head,
        )
    scale = None if args.scale == "none" else args.scale
    return DarknetYOLOv8(
        num_classes=num_classes,
        width=args.width,
        depth=args.depth,
        reg_max=args.reg_max,
        scale=scale,
    )


def compute_class_counts(dataset):
    counts = torch.zeros(len(dataset.classes), dtype=torch.float32)
    image_counts = torch.zeros(len(dataset.classes), dtype=torch.float32)
    image_class_sets = []
    for item in dataset.items:
        class_ids = set()
        for ann in item["annotations"]:
            class_name = ann.get("class")
            if class_name in dataset.class_to_idx:
                class_ids.add(dataset.class_to_idx[class_name])
                counts[dataset.class_to_idx[class_name]] += 1
        for class_id in class_ids:
            image_counts[class_id] += 1
        image_class_sets.append(class_ids)
    return counts, image_counts, image_class_sets


def make_class_weights(class_counts, power=0.5, max_weight=3.0):
    safe_counts = class_counts.clamp(min=1.0)
    weights = (safe_counts.mean() / safe_counts).pow(power)
    weights = weights / weights.mean().clamp(min=1e-9)
    return weights.clamp(max=max_weight)


def make_balanced_sampler(
    dataset,
    class_weights,
    image_class_sets,
    empty_sample_weight=0.25,
    focused_class=None,
    focused_oversample=1.0,
):
    focused_class_id = dataset.class_to_idx.get(focused_class) if focused_class else None
    sample_weights = []
    for class_ids in image_class_sets:
        if class_ids:
            # Preserve the boost for a minority class when it shares an image
            # with a frequent class such as person.
            weight = float(class_weights[list(class_ids)].max())
            if focused_class_id in class_ids:
                weight *= max(1.0, float(focused_oversample))
            sample_weights.append(weight)
        else:
            sample_weights.append(float(empty_sample_weight))
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def log_class_balance(classes, class_counts, image_counts, class_weights, sampler_weights):
    parts = []
    for name, count, image_count, weight, sampler_weight in zip(
        classes,
        class_counts.tolist(),
        image_counts.tolist(),
        class_weights.tolist(),
        sampler_weights.tolist(),
    ):
        parts.append(
            f"{name}: boxes={int(count)} images={int(image_count)} "
            f"loss_weight={weight:.3f} sampler_weight={sampler_weight:.3f}"
        )
    print("class balance:", "; ".join(parts), flush=True)


def should_validate(epoch, epochs, val_interval):
    interval = max(1, int(val_interval))
    return epoch == 1 or epoch == epochs or epoch % interval == 0


def should_debug_batch(args, batch_idx):
    if not args.debug_train:
        return False
    interval = max(0, args.debug_interval)
    return interval > 0 and batch_idx % interval == 0


def debug_train_log(args, message):
    if args.debug_train:
        print(f"[debug] {message}", flush=True)


def apply_multiscale(images, targets, base_size, sizes=None, min_ratio=0.80, max_ratio=1.20):
    if sizes:
        valid_sizes = [size for size in sizes if size > 0 and size % 32 == 0]
        if not valid_sizes:
            raise ValueError("--multi_scale_sizes values must be positive multiples of 32.")
        size = random.choice(valid_sizes)
    else:
        min_size = max(32, int(base_size * min_ratio))
        max_size = max(min_size, int(base_size * max_ratio))
        size = random.randrange(min_size // 32, max_size // 32 + 1) * 32
    if size == images.shape[-1]:
        return images, targets, size
    scale = size / images.shape[-1]
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    for target in targets:
        if target["boxes"].numel() > 0:
            target["boxes"] = target["boxes"] * scale
    return images, targets, size


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args):
    model.train()
    running = 0.0
    pbar = tqdm(total=len(loader), desc=f"epoch {epoch}", leave=False, disable=args.debug_train or args.no_progress)
    use_amp = device.startswith("cuda")
    iterator = iter(loader)
    epoch_start = time.perf_counter()
    last_logs = {}
    last_train_size = args.img_size
    total_boxes = 0
    multi_scale_sizes = parse_int_list(args.multi_scale_sizes)
    use_multiscale = args.multi_scale and not (
        args.close_multiscale_epochs > 0 and epoch > args.epochs - args.close_multiscale_epochs
    )
    for batch_idx in range(1, len(loader) + 1):
        images, targets = next(iterator)
        total_boxes += sum(int(t["boxes"].shape[0]) for t in targets)
        if use_multiscale:
            images, targets, train_size = apply_multiscale(
                images,
                targets,
                args.img_size,
                sizes=multi_scale_sizes,
                min_ratio=args.multi_scale_min,
                max_ratio=args.multi_scale_max,
            )
        else:
            train_size = args.img_size
        last_train_size = train_size
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss, logs = criterion(outputs, targets)
        last_logs = logs
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        if hasattr(args, "ema") and args.ema is not None:
            args.ema.update(model)
        running += float(loss.detach().cpu())
        if args.debug_train and (should_debug_batch(args, batch_idx) or batch_idx == len(loader)):
            percent = batch_idx / max(1, len(loader)) * 100
            print(
                f"\repoch={epoch} progress={percent:6.2f}% "
                f"batch={batch_idx}/{len(loader)} loss={logs['loss']:.4f} "
                f"box={logs['box']:.3f} cls={logs['cls']:.3f} dfl={logs.get('dfl', 0.0):.3f} "
                f"size={train_size}",
                end="",
                flush=True,
            )
        pbar.set_postfix(
            loss=f"{logs['loss']:.3f}",
            box=f"{logs['box']:.3f}",
            dfl=f"{logs.get('dfl', 0.0):.3f}",
            cls=f"{logs['cls']:.3f}",
            size=train_size,
        )
        pbar.update(1)
    pbar.close()
    if args.debug_train:
        print(flush=True)
        epoch_dt = time.perf_counter() - epoch_start
        mem_text = ""
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            mem_text = f" max_mem={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
        debug_train_log(
            args,
            f"epoch={epoch} summary batches={len(loader)} "
            f"avg_loss={running / max(1, len(loader)):.4f} "
            f"last_box={last_logs.get('box', 0.0):.3f} "
            f"last_cls={last_logs.get('cls', 0.0):.3f} "
            f"last_dfl={last_logs.get('dfl', 0.0):.3f} "
            f"last_size={last_train_size} boxes={total_boxes} "
            f"dt={epoch_dt:.2f}s{mem_text}",
        )
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, args):
    model.eval()
    running = 0.0
    count = 0
    use_amp = device.startswith("cuda")
    for images, targets in tqdm(loader, desc="val loss", leave=False, disable=args.no_progress):
        images = images.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, _ = criterion(model(images), targets)
        running += float(loss.detach().cpu())
        count += 1
    return running / max(1, count)


@torch.no_grad()
def cache_validation_predictions(model, dataset, args, conf_threshold):
    model.eval()
    raw_model = unwrap_model(model)
    from utils.inference import _tta_flip_boxes, _tta_scale_boxes
    predictions = []
    for item in tqdm(dataset.items, desc="val predict", leave=False, disable=args.no_progress):
        info = item["info"]
        image_path = Path(args.val_image_dir) / Path(info["file_name"]).name
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        padded, _, meta = resize_letterbox(image, torch.empty((0, 4)), args.img_size)
        tensor = image_to_tensor(padded, args.imagenet_normalize).unsqueeze(0).to(args.device)
        boxes, scores, labels = decode_outputs(
            model(tensor), args.img_size, raw_model.strides, len(dataset.classes), conf_threshold
        )
        if args.val_tta:
            # Horizontal flip TTA
            flipped_tensor = torch.flip(tensor, dims=[3])
            flip_boxes, flip_scores, flip_labels = decode_outputs(
                model(flipped_tensor), args.img_size, raw_model.strides, len(dataset.classes), conf_threshold
            )
            if flip_boxes.numel() > 0:
                flip_boxes = _tta_flip_boxes(flip_boxes, args.img_size)
                boxes = torch.cat([boxes, flip_boxes], dim=0)
                scores = torch.cat([scores, flip_scores], dim=0)
                labels = torch.cat([labels, flip_labels], dim=0)
            # Multi-scale TTA
            for s in [0.85, 1.15]:
                scaled_size = int(round(args.img_size * s / 32)) * 32
                if scaled_size == args.img_size:
                    continue
                scaled_padded, _, _ = resize_letterbox(image, torch.empty((0, 4)), scaled_size)
                scaled_tensor = image_to_tensor(scaled_padded, args.imagenet_normalize).unsqueeze(0).to(args.device)
                s_boxes, s_scores, s_labels = decode_outputs(
                    model(scaled_tensor), scaled_size, raw_model.strides, len(dataset.classes), conf_threshold
                )
                if s_boxes.numel() > 0:
                    s_boxes = _tta_scale_boxes(s_boxes, scaled_size, args.img_size)
                    boxes = torch.cat([boxes, s_boxes], dim=0)
                    scores = torch.cat([scores, s_scores], dim=0)
                    labels = torch.cat([labels, s_labels], dim=0)
        boxes = undo_letterbox(boxes.cpu(), meta)
        boxes = clip_boxes(boxes, meta["orig_w"], meta["orig_h"])
        predictions.append({"image_id": Path(image_path).name, "boxes": boxes.cpu(), "scores": scores.cpu(), "labels": labels.cpu()})
    return predictions


def score_mask_for_thresholds(scores, labels, conf_threshold, num_classes):
    if isinstance(conf_threshold, (list, tuple)):
        thresholds = torch.tensor(conf_threshold, dtype=scores.dtype, device=scores.device)
        thresholds = thresholds[:num_classes]
        return scores >= thresholds[labels]
    return scores >= conf_threshold


def materialize_predictions(cache, classes, conf_threshold, iou_threshold, max_det):
    predictions = []
    for item in cache:
        boxes = item["boxes"]
        scores = item["scores"]
        labels = item["labels"]
        score_mask = score_mask_for_thresholds(scores, labels, conf_threshold, len(classes))
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


def evaluate_cached_predictions(cache, classes, conf_threshold, iou_threshold, max_det, output_path, score_path, val_data):
    predictions = materialize_predictions(cache, classes, conf_threshold, iou_threshold, max_det)
    save_predictions(predictions, output_path)
    metrics = run_external_evaluator(val_data, output_path, score_path)
    return metric_value(metrics), metrics


def refine_class_conf_thresholds(cache, dataset, args, base_conf, iou_threshold, output_path, score_path):
    class_thresholds = [float(base_conf)] * len(dataset.classes)
    best_score, best_metrics = evaluate_cached_predictions(
        cache, dataset.classes, class_thresholds, iou_threshold, args.max_det, output_path, score_path, args.val_data
    )
    records = []
    if best_score is None:
        return class_thresholds, best_score, best_metrics, records

    conf_values = parse_float_list(args.val_conf_values)
    for cls_idx, cls_name in enumerate(dataset.classes):
        class_best = {"score": best_score, "threshold": class_thresholds[cls_idx], "metrics": best_metrics}
        for conf in conf_values:
            trial = list(class_thresholds)
            trial[cls_idx] = float(conf)
            score, metrics = evaluate_cached_predictions(
                cache, dataset.classes, trial, iou_threshold, args.max_det, output_path, score_path, args.val_data
            )
            records.append(
                {
                    "class": cls_name,
                    "class_index": cls_idx,
                    "conf_threshold": conf,
                    "iou_threshold": iou_threshold,
                    "score": score,
                    "metrics": metrics,
                }
            )
            if score is not None and score > class_best["score"]:
                class_best = {"score": score, "threshold": float(conf), "metrics": metrics}
        class_thresholds[cls_idx] = class_best["threshold"]
        best_score = class_best["score"]
        best_metrics = class_best["metrics"]
    return class_thresholds, best_score, best_metrics, records


@torch.no_grad()
def predict_validation(model, dataset, args, epoch):
    model.eval()
    ckpt_dir = Path(args.checkpoint_dir)
    if args.no_map50_sweep:
        output_path = ckpt_dir / f"val_predictions_epoch{epoch}.json"
        predictions = []
        for item in tqdm(dataset.items, desc="val predict", leave=False, disable=args.no_progress):
            info = item["info"]
            image_path = Path(args.val_image_dir) / Path(info["file_name"]).name
            predictions.append(
                predict_image(
                    model,
                    image_path,
                    dataset.classes,
                    img_size=args.img_size,
                    conf_threshold=args.conf_threshold,
                    iou_threshold=args.iou_threshold,
                    max_det=args.max_det,
                    device=args.device,
                    tta=args.val_tta,
                    imagenet_normalize=args.imagenet_normalize,
                )
            )
        save_predictions(predictions, output_path)
        score_path = ckpt_dir / f"val_score_epoch{epoch}.json"
        metrics = run_external_evaluator(args.val_data, output_path, score_path)
        score = metric_value(metrics)
        if score is not None:
            return score, metrics, {"conf_threshold": args.conf_threshold, "iou_threshold": args.iou_threshold}
        print("warning: validation evaluator did not return a usable score; using 0.0 for this epoch")
        return 0.0, metrics, {"conf_threshold": args.conf_threshold, "iou_threshold": args.iou_threshold}

    conf_values = parse_float_list(args.val_conf_values)
    iou_values = parse_float_list(args.val_iou_values)
    cache = cache_validation_predictions(model, dataset, args, min(conf_values))
    best = {"score": -1e9, "metrics": None, "conf_threshold": args.conf_threshold, "iou_threshold": args.iou_threshold}
    records = []
    output_path = ckpt_dir / f"val_predictions_epoch{epoch}_sweep_tmp.json"
    score_path = ckpt_dir / f"val_score_epoch{epoch}_sweep_tmp.json"
    for conf in conf_values:
        for iou in iou_values:
            score, metrics = evaluate_cached_predictions(
                cache, dataset.classes, conf, iou, args.max_det, output_path, score_path, args.val_data
            )
            records.append({"conf_threshold": conf, "iou_threshold": iou, "score": score, "metrics": metrics})
            if score is not None and score > best["score"]:
                best = {"score": score, "metrics": metrics, "conf_threshold": conf, "iou_threshold": iou}
    classwise_records = []
    if best["metrics"] is not None and not args.no_classwise_conf_sweep:
        class_thresholds, classwise_score, classwise_metrics, classwise_records = refine_class_conf_thresholds(
            cache,
            dataset,
            args,
            best["conf_threshold"],
            best["iou_threshold"],
            output_path,
            score_path,
        )
        if classwise_score is not None and classwise_score >= best["score"]:
            best = {
                "score": classwise_score,
                "metrics": classwise_metrics,
                "conf_threshold": min(class_thresholds),
                "iou_threshold": best["iou_threshold"],
                "class_conf_thresholds": class_thresholds,
                "class_conf_thresholds_by_name": {
                    name: threshold for name, threshold in zip(dataset.classes, class_thresholds)
                },
            }
    (ckpt_dir / f"val_sweep_epoch{epoch}.json").write_text(
        json.dumps({"best": best, "records": records, "classwise_records": classwise_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if best["metrics"] is None:
        print("warning: validation evaluator did not return a usable score; using 0.0 for this epoch")
        return 0.0, None, {"conf_threshold": args.conf_threshold, "iou_threshold": args.iou_threshold}
    if args.verbose:
        print(
            f"best val thresholds: conf={best['conf_threshold']:.3f} "
            f"iou={best['iou_threshold']:.2f} score={best['score']:.4f}"
        )
        if "class_conf_thresholds_by_name" in best:
            print(
                "best class conf thresholds: "
                + ", ".join(f"{name}={threshold:.3f}" for name, threshold in best["class_conf_thresholds_by_name"].items())
            )
    return float(best["score"]), best["metrics"], {
        "conf_threshold": best["conf_threshold"],
        "iou_threshold": best["iou_threshold"],
        "class_conf_thresholds": best.get("class_conf_thresholds"),
        "class_conf_thresholds_by_name": best.get("class_conf_thresholds_by_name"),
    }


def save_checkpoint(path, model, epoch, class_names, args, score, thresholds):
    raw_model = unwrap_model(model)
    ckpt = {
        "model": raw_model.state_dict(),
        "epoch": epoch,
        "classes": class_names,
        "args": dict(vars(args)),
        "architecture": getattr(raw_model, "architecture", raw_model.__class__.__name__),
        "score": score,
        "best_thresholds": thresholds,
    }
    ckpt["args"].pop("ema", None)
    torch.save(ckpt, path)


def update_top_checkpoints(top_checkpoints, path, score, keep):
    if keep <= 0 or score is None:
        return top_checkpoints
    top_checkpoints.append((float(score), Path(path)))
    top_checkpoints.sort(key=lambda item: item[0], reverse=True)
    while len(top_checkpoints) > keep:
        _, remove_path = top_checkpoints.pop()
        if remove_path.exists():
            remove_path.unlink()
    return top_checkpoints


def main():
    args = parse_args()
    set_seed(args.seed)
    args.device, device_ids = parse_training_device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DetectionDataset(
        args.train_data,
        args.image_dir,
        img_size=args.img_size,
        augment=True,
        mosaic_prob=args.mosaic_prob,
        mixup_prob=args.mixup_prob,
        affine_prob=args.affine_prob,
        translate=args.translate,
        scale_gain=args.scale_gain,
        shear=args.shear,
        perspective=args.perspective,
        perspective_prob=args.perspective_prob,
        fliplr=args.fliplr,
        flipud=args.flipud,
        copypaste_prob=args.copypaste_prob,
        focused_class=args.focused_class,
        focused_copypaste_prob=args.focused_copypaste_prob,
        focused_mosaic_donor_prob=args.focused_mosaic_donor_prob,
        focused_min_scale=args.focused_min_scale,
        imagenet_normalize=args.imagenet_normalize,
    )
    val_ds = DetectionDataset(
        args.val_data,
        args.val_image_dir,
        img_size=args.img_size,
        augment=False,
        imagenet_normalize=args.imagenet_normalize,
    )
    class_counts, image_counts, image_class_sets = compute_class_counts(train_ds)
    class_weights = make_class_weights(class_counts, args.class_balance_power, args.max_class_weight)
    sampler_class_weights = make_class_weights(image_counts, args.class_balance_power, args.max_class_weight)
    log_class_balance(train_ds.classes, class_counts, image_counts, class_weights, sampler_class_weights)
    sampler = None
    if not args.no_class_balanced_sampler:
        sampler = make_balanced_sampler(
            train_ds,
            sampler_class_weights,
            image_class_sets,
            args.empty_sample_weight,
            focused_class=args.focused_class,
            focused_oversample=args.focused_oversample,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=args.device.startswith("cuda"),
    )

    model = build_model(args, len(train_ds.classes)).to(args.device)
    if len(device_ids) > 1:
        print(f"Using DataParallel on CUDA devices: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    criterion = YOLOLoss(
        num_classes=len(train_ds.classes),
        img_size=args.img_size,
        strides=unwrap_model(model).strides,
        reg_max=args.reg_max,
        box_weight=args.box_weight,
        dfl_weight=args.dfl_weight,
        cls_weight=args.cls_weight,
        tal_topk=args.tal_topk,
        tal_alpha=args.tal_alpha,
        tal_beta=args.tal_beta,
        class_weights=None if args.no_class_loss_weights else class_weights,
    )
    optimizer = make_optimizer(model, args.lr, args.weight_decay, args.backbone_lr_scale)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    args.ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    best_score = -1e9
    best_thresholds = {"conf_threshold": args.conf_threshold, "iou_threshold": args.iou_threshold}
    history = []
    backbone_frozen = None
    top_checkpoints = []
    validations_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        should_freeze_backbone = args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs
        if should_freeze_backbone != backbone_frozen:
            if set_backbone_trainable(model, not should_freeze_backbone):
                state = "frozen" if should_freeze_backbone else "unfrozen"
                if args.verbose:
                    print(f"backbone {state} at epoch {epoch}", flush=True)
            backbone_frozen = should_freeze_backbone
        if args.close_mosaic_epochs > 0 and epoch > args.epochs - args.close_mosaic_epochs:
            train_ds.mosaic_prob = 0.0
            train_ds.mixup_prob = 0.0
            train_ds.copypaste_prob = 0.0
            train_ds.focused_copypaste_prob = 0.0
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, args.device, epoch, args)
        scheduler.step()
        eval_model = args.ema.ema if args.ema is not None else model
        lr_now = optimizer.param_groups[0]["lr"]
        do_validate = should_validate(epoch, args.epochs, args.val_interval)
        score = None
        metrics = None
        val_loss = None
        thresholds = best_thresholds
        if do_validate:
            val_loss = evaluate_loss(eval_model, val_loader, criterion, args.device, args)
            score, metrics, thresholds = predict_validation(eval_model, val_ds, args, epoch)
            best_thresholds = thresholds
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "score": score,
                "metrics": metrics,
                "thresholds": thresholds,
                "lr": lr_now,
                "validated": do_validate,
            }
        )
        if do_validate:
            conf_text = thresholds.get("conf_threshold", args.conf_threshold)
            if isinstance(conf_text, (list, tuple)):
                conf_text = min(float(x) for x in conf_text)
            iou_text = thresholds.get("iou_threshold", args.iou_threshold)
            map5095 = metrics.get("mAP@0.5:0.95") if isinstance(metrics, dict) else None
            map5095_text = f" val_map50_95={float(map5095):.4f}" if map5095 is not None else ""
            print(
                f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_map50={score:.4f}{map5095_text} lr={lr_now:.6g} "
                f"conf={float(conf_text):.3f} iou={float(iou_text):.2f}"
            )
        else:
            print(f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss=skip val_map50=skip lr={lr_now:.6g}")
        print(f"saving checkpoint: {ckpt_dir / 'last.pth'}", flush=True)
        save_checkpoint(ckpt_dir / "last.pth", eval_model, epoch, train_ds.classes, args, score, thresholds)
        improved = do_validate and score > best_score + max(0.0, args.early_stopping_min_delta)
        if improved:
            best_score = score
            validations_without_improvement = 0
            print(f"saving best checkpoint: {ckpt_dir / 'best.pth'}", flush=True)
            save_checkpoint(ckpt_dir / "best.pth", eval_model, epoch, train_ds.classes, args, score, thresholds)
        elif do_validate:
            validations_without_improvement += 1
        if do_validate and args.save_top_k > 0:
            top_path = ckpt_dir / f"top_epoch{epoch:03d}_map{score:.6f}.pth"
            print(f"saving top checkpoint: {top_path}", flush=True)
            save_checkpoint(top_path, eval_model, epoch, train_ds.classes, args, score, thresholds)
            top_checkpoints = update_top_checkpoints(top_checkpoints, top_path, score, args.save_top_k)
        (ckpt_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if (
            do_validate
            and args.early_stopping_patience > 0
            and validations_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"early stopping at epoch {epoch}: mAP50 did not improve by at least "
                f"{args.early_stopping_min_delta:g} for {validations_without_improvement} validations; "
                f"best={best_score:.6f}"
            )
            break


if __name__ == "__main__":
    main()
