import json
from pathlib import Path

import torch
from PIL import Image

from .boxes import clip_boxes, nms
from .data import image_to_tensor, resize_letterbox, undo_letterbox


@torch.no_grad()
def decode_outputs(outputs, img_size, strides, num_classes, conf_threshold=0.25):
    all_boxes = []
    all_scores = []
    all_labels = []
    for pred, stride in zip(outputs, strides):
        pred = pred[0]
        _, h, w = pred.shape
        device = pred.device
        # Infer reg_max from output channels: C = 4 * (reg_max + 1) + num_classes.
        reg_channels = pred.shape[0] - num_classes
        reg_max = reg_channels // 4 - 1
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        cx = (xx.float() + 0.5) * stride
        cy = (yy.float() + 0.5) * stride
        project = torch.arange(reg_max + 1, device=device, dtype=pred.dtype)
        dist = pred[:reg_channels].view(4, reg_max + 1, h, w).softmax(dim=1)
        dist = (dist * project.view(1, -1, 1, 1)).sum(dim=1) * stride
        boxes = torch.stack([cx - dist[0], cy - dist[1], cx + dist[2], cy + dist[3]], dim=-1).reshape(-1, 4)
        cls = pred[reg_channels:].sigmoid().permute(1, 2, 0).reshape(-1, num_classes)
        scores, labels = cls.max(dim=1)
        mask = scores >= conf_threshold
        all_boxes.append(boxes[mask])
        all_scores.append(scores[mask])
        all_labels.append(labels[mask])
    if not all_boxes:
        device = outputs[0].device
        return torch.empty((0, 4), device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    boxes = clip_boxes(boxes, img_size, img_size)
    return boxes, scores, labels


def score_mask_for_thresholds(scores, labels, conf_threshold, num_classes):
    if isinstance(conf_threshold, (list, tuple)):
        thresholds = torch.tensor(conf_threshold, dtype=scores.dtype, device=scores.device)
        thresholds = thresholds[:num_classes]
        return scores >= thresholds[labels]
    return scores >= conf_threshold


def min_conf_threshold(conf_threshold):
    if isinstance(conf_threshold, (list, tuple)):
        return min(float(x) for x in conf_threshold)
    return conf_threshold


def _tta_flip_boxes(boxes, img_size):
    """Flip predicted boxes horizontally."""
    if boxes.numel() == 0:
        return boxes
    flipped = boxes.clone()
    old_x1 = flipped[:, 0].clone()
    old_x2 = flipped[:, 2].clone()
    flipped[:, 0] = img_size - old_x2
    flipped[:, 2] = img_size - old_x1
    return flipped


def _tta_scale_boxes(boxes, from_size, to_size):
    """Rescale boxes from one image size to another."""
    if boxes.numel() == 0:
        return boxes
    scale = to_size / from_size
    return boxes * scale


@torch.no_grad()
def predict_image(
    model,
    image_path,
    class_names,
    img_size=960,
    conf_threshold=0.25,
    iou_threshold=0.5,
    max_det=100,
    device="cpu",
    tta=False,
    tta_scales=None,
    imagenet_normalize=False,
):
    image = Image.open(image_path).convert("RGB")
    padded, _, meta = resize_letterbox(image, torch.empty((0, 4)), img_size)
    tensor = image_to_tensor(padded, imagenet_normalize).unsqueeze(0).to(device)
    decode_conf = min_conf_threshold(conf_threshold)
    boxes, scores, labels = decode_outputs(model(tensor), img_size, model.strides, len(class_names), decode_conf)
    if tta:
        # Horizontal flip TTA
        flipped_tensor = torch.flip(tensor, dims=[3])
        flip_boxes, flip_scores, flip_labels = decode_outputs(
            model(flipped_tensor), img_size, model.strides, len(class_names), decode_conf
        )
        if flip_boxes.numel() > 0:
            flip_boxes = _tta_flip_boxes(flip_boxes, img_size)
            boxes = torch.cat([boxes, flip_boxes], dim=0)
            scores = torch.cat([scores, flip_scores], dim=0)
            labels = torch.cat([labels, flip_labels], dim=0)

        # Multi-scale TTA
        scales = tta_scales if tta_scales else [0.85, 1.15]
        for s in scales:
            scaled_size = int(round(img_size * s / 32)) * 32
            if scaled_size == img_size:
                continue
            scaled_padded, _, _ = resize_letterbox(image, torch.empty((0, 4)), scaled_size)
            scaled_tensor = image_to_tensor(scaled_padded, imagenet_normalize).unsqueeze(0).to(device)
            s_boxes, s_scores, s_labels = decode_outputs(
                model(scaled_tensor), scaled_size, model.strides, len(class_names), decode_conf
            )
            if s_boxes.numel() > 0:
                s_boxes = _tta_scale_boxes(s_boxes, scaled_size, img_size)
                boxes = torch.cat([boxes, s_boxes], dim=0)
                scores = torch.cat([scores, s_scores], dim=0)
                labels = torch.cat([labels, s_labels], dim=0)

    score_mask = score_mask_for_thresholds(scores, labels, conf_threshold, len(class_names))
    boxes = boxes[score_mask]
    scores = scores[score_mask]
    labels = labels[score_mask]

    keep_all = []
    for cls_idx in range(len(class_names)):
        mask = labels == cls_idx
        if mask.any():
            keep = nms(boxes[mask], scores[mask], iou_threshold)
            idx = torch.where(mask)[0][keep]
            keep_all.append(idx)
    if keep_all:
        keep_idx = torch.cat(keep_all)
        keep_idx = keep_idx[scores[keep_idx].argsort(descending=True)][:max_det]
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]
    else:
        boxes = boxes[:0]
        scores = scores[:0]
        labels = labels[:0]

    boxes = undo_letterbox(boxes.cpu(), meta)
    boxes = clip_boxes(boxes, meta["orig_w"], meta["orig_h"])
    result_boxes = []
    for box, score, label in zip(boxes, scores.cpu(), labels.cpu()):
        x1, y1, x2, y2 = box.tolist()
        x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue
        result_boxes.append(
            {
                "class": class_names[int(label)],
                "confidence": round(float(score), 6),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return {"image_id": Path(image_path).name, "boxes": result_boxes}


def save_predictions(predictions, output_path):
    Path(output_path).write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
