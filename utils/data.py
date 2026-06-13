import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


DEFAULT_CLASSES = ["person", "car", "dog", "cat", "chair"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_to_tensor(image, imagenet_normalize=False):
    tensor = TF.to_tensor(image)
    if imagenet_normalize:
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor


def read_classes(path=None):
    if path and Path(path).exists():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "classes" in data:
            return data["classes"]
    return DEFAULT_CLASSES


def load_annotation_file(annotation_path):
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    classes = data.get("classes", DEFAULT_CLASSES)
    by_image = {img["id"]: {"info": img, "annotations": []} for img in data.get("images", [])}
    for ann in data.get("annotations", []):
        if ann["image_id"] in by_image:
            by_image[ann["image_id"]]["annotations"].append(ann)
    return classes, list(by_image.values())


def resize_letterbox(image, boxes, size):
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    if boxes.numel() > 0:
        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
    meta = {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "orig_w": w, "orig_h": h}
    return canvas, boxes, meta


def undo_letterbox(boxes, meta):
    boxes = boxes.clone()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta["pad_x"]) / meta["scale"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta["pad_y"]) / meta["scale"]
    return boxes


def clip_and_filter_boxes(boxes, labels, width, height, min_size=2.0, min_area=8.0):
    if boxes.numel() == 0:
        return boxes.reshape(0, 4), labels
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    keep = (wh[:, 0] >= min_size) & (wh[:, 1] >= min_size) & ((wh[:, 0] * wh[:, 1]) >= min_area)
    return boxes[keep], labels[keep]


def paste_clipped(canvas, image, offset_x, offset_y):
    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    dst_x2 = min(canvas.width, offset_x + image.width)
    dst_y2 = min(canvas.height, offset_y + image.height)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return
    src_x1 = dst_x1 - offset_x
    src_y1 = dst_y1 - offset_y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    canvas.paste(image.crop((src_x1, src_y1, src_x2, src_y2)), (dst_x1, dst_y1))


def boxes_to_corners(boxes):
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    return torch.stack(
        [
            torch.stack([x1, y1], dim=1),
            torch.stack([x2, y1], dim=1),
            torch.stack([x2, y2], dim=1),
            torch.stack([x1, y2], dim=1),
        ],
        dim=1,
    )


def corners_to_boxes(corners):
    x = corners[:, :, 0]
    y = corners[:, :, 1]
    return torch.stack([x.min(dim=1).values, y.min(dim=1).values, x.max(dim=1).values, y.max(dim=1).values], dim=1)


def transform_boxes_affine(boxes, matrix):
    if boxes.numel() == 0:
        return boxes
    corners = boxes_to_corners(boxes).reshape(-1, 2)
    ones = torch.ones((corners.shape[0], 1), dtype=corners.dtype, device=corners.device)
    points = torch.cat([corners, ones], dim=1)
    transformed = points @ matrix.to(dtype=corners.dtype, device=corners.device).t()
    return corners_to_boxes(transformed[:, :2].reshape(-1, 4, 2))


def homography_from_points(src_points, dst_points):
    rows = []
    values = []
    for (x, y), (u, v) in zip(src_points, dst_points):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.extend([u, v])
    a = torch.tensor(rows, dtype=torch.float32)
    b = torch.tensor(values, dtype=torch.float32)
    h = torch.linalg.solve(a, b)
    return torch.tensor(
        [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]],
        dtype=torch.float32,
    )


def transform_boxes_perspective(boxes, homography):
    if boxes.numel() == 0:
        return boxes
    corners = boxes_to_corners(boxes).reshape(-1, 2)
    ones = torch.ones((corners.shape[0], 1), dtype=corners.dtype, device=corners.device)
    points = torch.cat([corners, ones], dim=1)
    transformed = points @ homography.to(dtype=corners.dtype, device=corners.device).t()
    transformed_xy = transformed[:, :2] / transformed[:, 2:3].clamp(min=1e-6)
    return corners_to_boxes(transformed_xy.reshape(-1, 4, 2))


class DetectionDataset(Dataset):
    def __init__(
        self,
        annotation_path,
        image_dir=None,
        img_size=640,
        augment=False,
        mosaic_prob=0.45,
        mixup_prob=0.10,
        affine_prob=0.50,
        translate=0.10,
        scale_gain=0.20,
        shear=0.20,
        perspective=0.0,
        perspective_prob=0.0,
        fliplr=0.50,
        flipud=0.0,
        copypaste_prob=0.0,
        focused_class=None,
        focused_copypaste_prob=0.0,
        focused_mosaic_donor_prob=0.0,
        focused_min_scale=0.90,
        imagenet_normalize=False,
    ):
        self.annotation_path = Path(annotation_path)
        self.root = self.annotation_path.parent.parent
        self.classes, self.items = load_annotation_file(annotation_path)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.image_dir = Path(image_dir) if image_dir else None
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.affine_prob = affine_prob
        self.translate = translate
        self.scale_gain = scale_gain
        self.shear = shear
        self.perspective = perspective
        self.perspective_prob = perspective_prob
        self.fliplr = fliplr
        self.flipud = flipud
        self.copypaste_prob = copypaste_prob
        self.focused_class = focused_class
        self.focused_class_id = self.class_to_idx.get(focused_class) if focused_class else None
        self.focused_copypaste_prob = focused_copypaste_prob
        self.focused_mosaic_donor_prob = focused_mosaic_donor_prob
        self.focused_min_scale = focused_min_scale
        self.focused_indices = []
        if self.focused_class_id is not None:
            self.focused_indices = [
                idx
                for idx, item in enumerate(self.items)
                if any(ann.get("class") == focused_class for ann in item["annotations"])
            ]
        self.imagenet_normalize = imagenet_normalize

    def _sample_index(self, prefer_focused=False):
        if prefer_focused and self.focused_indices:
            return random.choice(self.focused_indices)
        return random.randrange(len(self.items))

    def __len__(self):
        return len(self.items)

    def _image_path(self, info):
        if self.image_dir is not None:
            return self.image_dir / Path(info["file_name"]).name
        return self.root / info["file_name"]

    def _load_item(self, index):
        item = self.items[index]
        info = item["info"]
        image = Image.open(self._image_path(info)).convert("RGB")
        boxes = []
        labels = []
        for ann in item["annotations"]:
            x1, y1, x2, y2 = ann["bbox"]
            if x2 > x1 and y2 > y1 and ann["class"] in self.class_to_idx:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_to_idx[ann["class"]])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.long)
        return image, boxes, labels, info

    def _mosaic(self, index):
        size = self.img_size
        # Random mosaic center in [0.25*size, 0.75*size] for diversity
        cx = random.randint(int(size * 0.25), int(size * 0.75))
        cy = random.randint(int(size * 0.25), int(size * 0.75))
        indices = [index]
        for _ in range(3):
            prefer_focused = random.random() < self.focused_mosaic_donor_prob
            indices.append(self._sample_index(prefer_focused=prefer_focused))
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        all_boxes = []
        all_labels = []
        # (x1, y1, x2, y2) of each quadrant: top-left, top-right, bottom-left, bottom-right
        quads = [(0, 0, cx, cy), (cx, 0, size, cy), (0, cy, cx, size), (cx, cy, size, size)]
        main_info = None

        for tile_idx, (sample_idx, (qx1, qy1, qx2, qy2)) in enumerate(zip(indices, quads)):
            image, boxes, labels, info = self._load_item(sample_idx)
            if tile_idx == 0:
                main_info = info
            tw, th = qx2 - qx1, qy2 - qy1
            tile_size = max(tw, th)
            image, boxes, _ = resize_letterbox(image, boxes, tile_size)
            # Center-crop to exact quadrant dimensions
            ox_crop = (tile_size - tw) // 2
            oy_crop = (tile_size - th) // 2
            image = image.crop((ox_crop, oy_crop, ox_crop + tw, oy_crop + th))
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] -= ox_crop
                boxes[:, [1, 3]] -= oy_crop
                boxes, labels = clip_and_filter_boxes(boxes, labels, tw, th)
            canvas.paste(image, (qx1, qy1))
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] += qx1
                boxes[:, [1, 3]] += qy1
                all_boxes.append(boxes)
                all_labels.append(labels)

        if all_boxes:
            boxes = torch.cat(all_boxes, dim=0)
            labels = torch.cat(all_labels, dim=0)
            boxes, labels = clip_and_filter_boxes(boxes, labels, size, size)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)
        return canvas, boxes, labels, main_info

    def _random_affine(self, image, boxes, labels):
        size = self.img_size
        min_scale = max(0.1, 1.0 - self.scale_gain)
        if self.focused_class_id is not None and (labels == self.focused_class_id).any():
            min_scale = max(min_scale, self.focused_min_scale)
        scale = random.uniform(min_scale, 1.0 + self.scale_gain)
        max_shift = int(size * self.translate)
        translate = (random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift))
        shear = [random.uniform(-self.shear, self.shear), random.uniform(-self.shear, self.shear)]
        center = [size * 0.5, size * 0.5]
        image = TF.affine(
            image,
            angle=0.0,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=(114, 114, 114),
        )
        if boxes.numel() > 0:
            inverse = TF._get_inverse_affine_matrix(center, 0.0, list(translate), scale, shear)
            inverse = torch.tensor(
                [[inverse[0], inverse[1], inverse[2]], [inverse[3], inverse[4], inverse[5]], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
            forward = torch.linalg.inv(inverse)
            boxes = transform_boxes_affine(boxes, forward)
            boxes, labels = clip_and_filter_boxes(boxes, labels, size, size)
        return image, boxes, labels

    def _random_perspective(self, image, boxes, labels):
        size = self.img_size
        max_jitter = int(round(size * self.perspective))
        if max_jitter <= 0:
            return image, boxes, labels
        src = [(0, 0), (size, 0), (size, size), (0, size)]
        dst = [
            (random.randint(0, max_jitter), random.randint(0, max_jitter)),
            (size - random.randint(0, max_jitter), random.randint(0, max_jitter)),
            (size - random.randint(0, max_jitter), size - random.randint(0, max_jitter)),
            (random.randint(0, max_jitter), size - random.randint(0, max_jitter)),
        ]
        image = TF.perspective(
            image,
            startpoints=src,
            endpoints=dst,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=(114, 114, 114),
        )
        if boxes.numel() > 0:
            homography = homography_from_points(src, dst)
            boxes = transform_boxes_perspective(boxes, homography)
            boxes, labels = clip_and_filter_boxes(boxes, labels, size, size)
        return image, boxes, labels

    def _mixup(self, image, boxes, labels):
        idx = random.randrange(len(self.items))
        if random.random() < self.mosaic_prob:
            image2, boxes2, labels2, _ = self._mosaic(idx)
        else:
            image2, boxes2, labels2, _ = self._load_item(idx)
            image2, boxes2, _ = resize_letterbox(image2, boxes2, self.img_size)
        alpha = random.uniform(0.35, 0.65)
        mixed = Image.blend(image, image2, alpha)
        if boxes2.numel() > 0:
            boxes = torch.cat([boxes, boxes2], dim=0) if boxes.numel() > 0 else boxes2
            labels = torch.cat([labels, labels2], dim=0) if labels.numel() > 0 else labels2
        return mixed, boxes, labels

    def _copypaste(self, image, boxes, labels, focused_only=False):
        """Copy object regions from a donor image and paste at random positions."""
        idx = self._sample_index(prefer_focused=focused_only)
        donor_img, donor_boxes, donor_labels, _ = self._load_item(idx)
        if donor_boxes.numel() == 0:
            return image, boxes, labels
        donor_img, donor_boxes, _ = resize_letterbox(donor_img, donor_boxes, self.img_size)
        candidate_indices = list(range(donor_boxes.shape[0]))
        if focused_only and self.focused_class_id is not None:
            candidate_indices = torch.where(donor_labels == self.focused_class_id)[0].tolist()
        if not candidate_indices:
            return image, boxes, labels
        n_copy = min(random.randint(1, 2 if focused_only else 3), len(candidate_indices))
        indices = random.sample(candidate_indices, n_copy)
        for i in indices:
            bx1, by1, bx2, by2 = donor_boxes[i].int().tolist()
            bw, bh = bx2 - bx1, by2 - by1
            if bw < 4 or bh < 4:
                continue
            crop = donor_img.crop((bx1, by1, bx2, by2))
            max_x = image.width - bw
            max_y = image.height - bh
            if max_x <= 0 or max_y <= 0:
                continue
            placement = None
            for _ in range(10):
                px = random.randint(0, max_x)
                py = random.randint(0, max_y)
                candidate = torch.tensor([[px, py, px + bw, py + bh]], dtype=torch.float32)
                if boxes.numel() == 0:
                    placement = (px, py, candidate)
                    break
                inter_x1 = torch.maximum(boxes[:, 0], candidate[0, 0])
                inter_y1 = torch.maximum(boxes[:, 1], candidate[0, 1])
                inter_x2 = torch.minimum(boxes[:, 2], candidate[0, 2])
                inter_y2 = torch.minimum(boxes[:, 3], candidate[0, 3])
                inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
                candidate_area = float(bw * bh)
                if float((inter / max(1.0, candidate_area)).max()) <= 0.30:
                    placement = (px, py, candidate)
                    break
            if placement is None:
                continue
            px, py, new_box = placement
            image.paste(crop, (px, py))
            if boxes.numel() > 0:
                boxes = torch.cat([boxes, new_box], dim=0)
                labels = torch.cat([labels, donor_labels[i : i + 1]], dim=0)
            else:
                boxes = new_box
                labels = donor_labels[i : i + 1]
        return image, boxes, labels

    def __getitem__(self, index):
        if self.augment and random.random() < self.mosaic_prob:
            image, boxes, labels, info = self._mosaic(index)
            meta = {"scale": 1.0, "pad_x": 0, "pad_y": 0, "orig_w": self.img_size, "orig_h": self.img_size}
        else:
            image, boxes, labels, info = self._load_item(index)
            image, boxes, meta = resize_letterbox(image, boxes, self.img_size)

        if self.augment and random.random() < self.fliplr:
            image = TF.hflip(image)
            w = image.width
            if boxes.numel() > 0:
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()
                boxes[:, 0] = w - old_x2
                boxes[:, 2] = w - old_x1

        if self.augment and random.random() < self.flipud:
            image = TF.vflip(image)
            h = image.height
            if boxes.numel() > 0:
                old_y1 = boxes[:, 1].clone()
                old_y2 = boxes[:, 3].clone()
                boxes[:, 1] = h - old_y2
                boxes[:, 3] = h - old_y1

        if self.augment:
            if random.random() < self.affine_prob:
                image, boxes, labels = self._random_affine(image, boxes, labels)
            if random.random() < self.perspective_prob:
                image, boxes, labels = self._random_perspective(image, boxes, labels)
            if random.random() < self.mixup_prob:
                image, boxes, labels = self._mixup(image, boxes, labels)
            if random.random() < self.copypaste_prob:
                image, boxes, labels = self._copypaste(image, boxes, labels)
            if random.random() < self.focused_copypaste_prob:
                image, boxes, labels = self._copypaste(image, boxes, labels, focused_only=True)
            if random.random() < 0.4:
                image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.25))
            if random.random() < 0.4:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
            if random.random() < 0.4:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
            if random.random() < 0.3:
                image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.5, 1.5))
            if random.random() < 0.5:
                h_shift = random.randint(-18, 18)
                img_hsv = image.convert("HSV")
                h, s, v = img_hsv.split()
                h = h.point(lambda x: (x + h_shift) % 256)
                image = Image.merge("HSV", (h, s, v)).convert("RGB")

        tensor = image_to_tensor(image, self.imagenet_normalize)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": info["id"],
            "meta": meta,
        }
        return tensor, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
