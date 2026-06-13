# ConvNeXt-YOLO11 Object Detector

This project implements a ConvNeXt-YOLO11 hybrid detector with a custom training, loss, inference, and NMS pipeline. It does not use a pretrained Ultralytics detector. By default it keeps an ImageNet-pretrained ConvNeXt-Tiny feature extractor, then applies YOLO11 C2PSA attention and a C3k2 PAN-FPN neck. The detection head, loss, decoding, and NMS remain implemented locally. If pretrained feature extractors are not allowed, pass `--no_backbone_pretrained`.

## Idea

The architecture is `ConvNeXt-Tiny -> C2PSA -> YOLO11 C3k2 PAN-FPN -> anchor-free decoupled DetectHead`. Training retains TaskAlignedAssigner targets, DFL box regression, BCE classification, per-class NMS, EMA, backbone freezing, class balancing, augmentation, cosine scheduling, and early stopping. The default backbone is a torchvision classification model, not a pretrained object detector.

## Files

- `models/convnext_yolo.py`: unchanged ConvNeXt-Tiny feature extractor connected to the YOLO11 model.
- `models/yolo11.py`: C3k2, C2PSA, YOLO11 PAN-FPN, and the ConvNeXt-YOLO11 detector.
- `models/darknet_yolov8.py`: optional Darknet/CSP backbone, PAN-FPN neck, anchor-free detection head.
- `utils/data.py`: JSON reader, image resizing with letterbox, bbox conversion, horizontal flip and color augmentation.
- `utils/loss.py`: TaskAlignedAssigner, BCE classification, CIoU localization loss, and DFL box regression.
- `utils/inference.py`: output decoding, confidence thresholding, per-class NMS, conversion back to original image coordinates.
- `train.py`: training loop, validation prediction export, best checkpoint saving.
- `predict.py`: required inference entry point.
- `average_checkpoints.py`: average top validation checkpoints into a model soup.

## Install

```bash
pip install -r requirements.txt
```

## Train

Required command:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Useful optional parameters:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --epochs 100 \
  --batch_size 8 \
  --img_size 960 \
  --lr 0.002 \
  --backbone convnext_tiny \
  --freeze_backbone_epochs 10 \
  --multi_scale
```

For the strongest validation `mAP@0.5`, the default training loop now validates the EMA model and sweeps confidence/NMS thresholds at every epoch. The best checkpoint stores those thresholds, and `predict.py` uses them automatically unless you pass explicit thresholds.
After the global sweep, it also refines confidence thresholds per class by default, which can help when minority classes have systematically lower scores. Disable this ablation with `--no_classwise_conf_sweep`.

Recommended mAP@0.5-focused run:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --epochs 120 \
  --batch_size 8 \
  --img_size 960 \
  --lr 0.001 \
  --backbone_lr_scale 0.10 \
  --backbone convnext_tiny \
  --imagenet_normalize \
  --freeze_backbone_epochs 10 \
  --multi_scale \
  --ema_decay 0.9998 \
  --mosaic_prob 0.15 \
  --mixup_prob 0.03 \
  --close_mosaic_epochs 20 \
  --multi_scale \
  --multi_scale_sizes 896,960,1024,1088,1152 \
  --close_multiscale_epochs 20 \
  --val_tta \
  --val_conf_values 0.001,0.003,0.005,0.01,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50 \
  --val_iou_values 0.45,0.50,0.55,0.60,0.65,0.70
```

For `--img_size 1024`, use multi-scale sizes near 1024. The older `416,448,480` recipe is only appropriate for a 448-ish training setup and is too small for this model/dataset if you are optimizing small-object recall.

If `--save_top_k` keeps several close validation checkpoints, create a model soup:

```bash
python average_checkpoints.py \
  --checkpoints ./models/top_epoch*.pth \
  --output ./models/model_soup.pth
```

Class-balanced sampling and class-weighted classification loss are enabled by default. They compute weights from `train.json` so minority classes such as `cat`, `dog`, `car`, and `chair` get sampled/loss-weighted more than `person`:

```bash
--class_balance_power 0.5 --max_class_weight 3.0 --empty_sample_weight 0.25
```

Disable them for ablations with:

```bash
--no_class_balanced_sampler --no_class_loss_weights
```

Augmentation options are enabled by default for training:

```bash
--mosaic_prob 0.45 --mixup_prob 0.10 --affine_prob 0.50 --translate 0.10 --scale_gain 0.20 --shear 0.20 --fliplr 0.50 --multi_scale
```

Vertical flip and perspective are available but disabled by default because upside-down people/vehicles and strong perspective warps are often unrealistic for this dataset:

```bash
--flipud 0.0 --perspective 0.0 --perspective_prob 0.0
```

Recommended for this dataset:

- Mosaic: useful because `car` and `chair` have many small boxes and fewer samples than `person`.
- Mild scale/translate: useful because object positions and image aspect ratios vary widely.
- Very mild shear: can improve robustness without distorting boxes too much.
- Horizontal flip: safe for all five classes.
- Vertical flip: disabled by default because it is usually unrealistic for natural vehicle/person images.
- Perspective: disabled by default; if tested, keep it mild such as `--perspective 0.02 --perspective_prob 0.10`.
- Moderate brightness/color/contrast jitter: useful for natural images.
- MixUp: keep low because one third of training images have no boxes and strong mixing can make labels noisy.
- Avoid strong random crop: `cat` and `dog` boxes are often very large, so aggressive crops can remove too much object content.

The best model is saved to:

```text
./models/best.pth
```

For Kaggle training, see `KAGGLE_TRAINING.md`.

## Predict

Required command:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

By default, `predict.py` loads:

```text
./models/best.pth
```

If this file is missing, `predict.py` automatically downloads the checkpoint from the public GitHub Release asset:

```text
https://github.com/maitungdeptraiiiii/yolov11/releases/download/v1.0.0/best.pth
```

You can override the download URL with `--checkpoint_url` or the `CHECKPOINT_URL` environment variable.

Use a custom checkpoint if needed:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output val_predictions.json \
  --checkpoint ./models/best.pth
```

## Tune Thresholds

After training, tune confidence and NMS IoU thresholds on validation:

```bash
python tune_threshold.py \
  --val_data ./public/annotations/val.json \
  --val_image_dir ./public/val/images \
  --checkpoint ./models/best.pth \
  --output ./models/threshold_tuning.json
```

Use the best thresholds from `threshold_tuning.json` when running `predict.py`:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json \
  --checkpoint ./models/best.pth \
  --tta
```

You can still override the stored thresholds manually with `--conf_threshold` and `--iou_threshold`.

## Output Format

`predictions.json` is a JSON array. Images with no detections are still included with an empty `boxes` list.

```json
[
  {
    "image_id": "img_7fd91a4c2e30.jpg",
    "boxes": [
      {
        "class": "person",
        "confidence": 0.91,
        "bbox": [48, 72, 210, 356]
      }
    ]
  }
]
```

## Notes for Report/Defense

- Data pipeline supports multiple objects per image and keeps boxes in `[xmin, ymin, xmax, ymax]`.
- Augmentation includes horizontal flip, brightness, contrast, and color jitter.
- The detector is anchor-free: all P3/P4/P5 anchor points participate in TaskAlignedAssigner matching.
- The head predicts left/top/right/bottom distance distributions plus class logits using the retained local `bbox + DFL` and `cls + BCE` design.
- The default model uses `--backbone convnext_tiny` with a YOLO11 C2PSA/C3k2 neck. The older custom Darknet/CSP model remains available with `--backbone darknet --scale n|s|m|l|x` for ablations.
- Inference applies confidence thresholding and NMS separately for each class.
