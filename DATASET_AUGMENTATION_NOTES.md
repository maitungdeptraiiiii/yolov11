# Dataset Scan and Augmentation Choice

Scanned dataset: `C:\Users\Admin\Downloads\final_public\public`

## Summary

Train:

- Images: 7500
- Annotations: 10642
- Images without boxes: 2500
- Max boxes per image: 22
- Class counts: dog 1028, person 5829, chair 1613, cat 833, car 1339
- Relative box area quantiles q10/q25/median/q75/q90: 0.0091, 0.0303, 0.1117, 0.3194, 0.6085
- Small/medium/large boxes by relative area: 1153 / 3917 / 5572
- Median relative area by class: dog 0.3054, person 0.1042, chair 0.0640, cat 0.4545, car 0.0396

Validation:

- Images: 1500
- Annotations: 2021
- Images without boxes: 500
- Class counts: dog 206, chair 282, person 1074, car 283, cat 176
- Relative box area quantiles q10/q25/median/q75/q90: 0.0087, 0.0293, 0.1197, 0.3342, 0.6363

## Decision

The dataset has a strong class imbalance toward `person`, many negative images, and a meaningful number of small objects. `car` and `chair` are the classes most likely to benefit from extra context and scale diversity. `cat` and `dog` are often large in the image, so aggressive cropping is risky.

Implemented augmentations:

- Horizontal flip with bbox update.
- Mosaic with probability 0.45.
- Mild scale and translation with probability 0.50.
- Low MixUp probability 0.10.
- Color, brightness, and contrast jitter.
- Optional multi-scale training with `--multi_scale`.

Recommended command:

```bash
python train.py \
  --train_data C:\Users\Admin\Downloads\final_public\public\annotations\train.json \
  --val_data C:\Users\Admin\Downloads\final_public\public\annotations\val.json \
  --image_dir C:\Users\Admin\Downloads\final_public\public\train\images \
  --val_image_dir C:\Users\Admin\Downloads\final_public\public\val\images \
  --checkpoint_dir ./models/ \
  --epochs 100 \
  --batch_size 8 \
  --img_size 960 \
  --multi_scale
```

For a quick ablation, compare:

```bash
--mosaic_prob 0.0 --mixup_prob 0.0 --affine_prob 0.0
```

against the default augmentation setup.
