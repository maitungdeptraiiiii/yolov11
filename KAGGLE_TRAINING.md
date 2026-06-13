# Train on Kaggle

This project can be trained on Kaggle with a GPU notebook. The commands below assume the dataset is added to the notebook and appears at:

```text
/kaggle/input/final-public/public
```

If Kaggle gives your dataset a different folder name, change `DATA_ROOT` accordingly.

## 1. Upload Code

Recommended options:

- Upload this project as a Kaggle Dataset, then add it to the notebook.
- Or zip this folder, upload it to the notebook, and unzip into `/kaggle/working/yolov8`.
- Or use GitHub if the project is pushed to a repository.

Your working directory should contain:

```text
models/
utils/
train.py
predict.py
tune_threshold.py
requirements.txt
```

## 2. Enable GPU

In Kaggle Notebook:

```text
Settings -> Accelerator -> GPU T4 x2 or P100
```

## 3. Install Dependencies

Kaggle usually already has PyTorch. Run:

```bash
pip install -q -r requirements.txt
```

## 4. Check Dataset Paths

```bash
DATA_ROOT=/kaggle/input/final-public/public
ls $DATA_ROOT
ls $DATA_ROOT/annotations
ls $DATA_ROOT/train/images | head
```

Expected files:

```text
classes.json
annotations/train.json
annotations/val.json
train/images/
val/images/
tools/evaluate_predictions.py
```

## 5. Train

For Kaggle T4 x2, start with this. `--device 0,1` trains the model in parallel on both GPUs:

```bash
DATA_ROOT=/kaggle/input/final-public/public

python train.py \
  --train_data $DATA_ROOT/annotations/train.json \
  --val_data $DATA_ROOT/annotations/val.json \
  --image_dir $DATA_ROOT/train/images \
  --val_image_dir $DATA_ROOT/val/images \
  --checkpoint_dir /kaggle/working/models \
  --epochs 80 \
  --batch_size 8 \
  --img_size 960 \
  --lr 0.001 \
  --backbone convnext_tiny \
  --freeze_backbone_epochs 10 \
  --num_workers 2 \
  --device 0,1 \
  --multi_scale
```

The default backbone is ConvNeXt-Tiny used only as an ImageNet feature extractor. If pretrained feature extractors are not allowed for your class, add `--no_backbone_pretrained`.

Check that Kaggle sees both GPUs before training:

```bash
python - <<'PY'
import torch
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

If GPU memory is not enough:

```bash
--batch_size 4 --img_size 960
```

If training is too slow, use:

```bash
--epochs 40 --img_size 640 --batch_size 8
```

If you only have one GPU selected, remove `--device 0,1` or use `--device 0`.

The best checkpoint will be saved at:

```text
/kaggle/working/models/best.pth
```

## 6. Tune Thresholds

After training:

```bash
DATA_ROOT=/kaggle/input/final-public/public

python tune_threshold.py \
  --val_data $DATA_ROOT/annotations/val.json \
  --val_image_dir $DATA_ROOT/val/images \
  --checkpoint /kaggle/working/models/best.pth \
  --output /kaggle/working/models/threshold_tuning.json \
  --pred_dir /kaggle/working/models/threshold_tuning
```

Open `/kaggle/working/models/threshold_tuning.json` and use the best `conf_threshold` and `iou_threshold`.

## 7. Predict Validation

Example:

```bash
DATA_ROOT=/kaggle/input/final-public/public

python predict.py \
  --image_dir $DATA_ROOT/val/images \
  --output /kaggle/working/val_predictions.json \
  --checkpoint /kaggle/working/models/best.pth \
  --conf_threshold 0.25 \
  --iou_threshold 0.50
```

Then evaluate:

```bash
python $DATA_ROOT/tools/evaluate_predictions.py \
  --ground_truth $DATA_ROOT/annotations/val.json \
  --predictions /kaggle/working/val_predictions.json \
  --output /kaggle/working/val_score.json
```

## 8. Save Outputs

Kaggle keeps files in `/kaggle/working` as notebook outputs. Download these files after the run:

```text
/kaggle/working/models/best.pth
/kaggle/working/models/threshold_tuning.json
/kaggle/working/val_predictions.json
/kaggle/working/val_score.json
```

For final submission, place `best.pth` in this project's `models/` folder.
