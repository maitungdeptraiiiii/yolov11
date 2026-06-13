import torch
import torch.nn.functional as F

from .boxes import box_iou, ciou_loss


class YOLOLoss:
    """YOLOv8-style loss: TaskAlignedAssigner + CIoU + DFL + BCE classification."""

    def __init__(
        self,
        num_classes,
        img_size=640,
        strides=(8, 16, 32),
        reg_max=16,
        box_weight=7.5,
        dfl_weight=1.5,
        cls_weight=0.5,
        class_weights=None,
        tal_topk=10,
        tal_alpha=0.5,
        tal_beta=6.0,
    ):
        self.num_classes = num_classes
        self.img_size = img_size
        self.strides = strides
        self.reg_max = reg_max
        self.box_weight = box_weight
        self.dfl_weight = dfl_weight
        self.cls_weight = cls_weight
        self.class_weights = class_weights.float() if class_weights is not None else None
        self.tal_topk = tal_topk
        self.tal_alpha = tal_alpha
        self.tal_beta = tal_beta
        self.project = torch.arange(reg_max + 1, dtype=torch.float32)

    @property
    def reg_channels(self):
        return 4 * (self.reg_max + 1)

    def _make_anchors(self, outputs):
        points = []
        strides = []
        device = outputs[0].device
        for pred, stride in zip(outputs, self.strides):
            _, _, h, w = pred.shape
            yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            point = torch.stack([(xx.float() + 0.5) * stride, (yy.float() + 0.5) * stride], dim=-1).view(-1, 2)
            points.append(point)
            strides.append(torch.full((h * w, 1), stride, device=device, dtype=point.dtype))
        return torch.cat(points, dim=0), torch.cat(strides, dim=0)

    def _flatten_outputs(self, outputs):
        reg_all = []
        cls_all = []
        for pred in outputs:
            reg = pred[:, : self.reg_channels]
            cls = pred[:, self.reg_channels : self.reg_channels + self.num_classes]
            reg_all.append(reg.permute(0, 2, 3, 1).reshape(pred.shape[0], -1, self.reg_channels))
            cls_all.append(cls.permute(0, 2, 3, 1).reshape(pred.shape[0], -1, self.num_classes))
        return torch.cat(reg_all, dim=1), torch.cat(cls_all, dim=1)

    def _decode_boxes(self, reg_logits, anchor_points, stride_tensor):
        b, n, _ = reg_logits.shape
        device = reg_logits.device
        project = self.project.to(device=device, dtype=reg_logits.dtype)
        dist = reg_logits.view(b, n, 4, self.reg_max + 1).softmax(dim=3)
        dist = (dist * project.view(1, 1, 1, -1)).sum(dim=3) * stride_tensor.view(1, n, 1)
        x1 = anchor_points[:, 0].view(1, n) - dist[:, :, 0]
        y1 = anchor_points[:, 1].view(1, n) - dist[:, :, 1]
        x2 = anchor_points[:, 0].view(1, n) + dist[:, :, 2]
        y2 = anchor_points[:, 1].view(1, n) + dist[:, :, 3]
        return torch.stack([x1, y1, x2, y2], dim=2)

    def _anchors_in_boxes(self, anchor_points, gt_boxes):
        x, y = anchor_points[:, 0:1], anchor_points[:, 1:2]
        return (x >= gt_boxes[:, 0]) & (y >= gt_boxes[:, 1]) & (x <= gt_boxes[:, 2]) & (y <= gt_boxes[:, 3])

    @torch.no_grad()
    def _assign_one_image(self, pred_scores, pred_boxes, anchor_points, gt_boxes, gt_labels):
        n = pred_scores.shape[0]
        target_boxes = torch.zeros((n, 4), dtype=pred_boxes.dtype, device=pred_boxes.device)
        target_scores = torch.zeros((n, self.num_classes), dtype=pred_scores.dtype, device=pred_scores.device)
        fg_mask = torch.zeros(n, dtype=torch.bool, device=pred_boxes.device)

        if gt_boxes.numel() == 0:
            return target_boxes, target_scores, fg_mask

        in_gts = self._anchors_in_boxes(anchor_points, gt_boxes)
        if not in_gts.any():
            return target_boxes, target_scores, fg_mask

        overlaps = box_iou(pred_boxes, gt_boxes).clamp(min=0)
        cls_scores = pred_scores[:, gt_labels].clamp(min=1e-9)
        align_metric = cls_scores.pow(self.tal_alpha) * overlaps.clamp(min=1e-6).pow(self.tal_beta)
        align_metric = align_metric * in_gts.float()

        topk = min(self.tal_topk, n)
        topk_metrics, topk_idxs = torch.topk(align_metric, topk, dim=0)
        candidate_mask = torch.zeros_like(align_metric, dtype=torch.bool)
        for gt_idx in range(gt_boxes.shape[0]):
            valid = topk_metrics[:, gt_idx] > 0
            candidate_mask[topk_idxs[valid, gt_idx], gt_idx] = True

        pos_mask = candidate_mask & in_gts & (align_metric > 0)
        if not pos_mask.any():
            return target_boxes, target_scores, fg_mask

        fg_mask = pos_mask.any(dim=1)
        matched_gt = (overlaps * pos_mask.float()).argmax(dim=1)
        fg_idx = torch.where(fg_mask)[0]
        matched_gt_fg = matched_gt[fg_idx]

        pos_align = align_metric.max(dim=0).values.clamp(min=1e-9)
        pos_overlap = (overlaps * pos_mask.float()).max(dim=0).values
        norm_align = (align_metric * pos_overlap.view(1, -1) / pos_align.view(1, -1)).clamp(max=1.0)
        assigned_scores = norm_align[fg_idx, matched_gt_fg]

        target_boxes[fg_idx] = gt_boxes[matched_gt_fg].to(dtype=target_boxes.dtype)
        target_scores[fg_idx, gt_labels[matched_gt_fg]] = assigned_scores.to(dtype=target_scores.dtype)
        return target_boxes, target_scores, fg_mask

    def _dfl_loss(self, pred_logits, target_dist, weight, normalizer):
        target = target_dist.clamp(0, self.reg_max - 1e-3)
        left = target.floor().long()
        right = left + 1
        wl = right.float() - target
        wr = target - left.float()

        pred = pred_logits.reshape(-1, self.reg_max + 1)
        loss = (
            F.cross_entropy(pred, left.reshape(-1), reduction="none") * wl.reshape(-1)
            + F.cross_entropy(pred, right.reshape(-1), reduction="none") * wr.reshape(-1)
        )
        loss = loss.view(-1, 4).mean(dim=1)
        return (loss * weight).sum() / normalizer

    def __call__(self, outputs, targets):
        device = outputs[0].device
        anchor_points, stride_tensor = self._make_anchors(outputs)
        reg_logits, cls_logits = self._flatten_outputs(outputs)
        pred_boxes = self._decode_boxes(reg_logits, anchor_points, stride_tensor)
        pred_scores = cls_logits.sigmoid()

        batch_size, num_anchors, _ = cls_logits.shape
        target_boxes = torch.zeros((batch_size, num_anchors, 4), dtype=pred_boxes.dtype, device=device)
        target_scores = torch.zeros((batch_size, num_anchors, self.num_classes), dtype=pred_scores.dtype, device=device)
        fg_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=device)

        for bi, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device).long()
            boxes, labels = self._filter_valid_targets(boxes, labels)
            assigned_boxes, assigned_scores, assigned_mask = self._assign_one_image(
                pred_scores[bi].detach(), pred_boxes[bi].detach(), anchor_points, boxes, labels
            )
            target_boxes[bi] = assigned_boxes
            target_scores[bi] = assigned_scores
            fg_mask[bi] = assigned_mask

        target_scores_sum = target_scores.sum().clamp(min=1.0)
        if self.class_weights is None:
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, target_scores, reduction="sum") / target_scores_sum
        else:
            class_weights = self.class_weights.to(device=device, dtype=cls_logits.dtype).view(1, 1, self.num_classes)
            positive_weight = 1.0 + (class_weights - 1.0) * target_scores
            cls_normalizer = (target_scores * class_weights).sum().clamp(min=1.0)
            cls_loss = (
                F.binary_cross_entropy_with_logits(cls_logits, target_scores, reduction="none") * positive_weight
            ).sum() / cls_normalizer

        if fg_mask.any():
            fg_weight = target_scores.sum(dim=2)[fg_mask]
            box_loss = (ciou_loss(pred_boxes[fg_mask], target_boxes[fg_mask]) * fg_weight).sum() / target_scores_sum

            anchor_fg = anchor_points.view(1, num_anchors, 2).expand(batch_size, -1, -1)[fg_mask]
            stride_fg = stride_tensor.view(1, num_anchors, 1).expand(batch_size, -1, -1)[fg_mask]
            target_fg = target_boxes[fg_mask]
            target_dist = torch.stack(
                [
                    (anchor_fg[:, 0] - target_fg[:, 0]) / stride_fg[:, 0],
                    (anchor_fg[:, 1] - target_fg[:, 1]) / stride_fg[:, 0],
                    (target_fg[:, 2] - anchor_fg[:, 0]) / stride_fg[:, 0],
                    (target_fg[:, 3] - anchor_fg[:, 1]) / stride_fg[:, 0],
                ],
                dim=1,
            )
            dfl_loss = self._dfl_loss(reg_logits[fg_mask].view(-1, 4, self.reg_max + 1), target_dist, fg_weight, target_scores_sum)
        else:
            box_loss = torch.tensor(0.0, device=device)
            dfl_loss = torch.tensor(0.0, device=device)

        total = self.box_weight * box_loss + self.dfl_weight * dfl_loss + self.cls_weight * cls_loss
        logs = {
            "loss": float(total.detach().cpu()),
            "box": float(box_loss.detach().cpu()),
            "dfl": float(dfl_loss.detach().cpu()),
            "cls": float(cls_loss.detach().cpu()),
            "positives": int(fg_mask.sum().detach().cpu()),
            "target_score": float(target_scores_sum.detach().cpu()),
        }
        return total, logs

    def _filter_valid_targets(self, boxes, labels):
        if boxes.numel() == 0:
            return boxes.reshape(0, 4), labels.reshape(0)
        wh = boxes[:, 2:4] - boxes[:, 0:2]
        keep = (wh[:, 0] > 1) & (wh[:, 1] > 1)
        return boxes[keep], labels[keep]
