import torch


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def ciou_loss(pred, target):
    pred = pred.float()
    target = target.float()
    iou = box_iou(pred, target).diag()

    pcx = (pred[:, 0] + pred[:, 2]) * 0.5
    pcy = (pred[:, 1] + pred[:, 3]) * 0.5
    tcx = (target[:, 0] + target[:, 2]) * 0.5
    tcy = (target[:, 1] + target[:, 3]) * 0.5
    center_dist = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

    enc_x1 = torch.min(pred[:, 0], target[:, 0])
    enc_y1 = torch.min(pred[:, 1], target[:, 1])
    enc_x2 = torch.max(pred[:, 2], target[:, 2])
    enc_y2 = torch.max(pred[:, 3], target[:, 3])
    diag = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + 1e-6

    pw = (pred[:, 2] - pred[:, 0]).clamp(min=1e-6)
    ph = (pred[:, 3] - pred[:, 1]).clamp(min=1e-6)
    tw = (target[:, 2] - target[:, 0]).clamp(min=1e-6)
    th = (target[:, 3] - target[:, 1]).clamp(min=1e-6)
    v = (4 / torch.pi**2) * (torch.atan(tw / th) - torch.atan(pw / ph)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)
    ciou = iou - center_dist / diag - alpha * v
    return 1 - ciou.clamp(min=-1, max=1)


def nms(boxes, scores, iou_threshold=0.5):
    keep = []
    order = scores.argsort(descending=True)
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].view(1, 4), boxes[order[1:]]).view(-1)
        order = order[1:][ious <= iou_threshold]
    if not keep:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return torch.stack(keep)


def clip_boxes(boxes, width, height):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    return boxes
