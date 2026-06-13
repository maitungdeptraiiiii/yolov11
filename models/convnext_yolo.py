import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .yolo11 import ConvNeXtYOLO11


class ConvNeXtTinyBackbone(nn.Module):
    """ConvNeXt-Tiny feature extractor returning optional stride-4 plus P3-P5 maps."""

    def __init__(self, pretrained=True, p2_head=False):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        self.features = model.features
        self.p2_head = p2_head
        self.channels = (96, 192, 384, 768) if p2_head else (192, 384, 768)

    def forward(self, x):
        c2 = c3 = c4 = c5 = None
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 1:
                c2 = x
            elif idx == 3:
                c3 = x
            elif idx == 5:
                c4 = x
            elif idx == 7:
                c5 = x
        return (c2, c3, c4, c5) if self.p2_head else (c3, c4, c5)


class ConvNeXtYOLO(ConvNeXtYOLO11):
    """ConvNeXt-Tiny detector using the YOLO11 C2PSA/C3k2 architecture."""

    def __init__(
        self,
        num_classes=5,
        reg_max=16,
        pretrained_backbone=True,
        neck_width=1.0,
        neck_depth=0.34,
        p2_head=False,
    ):
        backbone = ConvNeXtTinyBackbone(pretrained=pretrained_backbone, p2_head=p2_head)
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            reg_max=reg_max,
            neck_depth=neck_depth,
            p2_head=p2_head,
        )
