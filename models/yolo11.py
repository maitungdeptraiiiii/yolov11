import torch
from torch import nn
import torch.nn.functional as F

from .darknet_yolov8 import ConvBNAct, DetectHead


class YOLO11Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, kernels=(3, 3), expansion=0.5):
        super().__init__()
        hidden = max(1, int(c2 * expansion))
        self.cv1 = ConvBNAct(c1, hidden, kernels[0], 1)
        self.cv2 = ConvBNAct(hidden, c2, kernels[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Module):
    """C3 block with configurable bottleneck kernels."""

    def __init__(self, c1, c2, n=1, shortcut=True, expansion=0.5, kernel=3):
        super().__init__()
        hidden = max(1, int(c2 * expansion))
        self.cv1 = ConvBNAct(c1, hidden, 1, 1)
        self.cv2 = ConvBNAct(c1, hidden, 1, 1)
        self.blocks = nn.Sequential(
            *(YOLO11Bottleneck(hidden, hidden, shortcut, (kernel, kernel), 1.0) for _ in range(n))
        )
        self.cv3 = ConvBNAct(hidden * 2, c2, 1, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), dim=1))


class C3k2(nn.Module):
    """YOLO11 CSP block, using either bottlenecks or nested C3k blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, shortcut=True, expansion=0.5):
        super().__init__()
        self.hidden = max(1, int(c2 * expansion))
        self.cv1 = ConvBNAct(c1, self.hidden * 2, 1, 1)
        block = (
            lambda: C3k(self.hidden, self.hidden, n=2, shortcut=shortcut, expansion=1.0)
            if c3k
            else YOLO11Bottleneck(self.hidden, self.hidden, shortcut, (3, 3), 1.0)
        )
        self.blocks = nn.ModuleList(block() for _ in range(n))
        self.cv2 = ConvBNAct(self.hidden * (2 + n), c2, 1, 1)

    def forward(self, x):
        features = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            features.append(block(features[-1]))
        return self.cv2(torch.cat(features, dim=1))


class Attention(nn.Module):
    """Position-sensitive multi-head attention used by YOLO11 C2PSA."""

    def __init__(self, channels, num_heads=None, attn_ratio=0.5):
        super().__init__()
        num_heads = num_heads or max(1, channels // 64)
        while channels % num_heads:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.key_dim = max(1, int(self.head_dim * attn_ratio))
        self.scale = self.key_dim ** -0.5
        qkv_channels = channels + 2 * self.key_dim * num_heads
        self.qkv = ConvBNAct(channels, qkv_channels, 1, 1)
        self.proj = ConvBNAct(channels, channels, 1, 1)
        self.pe = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        points = height * width
        qkv = self.qkv(x).view(batch, self.num_heads, 2 * self.key_dim + self.head_dim, points)
        q, k, v = qkv.split((self.key_dim, self.key_dim, self.head_dim), dim=2)
        weights = (q.transpose(-2, -1) @ k) * self.scale
        weights = weights.softmax(dim=-1)
        attended = (v @ weights.transpose(-2, -1)).reshape(batch, channels, height, width)
        return self.proj(attended + self.pe(v.reshape(batch, channels, height, width)))


class PSABlock(nn.Module):
    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.attn = Attention(channels)
        self.ffn = nn.Sequential(
            ConvBNAct(channels, channels * 2, 1, 1),
            ConvBNAct(channels * 2, channels, 1, 1),
        )
        self.shortcut = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.shortcut else self.attn(x)
        return x + self.ffn(x) if self.shortcut else self.ffn(x)


class C2PSA(nn.Module):
    def __init__(self, c1, c2=None, n=1, expansion=0.5):
        super().__init__()
        c2 = c2 or c1
        if c1 != c2:
            raise ValueError("C2PSA requires equal input and output channels.")
        self.hidden = max(1, int(c2 * expansion))
        self.cv1 = ConvBNAct(c1, self.hidden * 2, 1, 1)
        self.blocks = nn.Sequential(*(PSABlock(self.hidden) for _ in range(n)))
        self.cv2 = ConvBNAct(self.hidden * 2, c2, 1, 1)

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, dim=1)
        return self.cv2(torch.cat((a, self.blocks(b)), dim=1))


class YOLO11PANFPN(nn.Module):
    """YOLO11 P3-P5 feature fusion topology."""

    def __init__(self, channels, depth=0.5):
        super().__init__()
        c3, c4, c5 = channels
        repeats = max(1, round(2 * depth))
        self.top4 = C3k2(c5 + c4, c4, repeats, c3k=False)
        self.top3 = C3k2(c4 + c3, c3, repeats, c3k=False)
        self.down3 = ConvBNAct(c3, c3, 3, 2)
        self.pan4 = C3k2(c3 + c4, c4, repeats, c3k=False)
        self.down4 = ConvBNAct(c4, c4, 3, 2)
        self.pan5 = C3k2(c4 + c5, c5, repeats, c3k=True)
        self.out_channels = (c3, c4, c5)

    def forward(self, feats):
        p3, p4, p5 = feats
        n4 = self.top4(torch.cat((F.interpolate(p5, size=p4.shape[-2:], mode="nearest"), p4), dim=1))
        n3 = self.top3(torch.cat((F.interpolate(n4, size=p3.shape[-2:], mode="nearest"), p3), dim=1))
        o4 = self.pan4(torch.cat((self.down3(n3), n4), dim=1))
        o5 = self.pan5(torch.cat((self.down4(o4), p5), dim=1))
        return n3, o4, o5


class YOLO11PANFPNP2(nn.Module):
    """YOLO11-style feature fusion extended with a stride-4 detection level."""

    def __init__(self, channels, depth=0.5):
        super().__init__()
        c2, c3, c4, c5 = channels
        repeats = max(1, round(2 * depth))
        self.top4 = C3k2(c5 + c4, c4, repeats, c3k=False)
        self.top3 = C3k2(c4 + c3, c3, repeats, c3k=False)
        self.top2 = C3k2(c3 + c2, c2, repeats, c3k=False)
        self.down2 = ConvBNAct(c2, c2, 3, 2)
        self.pan3 = C3k2(c2 + c3, c3, repeats, c3k=False)
        self.down3 = ConvBNAct(c3, c3, 3, 2)
        self.pan4 = C3k2(c3 + c4, c4, repeats, c3k=False)
        self.down4 = ConvBNAct(c4, c4, 3, 2)
        self.pan5 = C3k2(c4 + c5, c5, repeats, c3k=True)
        self.out_channels = (c2, c3, c4, c5)

    def forward(self, feats):
        p2, p3, p4, p5 = feats
        n4 = self.top4(torch.cat((F.interpolate(p5, size=p4.shape[-2:], mode="nearest"), p4), dim=1))
        n3 = self.top3(torch.cat((F.interpolate(n4, size=p3.shape[-2:], mode="nearest"), p3), dim=1))
        n2 = self.top2(torch.cat((F.interpolate(n3, size=p2.shape[-2:], mode="nearest"), p2), dim=1))
        o3 = self.pan3(torch.cat((self.down2(n2), n3), dim=1))
        o4 = self.pan4(torch.cat((self.down3(o3), n4), dim=1))
        o5 = self.pan5(torch.cat((self.down4(o4), p5), dim=1))
        return n2, o3, o4, o5


class ConvNeXtYOLO11(nn.Module):
    """ConvNeXt-Tiny backbone with YOLO11 C2PSA, C3k2 neck, and local detect head."""

    def __init__(self, backbone, num_classes=5, reg_max=16, neck_depth=0.5, p2_head=False):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.scale = "convnext_tiny_yolo11"
        self.architecture = "convnext_yolo11"
        self.p2_head = p2_head
        self.backbone = backbone
        self.c2psa = C2PSA(backbone.channels[-1], n=max(1, round(2 * neck_depth)))
        neck_class = YOLO11PANFPNP2 if p2_head else YOLO11PANFPN
        self.neck = neck_class(backbone.channels, depth=neck_depth)
        strides = (4, 8, 16, 32) if p2_head else (8, 16, 32)
        self.head = DetectHead(self.neck.out_channels, num_classes, reg_max=reg_max, strides=strides)
        self.strides = self.head.strides

    def forward(self, x):
        features = list(self.backbone(x))
        features[-1] = self.c2psa(features[-1])
        return self.head(self.neck(tuple(features)))
