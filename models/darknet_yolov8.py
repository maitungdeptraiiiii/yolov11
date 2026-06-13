import torch
from torch import nn
import torch.nn.functional as F


YOLOV8_SCALES = {
    "n": (0.33, 0.25, 1024),
    "s": (0.33, 0.50, 1024),
    "m": (0.67, 0.75, 768),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.25, 512),
}


def make_divisible(value, divisor=8):
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True):
        super().__init__()
        hidden = channels // 2
        self.cv1 = ConvBNAct(channels, hidden, 1, 1)
        self.cv2 = ConvBNAct(hidden, channels, 3, 1)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class C2f(nn.Module):
    """Small CSP/C2f block inspired by YOLOv8, implemented only with basic layers."""

    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.hidden = c2 // 2
        self.cv1 = ConvBNAct(c1, self.hidden * 2, 1, 1)
        self.blocks = nn.ModuleList(Bottleneck(self.hidden, shortcut=True) for _ in range(n))
        self.cv2 = ConvBNAct(self.hidden * (2 + n), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = ConvBNAct(c1, hidden, 1, 1)
        self.cv2 = ConvBNAct(hidden * 4, c2, 1, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class DarknetBackbone(nn.Module):
    """CSPDarknet-style feature extractor returning P3, P4, P5 feature maps."""

    def __init__(self, width=0.50, depth=0.34, max_channels=1024):
        super().__init__()

        def ch(v):
            return min(max_channels, make_divisible(v * width, 8))

        def rep(v):
            return max(1, round(v * depth))

        c1, c2, c3, c4, c5 = ch(64), ch(128), ch(256), ch(512), ch(1024)
        self.stem = ConvBNAct(3, c1, 3, 2)
        self.stage2 = nn.Sequential(ConvBNAct(c1, c2, 3, 2), C2f(c2, c2, rep(3)))
        self.stage3 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), C2f(c3, c3, rep(6)))
        self.stage4 = nn.Sequential(ConvBNAct(c3, c4, 3, 2), C2f(c4, c4, rep(6)))
        self.stage5 = nn.Sequential(ConvBNAct(c4, c5, 3, 2), C2f(c5, c5, rep(3)), SPPF(c5, c5))
        self.channels = (c3, c4, c5)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5


class PANFPN(nn.Module):
    def __init__(self, channels, width=0.50, depth=0.34):
        super().__init__()
        c3, c4, c5 = channels

        def rep(v):
            return max(1, round(v * depth))

        self.reduce5 = ConvBNAct(c5, c4, 1, 1)
        self.top4 = C2f(c4 + c4, c4, rep(3))
        self.reduce4 = ConvBNAct(c4, c3, 1, 1)
        self.top3 = C2f(c3 + c3, c3, rep(3))
        self.down3 = ConvBNAct(c3, c3, 3, 2)
        self.pan4 = C2f(c3 + c4, c4, rep(3))
        self.down4 = ConvBNAct(c4, c4, 3, 2)
        self.pan5 = C2f(c4 + c4, c5, rep(3))
        self.out_channels = (c3, c4, c5)

    def forward(self, feats):
        p3, p4, p5 = feats
        n5 = self.reduce5(p5)
        n4 = self.top4(torch.cat([F.interpolate(n5, size=p4.shape[-2:], mode="nearest"), p4], dim=1))
        n4r = self.reduce4(n4)
        n3 = self.top3(torch.cat([F.interpolate(n4r, size=p3.shape[-2:], mode="nearest"), p3], dim=1))
        o4 = self.pan4(torch.cat([self.down3(n3), n4], dim=1))
        o5 = self.pan5(torch.cat([self.down4(o4), n5], dim=1))
        return n3, o4, o5


class PANFPNP2(nn.Module):
    """Four-level PAN-FPN producing stride-4 through stride-32 features."""

    def __init__(self, channels, width=0.50, depth=0.34):
        super().__init__()
        c2, c3, c4, c5 = channels

        def rep(v):
            return max(1, round(v * depth))

        self.reduce5 = ConvBNAct(c5, c4, 1, 1)
        self.top4 = C2f(c4 + c4, c4, rep(3))
        self.reduce4 = ConvBNAct(c4, c3, 1, 1)
        self.top3 = C2f(c3 + c3, c3, rep(3))
        self.reduce3 = ConvBNAct(c3, c2, 1, 1)
        self.top2 = C2f(c2 + c2, c2, rep(2))
        self.down2 = ConvBNAct(c2, c2, 3, 2)
        self.pan3 = C2f(c2 + c3, c3, rep(3))
        self.down3 = ConvBNAct(c3, c3, 3, 2)
        self.pan4 = C2f(c3 + c4, c4, rep(3))
        self.down4 = ConvBNAct(c4, c4, 3, 2)
        self.pan5 = C2f(c4 + c4, c5, rep(3))
        self.out_channels = (c2, c3, c4, c5)

    def forward(self, feats):
        p2, p3, p4, p5 = feats
        n5 = self.reduce5(p5)
        n4 = self.top4(torch.cat([F.interpolate(n5, size=p4.shape[-2:], mode="nearest"), p4], dim=1))
        n4r = self.reduce4(n4)
        n3 = self.top3(torch.cat([F.interpolate(n4r, size=p3.shape[-2:], mode="nearest"), p3], dim=1))
        n3r = self.reduce3(n3)
        n2 = self.top2(torch.cat([F.interpolate(n3r, size=p2.shape[-2:], mode="nearest"), p2], dim=1))
        o3 = self.pan3(torch.cat([self.down2(n2), n3], dim=1))
        o4 = self.pan4(torch.cat([self.down3(o3), n4], dim=1))
        o5 = self.pan5(torch.cat([self.down4(o4), n5], dim=1))
        return n2, o3, o4, o5


class DetectHead(nn.Module):
    def __init__(self, channels, num_classes, reg_max=16, strides=None):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = tuple(strides or (8, 16, 32))
        if len(self.strides) != len(channels):
            raise ValueError("DetectHead strides must match the number of feature levels.")
        self.reg_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()
        reg_hidden = max(16, channels[0] // 4, 4 * reg_max)
        cls_hidden = max(channels[0], min(num_classes, 100))
        for c in channels:
            self.reg_heads.append(
                nn.Sequential(
                    ConvBNAct(c, reg_hidden, 3, 1),
                    ConvBNAct(reg_hidden, reg_hidden, 3, 1),
                    nn.Conv2d(reg_hidden, 4 * (reg_max + 1), 1),
                )
            )
            self.cls_heads.append(
                nn.Sequential(
                    ConvBNAct(c, cls_hidden, 3, 1),
                    ConvBNAct(cls_hidden, cls_hidden, 3, 1),
                    nn.Conv2d(cls_hidden, num_classes, 1),
                )
            )
        self.initialize_biases()

    def initialize_biases(self):
        for reg, cls, stride in zip(self.reg_heads, self.cls_heads, self.strides):
            reg[-1].bias.data.fill_(1.0)
            cls[-1].bias.data[: self.num_classes] = torch.log(torch.tensor(5.0 / self.num_classes / (640 / stride) ** 2))

    def forward(self, feats):
        outputs = []
        for feat, reg, cls in zip(feats, self.reg_heads, self.cls_heads):
            outputs.append(torch.cat([reg(feat), cls(feat)], dim=1))
        return outputs


class DarknetYOLOv8(nn.Module):
    """YOLOv8-like detector built from scratch with a Darknet/CSP backbone."""

    def __init__(self, num_classes=5, width=0.50, depth=0.34, reg_max=16, scale=None, max_channels=1024):
        super().__init__()
        if scale:
            depth, width, max_channels = YOLOV8_SCALES[scale]
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.scale = scale
        self.backbone = DarknetBackbone(width=width, depth=depth, max_channels=max_channels)
        self.neck = PANFPN(self.backbone.channels, width=width, depth=depth)
        self.head = DetectHead(self.neck.out_channels, num_classes, reg_max=reg_max)
        self.strides = self.head.strides

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))
