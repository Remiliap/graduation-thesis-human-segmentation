from torch import nn
import torch
import torch.nn.functional as F

from net.d_block import DBlock
from net.align import flow_warp
from transform_img import transform_label

class RegSegBody2(nn.Module):
    def __init__(self, dilations: list[list[int]]):
        super().__init__()
        group_width = 24
        attention = "se"
        self.stage4 = nn.Sequential(
            DBlock(32, 48, [1], group_width, 2, attention),
            DBlock(48, 48, [1], group_width, 1, attention),
        )
        self.stage8 = nn.Sequential(
            DBlock(48, 120, [1], group_width, 2, attention),
            *[DBlock(120, 120, [1], group_width, 1, attention)
              for i in range(5)]
        )
        self.stage16 = nn.Sequential(
            DBlock(120, 336, [1], group_width, 2, attention),
            *[DBlock(336, 336, ds, group_width, 1, attention)
              for ds in dilations[:-1]],
            DBlock(336, 384, dilations[-1], group_width, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        return x4, x8, x16

    def channels(self):
        return {"4": 48, "8": 120, "16": 384}


class Fusion_v2(nn.Module):
    def __init__(self, h_ch, l_ch, fusion_ch, stack=1) -> None:
        super().__init__()
        self.stack = stack
        self.h_conv = nn.Sequential(
            nn.Conv2d(h_ch, fusion_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_ch)
        )

        self.l_conv = nn.Sequential(
            nn.Conv2d(l_ch, l_ch, kernel_size=3,
                      padding=1, groups=l_ch, bias=False),
            nn.BatchNorm2d(l_ch),
            nn.Conv2d(l_ch, fusion_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_ch)
        )

        self.flow_make = nn.Conv2d(
            2*fusion_ch, 2, kernel_size=3, padding=1, bias=False)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_ch, fusion_ch, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(fusion_ch),
            nn.ReLU(True),
            nn.Conv2d(fusion_ch, h_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(h_ch)
        )

    def forward(self, h: torch.Tensor, l: torch.Tensor):
        h_origin = h
        h = self.h_conv(h)
        l = self.l_conv(l)

        h_flow_make = F.interpolate(h, l.size()[-2:], mode="bilinear")
        flow = self.flow_make(torch.cat([h_flow_make, l], dim=1))

        h, h_origin = flow_warp(h, h_origin, flow=flow, size=l.size()[-2:])

        h = l * torch.sigmoid(h)
        h = self.fusion_conv(h)
        h = h+h_origin

        return h


class Decode_V2(nn.Module):
    def __init__(self, num_classes: int, channels: list[int]):
        super().__init__()

        channels4, channels8, channels16 = channels["4"], channels["8"], channels["16"]

        self.conv16 = nn.Sequential(
            nn.Conv2d(channels16, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )
        self.fusion_8_16 = Fusion_v2(256, channels8, 64)
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )
        self.fusion_4_8 = Fusion_v2(128, channels4, 64)

        self.classifier = nn.Conv2d(128, num_classes, 1)
        self.num_classes = num_classes

    def forward(self, x4, x8, x16):
        x16 = self.conv16(x16)
        x8 = self.fusion_8_16(x16, x8)
        x8 = self.conv8(x8)
        x4 = self.fusion_4_8(x8, x4)
        return self.classifier(x4)


class RegSeg_gate0(nn.Module):

    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.body = RegSegBody2([[1], [1, 2]]+4*[[1, 4]]+7*[[1, 14]])
        self.decoder: nn.Module = Decode_V2(
            out_ch, self.body.channels())

    def forward(self, x: torch.Tensor):
        input_shape = x.shape[-2:]
        x = self.stem(x)
        f = self.body(x)
        x = self.decoder(*f)
        x = F.interpolate(x, size=input_shape,
                          mode='bilinear', align_corners=False)
        return x

class RegSeg_gate0_6(RegSeg_gate0):
    C12_TO_C5 = {
        1: [1, 2],
        2: [4, 12],
        3: [5, 6, 9, 10],
        4: [7, 8, 11],
        5: [3,],
    }

    def forward(self, x: torch.Tensor):
        o = super().forward(x)
        return transform_label(o, self.C12_TO_C5)