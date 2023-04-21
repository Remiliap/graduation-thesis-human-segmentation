from torch import nn
import torch
import torch.nn.functional as F

from net.d_block import DBlock

from net.dec_gate2 import Decode_gate2
from net.dec_gate import Decode_gate
from net.dec_align import Decode_align


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


class RegSeg_dp(nn.Module):
    decoders = [Decode_align, Decode_gate, Decode_gate2]

    def __init__(self, in_ch, out_ch, dec=0) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.body = RegSegBody2([[1], [1, 2]]+4*[[1, 4]]+7*[[1, 14]])
        self.decoder: nn.Module = RegSeg_dp.decoders[dec](
            out_ch, self.body.channels())

    def forward(self, x: torch.Tensor):
        input_shape = x.shape[-2:]
        x = self.stem(x)
        f = self.body(x)
        x = self.decoder(*f)
        x = F.interpolate(x, size=input_shape,
                          mode='bilinear', align_corners=False)
        return x
