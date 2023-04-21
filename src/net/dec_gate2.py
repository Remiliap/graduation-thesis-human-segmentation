from torch import nn
import torch
import torch.nn.functional as F


class Fusion_gate2(nn.Module):
    def __init__(self, h_ch, l_ch, fusion_ch, out_ch, low_gate=False) -> None:
        super().__init__()
        self.short_cut = []
        if h_ch != out_ch:
            self.short_cut = [
                nn.Conv2d(h_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
                nn.Upsample(scale_factor=2, mode="bilinear")
            ]
        self.short_cut = nn.Sequential(self.short_cut)
        

        self.h_conv = nn.Sequential(
            nn.Conv2d(h_ch, fusion_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_ch)
        )
        self.h_gate = nn.Sequential(
            nn.Conv2d(fusion_ch, 1, kernel_size=3,
                      padding=1, bias=False),
            nn.Sigmoid()
        )
        nn.init.orthogonal_(self.h_gate[0].weight)

        l_out_ch = min(l_ch, fusion_ch)

        if l_out_ch != l_ch:
            self.l_conv = nn.Sequential(
                nn.Conv2d(l_ch, l_out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(l_out_ch)
            )
        else:
            self.l_conv = nn.Sequential()

        if low_gate:
            self.l_gate = nn.Sequential(
                nn.Conv2d(l_out_ch, 1, kernel_size=3,
                          padding=1, bias=False),
                nn.Sigmoid()
            )
            nn.init.orthogonal_(self.l_gate[0].weight)
        else:
            self.l_gate = None

        fusion_ch = fusion_ch + l_out_ch

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_ch, 2*fusion_ch,
                      kernel_size=3, padding=1, groups=32),
            nn.BatchNorm2d(2*fusion_ch),
            nn.ReLU(True),
            nn.Conv2d(2*fusion_ch, h_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(h_ch)
        )

    def forward(self, h: torch.Tensor, l: torch.Tensor):
        h_short_cut = self.short_cut(h)
        h = self.h_conv(h)
        l = self.l_conv(l)

        hg = self.h_gate(h)

        h = F.interpolate(h, l.size()[-2:], mode="bilinear")

        f = torch.cat([h, l], dim=1)
        if self.l_gate != None:
            g = hg + self.l_gate(l)
        else:
            g = hg

        f = f * g
        f = self.fusion_conv(f)

        return F.relu(f+h_origin, True)


class Decode_gate2(nn.Module):
    def __init__(self, num_classes: int, channels: list[int]):
        super().__init__()

        channels4, channels8, channels16 = channels["4"], channels["8"], channels["16"]

        self.conv16 = nn.Sequential(
            nn.Conv2d(channels16, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )
        self.fusion_8_16 = Fusion_gate2(256, channels8, 64)
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )
        self.fusion_4_8 = Fusion_gate2(128, channels4, 64)

        self.classifier = nn.Conv2d(128, num_classes, 1)
        self.num_classes = num_classes

    def forward(self, x4, x8, x16):
        x16 = self.conv16(x16)
        x8 = self.fusion_8_16(x16, x8)
        x8 = self.conv8(x8)
        x4 = self.fusion_4_8(x8, x4)
        return self.classifier(x4)
