from torch import nn
import torch
import torch.nn.functional as F


class Fusion_gate(nn.Module):
    def __init__(self, h_ch, l_ch, fusion_ch) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(h_ch, fusion_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_ch),
            nn.Sigmoid()
        )
        nn.init.orthogonal_(self.gate[0].weight)

        g = l_ch // 4
        self.l_conv = nn.Sequential(
            nn.Conv2d(l_ch, l_ch, kernel_size=3,
                      padding=1, groups=g, bias=False),
            nn.BatchNorm2d(l_ch),
            nn.Conv2d(l_ch, fusion_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_ch)
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_ch, 2*fusion_ch,
                      kernel_size=3, padding=1, groups=32),
            nn.BatchNorm2d(2*fusion_ch),
            nn.ReLU(True),
            nn.Conv2d(2*fusion_ch, h_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(h_ch)
        )

    def forward(self, h: torch.Tensor, l: torch.Tensor):
        h_origin = h
        h = self.gate(h)
        l = self.l_conv(l)

        h = F.interpolate(h, l.size()[-2:], mode="bilinear")
        h_origin = F.interpolate(h_origin, l.size()[-2:], mode="bilinear")

        h = l * h
        h = self.fusion_conv(h)

        return F.relu(h+h_origin, True)


class Decode_gate(nn.Module):
    def __init__(self, num_classes: int, channels: list[int]):
        super().__init__()

        channels4, channels8, channels16 = channels["4"], channels["8"], channels["16"]

        self.conv16 = nn.Sequential(
            nn.Conv2d(channels16, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )
        self.fusion_8_16 = Fusion_gate(256, channels8, 64)
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )
        self.fusion_4_8 = Fusion_gate(128, channels4, 64)

        self.classifier = nn.Conv2d(128, num_classes, 1)
        self.num_classes = num_classes

    def forward(self, x4, x8, x16):
        x16 = self.conv16(x16)
        x8 = self.fusion_8_16(x16, x8)
        x8 = self.conv8(x8)
        x4 = self.fusion_4_8(x8, x4)
        return self.classifier(x4)
