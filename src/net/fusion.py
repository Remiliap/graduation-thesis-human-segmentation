from torch import nn
import torch
import torch.nn.functional as F

class Fusion(nn.Module):
    def __init__(self, h_ch, l_ch, out_ch, scale_factor=2) -> None:
        super().__init__()
        if scale_factor < 2 or scale_factor % 2 != 0:
            raise RuntimeError(
                "Scale factor can't be {}".format(scale_factor))
        self.h_self_conv = nn.Sequential(
            nn.Conv2d(h_ch, h_ch, kernel_size=3, padding=1, groups=h_ch),
            nn.BatchNorm2d(h_ch),
            nn.Conv2d(h_ch, h_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.h_fusion_conv = nn.Sequential(
            nn.Conv2d(h_ch, h_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(h_ch),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear"),
            nn.Sigmoid()
        )

        self.l_self_conv = nn.Sequential(
            nn.Conv2d(l_ch, l_ch, kernel_size=3, padding=1, groups=l_ch),
            nn.BatchNorm2d(l_ch),
            nn.Conv2d(l_ch, h_ch, kernel_size=1),
        )
        self.l_fusion_conv = nn.Sequential(
            nn.Conv2d(l_ch, h_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(h_ch),
        )
        if scale_factor > 2:
            scale_factor /= 2
            self.l_fusion_conv.append(
                nn.AvgPool2d(scale_factor, scale_factor)
            )

        self.out_conv = nn.Sequential(
            nn.Conv2d(h_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, h: torch.Tensor, l: torch.Tensor):
        l_self = self.l_self_conv(l)
        l_fusion = self.l_fusion_conv(l)
        h_self = self.h_self_conv(h)
        h_fusion = self.h_fusion_conv(h)
        l = l_self * h_fusion
        h = h_self * l_fusion
        h = F.interpolate(h, l.shape[-2:], mode="bilinear")
        return self.out_conv(h+l)
