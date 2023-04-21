from torch import nn
import torch
import torch.nn.functional as F

from net.align import AlignModule

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if apply_act:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
class Decode_align(nn.Module):
    def __init__(self, num_classes: int, channels: list[int]):
        super().__init__()

        channels4, channels8, channels16 = channels["4"], channels["8"], channels["16"]

        self.head16 = ConvBnAct(channels16, 256, kernel_size=1)
        self.head8 = ConvBnAct(channels8, 256, kernel_size=1)
        self.head4 = ConvBnAct(channels4, 16, kernel_size=1)

        self.up16 = AlignModule(256, 256, 128, 128)
        self.up8 = AlignModule(128, 16, 64)

        self.conv8 = ConvBnAct(256, 128, kernel_size=3, padding=1)
        self.conv4 = ConvBnAct(128+16, 128, kernel_size=3, padding=1)

        self.classifier = nn.Conv2d(128, num_classes, 1)

        self.num_classes = num_classes

    def forward(self, x4, x8, x16):
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)

        x16 = self.up16(x8, x16)

        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = self.up8(x4, x8)

        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4