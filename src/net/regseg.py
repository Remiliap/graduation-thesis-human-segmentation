from torch import nn
import torch
import torch.nn.functional as F


class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, weight_in, weight_se):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(weight_in, weight_se, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(weight_se, weight_in, 1, bias=True)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.act1(self.conv1(y))
        y = self.act2(self.conv2(y))
        return x * y


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Shortcut, self).__init__()
        if in_channels == out_channels and stride == 1:
            self.conv = None
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor):
        x = self.conv(x) if self.conv != None else x
        return x


class YBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 dilation, group_width, stride):
        super(YBlock, self).__init__()
        groups = out_channels // group_width

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.activate = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               groups=groups, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se = SEModule(out_channels, in_channels//4)

        self.shortcut = Shortcut(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activate(x)
        x = self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activate(x + shortcut)
        return x


class DilatedConv(nn.Module):
    def __init__(self, input_channel: int, dilations: list[int],
                 group_width: int, stride: int, bias: bool):
        super().__init__()
        num_splits = len(dilations)

        assert (input_channel % num_splits == 0)
        conv_channel = input_channel // num_splits

        assert (conv_channel % group_width == 0)
        groups = conv_channel // group_width

        convs = []
        for d in dilations:
            convs.append(nn.Conv2d(conv_channel, conv_channel, 3, padding=d,
                         dilation=d, stride=stride, bias=bias, groups=groups))
        self.convs = nn.ModuleList(convs)

        self.num_splits = num_splits

    def forward(self, x):
        x = torch.tensor_split(x, self.num_splits, dim=1)
        res = []
        for i in range(self.num_splits):
            res.append(self.convs[i](x[i]))
        return torch.cat(res, dim=1)


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dilations, group_width, stride, attention="se"):
        super().__init__()
        groups = out_channels//group_width

        self.activate = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        if len(dilations) == 1:
            dilation = dilations[0]
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                                   groups=groups, padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = DilatedConv(
                out_channels, dilations, group_width=group_width, stride=stride, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if attention == "se":
            self.se = SEModule(out_channels, in_channels//4)
        elif attention == "se2":
            self.se = SEModule(out_channels, out_channels//4)
        else:
            self.se = None

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential()
            if stride != 1:
                self.shortcut.append(nn.AvgPool2d(2, 2, ceil_mode=True))
            self.shortcut.append(Shortcut(in_channels, out_channels, 1))
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activate(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activate(x + shortcut)
        return x


class RegSegBody(nn.Module):
    def __init__(self, dilations: list[list[int]]):
        super().__init__()
        group_width = 16
        attention = "se"
        self.stage4 = DBlock(32, 48, [1], group_width, 2, attention)
        self.stage8 = nn.Sequential(
            DBlock(48, 128, [1], group_width, 2, attention),
            DBlock(128, 128, [1], group_width, 1, attention),
            DBlock(128, 128, [1], group_width, 1, attention)
        )
        self.stage16 = nn.Sequential(
            DBlock(128, 256, [1], group_width, 2, attention),
            *[DBlock(256, 256, ds, group_width, 1, attention)
              for ds in dilations[:-1]],
            DBlock(256, 320, dilations[-1], group_width, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        return {"4": x4, "8": x8, "16": x16}

    def channels(self):
        return {"4": 48, "8": 128, "16": 320}


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


class Exp2_Decoder29(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4, channels8, channels16 = channels["4"], channels["8"], channels["16"]
        self.head16 = ConvBnAct(channels16, 256, 1)
        self.head8 = ConvBnAct(channels8, 256, 1)
        self.head4 = ConvBnAct(channels4, 16, 1)
        self.conv8 = ConvBnAct(256, 128, 3, 1, 1)
        self.conv4 = ConvBnAct(128+16, 128, 3, 1, 1)
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def forward(self, x4: torch.Tensor, x8: torch.Tensor, x16: torch.Tensor):
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x16 = F.interpolate(
            x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(
            x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4


class RegSeg(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.body = RegSegBody2([[1], [1, 2]]+4*[[1, 4]]+7*[[1, 14]])
        self.decoder = Exp2_Decoder29(out_ch, self.body.channels())

    def forward(self, x: torch.Tensor):
        input_shape = x.shape[-2:]
        x = self.stem(x)
        x = self.body(x)
        x = self.decoder(*x)
        x = F.interpolate(x, size=input_shape,
                          mode='bilinear', align_corners=False)
        return x
