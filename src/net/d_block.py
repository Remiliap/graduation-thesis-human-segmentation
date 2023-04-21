from torch import nn
import torch
import torch.nn.functional as F


class SEModule(nn.Module):
    @staticmethod
    def avg_pool(c):
        return nn.AdaptiveAvgPool2d(1), c

    @staticmethod
    def max_pool(c):
        return nn.AdaptiveMaxPool2d(1), c

    class Avg_max_pool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.max_p = nn.AdaptiveMaxPool2d(1)
            self.avg_p = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            m = self.max_p(x)
            a = self.avg_p(x)
            return torch.cat([m, a], 1)

    @staticmethod
    def avg_max_pool(c):
        return SEModule.Avg_max_pool(), 2*c

    def __init__(self, channels_in, channels_se, pool=avg_pool):
        super().__init__()
        self.pool,channels_pool = pool(channels_in)

        self.conv1 = nn.Conv2d(channels_pool, channels_se, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels_se, channels_in, 1, bias=True)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
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
