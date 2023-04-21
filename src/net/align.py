from torch import nn
import torch
import torch.nn.functional as F


class AlignModule(nn.Module):
    def __init__(self, h_inch, l_inch, h_outch=None, l_outch=None):
        super(AlignModule, self).__init__()
        if h_outch == None:
            self.down_h = nn.Sequential()
            h_outch = h_inch
        else:
            self.down_h = nn.Conv2d(h_inch, h_outch, 1, bias=False)

        if l_outch == None:
            self.down_l = nn.Sequential()
            l_outch = l_inch
        else:
            self.down_l = nn.Conv2d(l_inch, l_outch, 1, bias=False)

        self.flow_make = nn.Conv2d(
            h_outch+l_outch, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, low: torch.Tensor, height: torch.Tensor):
        h_feature_orign = height
        size = low.size()[2:]

        low = self.down_l(low)
        height = self.down_h(height)

        height = F.interpolate(
            height, size=size, mode="bilinear", align_corners=False)
        
        flow = self.flow_make(torch.cat([height, low], 1))
        
        height = self.flow_warp(h_feature_orign, flow, size=size)

        return height

    @staticmethod
    def flow_warp(inputs: torch.Tensor, flow: torch.Tensor, size: tuple[int, int]):
        out_h, out_w = size  # 对应高分辨率的low-level feature的特征图尺寸
        n, c, h, w = inputs.size()  # 对应低分辨率的high-level feature的4个输入维度

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(
            inputs).to(inputs.device)
        # 从-1到1等距离生成out_h个点，每一行重复out_w个点，最终生成(out_h, out_w)的像素点
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        # 生成w的转置矩阵
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        # 展开后进行合并
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        # grid指定由input空间维度归一化的采样像素位置，其大部分值应该在[ -1, 1]的范围内
        # 如x=-1,y=-1是input的左上角像素，x=1,y=1是input的右下角像素。
        # 具体可以参考《Spatial Transformer Networks》，下方参考文献[2]
        output = F.grid_sample(inputs, grid)
        return output

def flow_warp(*inputs: torch.Tensor, flow: torch.Tensor, size: tuple[int, int]):
    out_h, out_w = size

    input0 = inputs[0]
    n, c, h, w = input0.size()

    norm = torch.tensor([[[[out_w, out_h]]]]).type_as(
        input0).to(input0.device)
    # 从-1到1等距离生成out_h个点，每一行重复out_w个点，最终生成(out_h, out_w)的像素点
    w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
    # 生成w的转置矩阵
    h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
    # 展开后进行合并
    grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
    grid = grid.repeat(n, 1, 1, 1).type_as(input0).to(input0.device)
    grid = grid + flow.permute(0, 2, 3, 1) / norm

    return [F.grid_sample(input, grid)
            for input in inputs]