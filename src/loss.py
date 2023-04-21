import torch
import torch.nn.functional as F


def filter(input: torch.Tensor, window: torch.Tensor):
    channel = input.size(1)
    window_size = window.size(0)
    window = window.view((1, 1, 1, -1)).expand(channel, -1, -1, -1)

    if input.size(-1) < window_size:
        out = torch.mean(input, -1, keepdim=True)
    else:
        out = F.conv2d(input, window, groups=channel)

    if input.size(-2) < window_size:
        out = torch.mean(input, -2, keepdim=True)
    else:
        out = F.conv2d(out, window.transpose(2, 3), groups=channel)
    return out


def gaussian(window_size, sigma):
    x = torch.arange(window_size, dtype=torch.float32)

    gauss: torch.Tensor = torch.exp(
        -((x - window_size//2)**2) /
        (2 * sigma**2))
    return gauss/gauss.sum()


def activate(input: torch.Tensor):
    """根据图像通道数选择激活函数"""
    if input.size(1) > 1:
        return torch.softmax(input, 1)
    else:
        return torch.sigmoid(input)


class Soft_dice_loss(torch.nn.Module):
    """
    args:
        input: (Batch,Channel,Width,Height)
        target: shape same as input

    intersection(x,y) = sum(x * y)
    union(x,y) = sum(x + y)

    loss_(input,target,Batch,Channel) = 1 -
        ( 2*intersection(input[Batch,Channel], target[Batch,Channel]) + smooth ) /
        ( union(input[Batch,Channel]^p, target[Batch,Channel]^p) + smooth )

    loss = loss_.mean() if reduction == 'mean'
    loss = loss_.sum() if reduction == 'sum'
    """

    def __init__(self,
                 p=1,
                 smooth=1, reduction='mean', activated=False):
        super().__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
        self.activated = activated

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if not self.activated:
            input = activate(input)
        return self.fn(input, target)

    def fn(self, input: torch.Tensor, target: torch.Tensor):
        intersection = torch.sum((input*target), dim=(2, 3))
        union = torch.sum(input.pow(self.p)+target.pow(self.p), dim=(2, 3))

        loss = 1 - (2*intersection.flatten()+self.smooth) / \
            (union.flatten()+self.smooth)

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class Focal_loss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', mul_class=True) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.mul_class = mul_class

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.mul_class:
            ce_loss = F.cross_entropy(input, target, reduction='none')
        else:
            ce_loss = F.binary_cross_entropy_with_logits(
                input, target, reduction='none')

        pt = torch.exp(-ce_loss)

        loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, sigma=1.5, dynamic_range=(0, 1), activated=False) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.dynamic_range = dynamic_range
        self.activated = activated

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if not self.activated:
            input = activate(input)
        _, _, ssim = self.fn(input, target)
        ssim = (1-ssim).mean((2, 3))
        loss = ssim.mean()
        return loss

    def fn(self, img1: torch.Tensor, img2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if img1.shape != img2.shape:
            raise RuntimeError("Input img must have same shape.")

        # 图像动态范围
        dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]

        C1 = (0.01*dynamic_range)**2
        C2 = (0.03*dynamic_range)**2
        # C3 = C2/2

        window = gaussian(self.window_size, self.sigma).to(img1.device)

        mu1 = filter(img1, window)
        mu2 = filter(img2, window)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = filter(img1 * img1, window) - mu1_sq
        sigma2_sq = filter(img2 * img2, window) - mu2_sq
        sigma12 = filter(img1 * img2, window) - mu1_mu2

        l_map = (2*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)
        cs_map = (2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
        ssim_map = l_map * cs_map

        return l_map, cs_map, ssim_map


class MS_SSIM(torch.nn.Module):
    l5_weights = torch.tensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float32)

    def __init__(self, weights: torch.Tensor, window_size=11,
                 normalize=False, dynamic_range=(0, 1), activated=False
                 ) -> None:
        super().__init__()

        self.window_size = window_size
        self.normalize = normalize
        self.ssim = SSIM(self.window_size,
                         dynamic_range=dynamic_range, activated=True)
        self.weights = weights
        self.activated = activated

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if not self.activated:
            input = activate(input)
        msssim = self.fn(input, target)
        loss = (1-msssim).mean()
        return loss

    def fn(self, img1: torch.Tensor, img2: torch.Tensor):
        self.weights = self.weights.to(img1.device)

        mcs = []

        for _ in range(self.weights.size(0)-1):
            l, cs, ssim = self.ssim.fn(img1, img2)

            mcs.append(cs.mean((2, 3)))

            padding = [s % 2 for s in img1.shape[2:]]
            img1 = F.avg_pool2d(img1, (2, 2), padding=padding)
            img2 = F.avg_pool2d(img2, (2, 2), padding=padding)
        l, cs, ssim = self.ssim.fn(img1, img2)

        mcs = torch.stack(mcs + [ssim.mean((2, 3))], 2)

        # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        if self.normalize:
            mcs = (mcs + 1) / 2

        output = torch.prod(mcs**self.weights, 2)

        return output
