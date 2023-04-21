import torch.nn as nn
import torch
from torchvision.models.vgg import VGG
from torchvision import models


class VGG16_rm_fc(VGG):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5) -> None:
        vgg16 = models.vgg16(weights=models.VGG16_Weights)
        super().__init__(vgg16.features, num_classes, init_weights, dropout)
        del vgg16.classifier
        del self.classifier
        self.load_state_dict(vgg16.state_dict())

        

        for name, param in self.named_parameters():
            print(name, param.size())

    def forward(self, x: torch.Tensor):
        # output为每个maxpooling的输出
        output = []
        # 获取每个maxpooling层输出的特征图
        for layer in self.features.children():
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                output.append(x)
        return output


class FCN8s(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        self.pretrained_net = VGG16_rm_fc(2)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=1,
                               stride=1, padding=0, dilation=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=1,
                               stride=1, padding=0, dilation=1)
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        # 使输出类别等于分类数
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output: list[torch.Tensor] = self.pretrained_net(x)

        output.reverse()
        x5 = output[0]
        x4 = output[1]
        x3 = output[2]

        score = self.relu(self.conv6(x5))
        score = self.relu(self.conv7(score))
        score = self.relu(self.deconv1(x5))

        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))

        score = self.bn2(score + x3)

        score = self.bn3(self.relu(self.deconv3(score)))

        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))

        score = self.classifier(score)

        return score
