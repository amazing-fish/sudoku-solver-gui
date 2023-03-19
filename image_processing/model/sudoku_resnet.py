import torch
import torch.nn as nn
from torchvision.models import resnet101


class SudokuResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SudokuResNet, self).__init__()

        # 加载预训练的ResNet-101模型
        base_model = resnet101(pretrained=pretrained)
        layers = list(base_model.children())[:-1]  # 移除最后一个全连接层

        # 添加自定义的全连接层，输出为9 * 9 * 10
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(512, 9 * 9 * 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
