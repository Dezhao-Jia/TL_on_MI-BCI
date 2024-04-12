import torch
from torch import nn
import torch.nn.functional as F

# 网络原型 —— https://github.com/frechele/SENet-PyTorch/tree/master


# Squeeze and Excitation Block Module -- SEBlock
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=2):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(nn.Conv2d(channels, channels // reduction, kernel_size=(1, 1), bias=False),
                                nn.ReLU(),
                                nn.Conv2d(channels // reduction, channels, kernel_size=(1, 1), bias=False),
                                )

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)  # Squeeze
        w = self.fc(w)
        w = torch.sigmoid(w)

        return x * w  # Scale and add bias


# Residual Block with SEBlock
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.se_block = SEBlock(channels)

    def forward(self, x):
        path = self.se_block(x)
        path = x + path

        return F.relu(path)


# Network Module
class SENet(nn.Module):
    def __init__(self, in_channel, blocks=1):
        super(SENet, self).__init__()
        self.res_blocks = nn.Sequential(*[ResBlock(in_channel) for _ in range(blocks - 1)])

        self.out_conv = nn.Sequential(nn.Conv2d(in_channel, 1, 1, padding=0, bias=False),
                                      nn.BatchNorm2d(1),
                                      nn.ReLU()
                                      )

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.out_conv(x)
        return x
