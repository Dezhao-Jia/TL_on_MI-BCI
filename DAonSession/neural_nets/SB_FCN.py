import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import warnings
warnings.filterwarnings("ignore")


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

        return x * w


class SB_FCN(nn.Module):
    def __init__(self, in_chans, n_classes, n_filters_time=48, n_filters_spat=24, drop_prob=0.5):
        super(SB_FCN, self).__init__()
        self.feat_mode = nn.Sequential(nn.Conv2d(1, n_filters_spat // 2, kernel_size=(1, 150), padding=(0, 150 // 2)),
                                       nn.Conv2d(n_filters_spat // 2, n_filters_spat, kernel_size=(in_chans, 1),
                                                 groups=2),
                                       nn.BatchNorm2d(n_filters_spat),
                                       nn.ELU(),
                                       nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
                                       nn.Dropout2d(p=drop_prob),
                                       nn.Conv2d(n_filters_spat, n_filters_time, kernel_size=(1, 25),
                                                 padding=(0, 25 // 2), groups=2),
                                       nn.BatchNorm2d(n_filters_time),
                                       nn.ELU(),
                                       nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)),
                                       nn.Dropout2d(p=drop_prob),
                                       )

        self.aten_mode = SEBlock(n_filters_time)
        self.cls_mode = nn.Sequential(nn.Conv2d(n_filters_time, n_classes, kernel_size=(1, 17), groups=2, bias=True),
                                      nn.LogSoftmax(dim=1),
                                      )

    def forward(self, x):
        feat = self.spat_mode(x)
        feat_ = self.temp_mode(feat)
        res = self.cls_mode(feat_)
        res = rearrange(res, 'n c h w -> n (c h w)')

        return feat_, res


if __name__ == '__main__':
    x = torch.rand(1, 1, 22, 1125)
    net = SB_FCN(in_chans=22, n_classes=4)
    print(net)
    _, o = net(x)

    print(o.shape)
