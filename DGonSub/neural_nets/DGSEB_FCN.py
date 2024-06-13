import torch
import torch.nn as nn

from einops import rearrange
from attention_model.SENet import SEBlock


class Muti_head_Classifier(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=2, bias=True):
        super(Muti_head_Classifier, self).__init__()
        self.cls01 = nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups, bias=bias)
        self.cls02 = nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups, bias=bias)
        self.cls03 = nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups, bias=bias)
        self.weight = torch.stack([self.cls01.weight, self.cls02.weight, self.cls03.weight], dim=1)

    def forward(self, x):
        x1 = x.clone()
        x1[:, :, :, 0:2] = 0
        o1 = self.cls01(x1)
        x2 = x.clone()
        x2[:, :, :, 0] = 0
        x2[:, :, :, -1] = 0
        o2 = self.cls02(x2)
        x3 = x.clone()
        x3[:, :, :, -2:] = 0
        o3 = self.cls03(x3)

        return o1 + o2 + o3


class DGSEB_FNN(nn.Module):
    def __init__(self, in_chans, n_classes, n_filters_time=48, n_filters_spat=24, drop_prob=0.5):
        super(DGSEB_FNN, self).__init__()
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

        self.attn_mode = SEBlock(n_filters_time)
        self.b_mode = Muti_head_Classifier(n_filters_time, n_classes, kernel_size=(1, 8), groups=2, bias=True)
        self.g_mode = nn.Conv2d(n_filters_time, n_classes, kernel_size=(1, 8), groups=2, bias=True)
        self.a_func = nn.LogSoftmax(dim=1)

    def forward(self, x):
        feat = self.feat_mode(x)
        feat = self.attn_mode(feat)
        res_b = self.b_mode(feat)
        res_b = self.a_func(res_b)
        res_b = rearrange(res_b, 'n c h w -> n (c h w)')
        res_g = self.g_mode(feat)
        res_g = self.a_func(res_g)
        res_g = rearrange(res_g, 'n c h w -> n (c h w)')

        return feat, res_b, res_g


if __name__ == '__main__':
    x = torch.rand(1, 1, 22, 1125)
    mode = DGSEB_FNN(in_chans=22, n_classes=4)
    f, o1, o2 = mode(x)

    print(mode.b_mode.weight.shape, type(mode.b_mode.weight))
    print(mode.g_mode.weight.shape, type(mode.g_mode.weight))

    print(mode)
    print(f.shape)
    print('val of bias res {}, shape of bias res {}'.format(o1, o1.shape))
    print('val of gene res {}, shape of gene res {}'.format(o2, o2.shape))

