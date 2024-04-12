import torch
import torch.nn as nn


class channel_wise_attention(nn.Module):
    def __init__(self, num_channels, reduce):
        super(channel_wise_attention, self).__init__()
        self.num_channels = num_channels
        self.reduce = reduce
        self.mean = nn.AvgPool2d((1, 1000))

        self.fc = nn.Sequential(nn.Linear(self.num_channels, self.num_channels // self.reduce),
                                nn.Tanh(),
                                nn.Linear(self.num_channels // self.reduce, self.num_channels))

        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        n, b, c, l = x.shape
        # x -> (N, B, C, L)
        feature_map = self.mean(x).permute(0, 1, 3, 2)
        feature_map_fc = self.fc(feature_map)

        # softmax
        v = self.softmax(feature_map_fc)

        # channel_wise_attention
        v = v.view(-1, self.num_channels)
        v = [v] * (b * l)
        tmp = torch.cat(v, dim=1)

        vr = torch.reshape(tmp, [-1, b, c, l])
        channel_wise_attention_fm = x * vr

        return channel_wise_attention_fm


# i = torch.rand(3, 1, 22, 1000)
# net = channel_wise_attention(22, 2)
# o = net(i)
