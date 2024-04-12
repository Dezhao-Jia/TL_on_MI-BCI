import torch
import torch.nn as nn


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4

        self.pro_conv = nn.Sequential(nn.Conv2d(in_channels, inter_channels, (1, 7), padding=(0, 3), bias=False),
                                      nn.BatchNorm2d(inter_channels),
                                      nn.ReLU())

        self.sc = CAM_Module(inter_channels)

        self.last_conv = nn.Sequential(nn.Conv2d(inter_channels, out_channels, (1, 5), padding=(0, 2), bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.3, False),
                                       )

    def forward(self, x):
        feat = self.pro_conv(x)
        sc_feat = self.sc(feat)
        sc_out = self.last_conv(sc_feat)

        return sc_out
