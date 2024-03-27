import torch
import torch.nn as nn

from loss_funcs.grad_reverse import grad_reverse
from DAonSession.neural_nets.SB_FCN import SB_FCN


class Net(nn.Module):
    def __init__(self, drop_prob=0.5, reverse=False):
        super(Net, self).__init__()
        self.feat_model = None
        self.cls_pred = None
        self.dis_pred = None
        self.reverse = reverse
        self.feat_model = nn.Sequential(SB_FCN(in_chans=22, n_classes=4, drop_prob=drop_prob).feat_mode,
                                        SB_FCN(in_chans=22, n_classes=4, drop_prob=drop_prob).aten_mode,
                                        )
        self.cls_pred = nn.Sequential(nn.Conv2d(48, 4, kernel_size=(1, 8)),
                                      nn.Flatten(),
                                      )
        self.dis_pred = nn.Sequential(nn.Conv2d(48, 2, kernel_size=(1, 8)),
                                      nn.Flatten(),
                                      )

    def forward(self, x):
        feat = self.feat_model(x)
        cls_pred = self.cls_pred(feat)
        if self.reverse:
            feat = grad_reverse(feat, lambd=0.2)
        dis_pred = self.dis_pred(feat)

        return feat, cls_pred, dis_pred


if __name__ == '__main__':
    x = torch.rand(1, 1, 22, 1125)
    net = Net()
    feat, cls_pred, dis_pred = net(x)

    print(feat.shape, cls_pred.shape, dis_pred.shape)
