import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable


# 参考链接02 ：https://zhuanlan.zhihu.com/p/163749765
# 参考链接01 ：https://zhuanlan.zhihu.com/p/163749765
# 相关距离的取之越大，表示图像之间的相似成都越高；反之，则表示图像之间的相似程度越低


class localCrossCorrelation(nn.Module):
    """
    local (over window) normalization cross correlation (square)
    """

    def __init__(self, win=9, eps=1e-5):
        super(localCrossCorrelation, self).__init__()
        self.win = win
        self.eps = eps

    def forward(self, I, J):
        """
        I.shape == (N, 1, H, W)
        J.shape == (N, 1, H, W)
        """
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J

        filters = Variable(torch.ones(1, 1, 1, self.win))
        if I.is_cuda:
            filters = filters.cuda()
        padding = (0, self.win // 2)

        I_sum = F.conv2d(I, filters, stride=1, padding=padding)
        J_sum = F.conv2d(J, filters, stride=1, padding=padding)
        I2_sum = F.conv2d(I2, filters, stride=1, padding=padding)
        J2_sum = F.conv2d(J2, filters, stride=1, padding=padding)
        IJ_sum = F.conv2d(IJ, filters, stride=1, padding=padding)

        u_I = I_sum / self.win
        u_J = J_sum / self.win

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * self.win
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * self.win
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * self.win

        cc = cross * cross / (I_var * J_var + self.eps)
        lcc = -1.0 * torch.mean(cc) + 1

        return lcc


class globalCrossCorrelation(nn.Module):
    def __init__(self):
        super(globalCrossCorrelation, self).__init__()

    def forward(self, I, J):
        """
        I.shape == (N, C, H, W)
        J.shape == (N, C, H, W)
        """
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J

        # average value
        I_avg, J_avg = I.mean(), J.mean()
        I2_avg, J2_avg = I2.mean(), J2.mean()
        IJ_avg = IJ.mean()

        cross = IJ_avg - I_avg * J_avg
        I_var = I2_avg - I_avg.pow(2)
        J_var = J2_avg - J_avg.pow(2)

        cc = cross / (I_var.sqrt() * J_var.sqrt() + np.finfo(float).eps)

        return -1.0 * cc + 1


if __name__ == '__main__':
    x1 = torch.rand(1, 1, 22, 1125)
    x2 = torch.rand(1, 1, 22, 1125)

    lcc = localCrossCorrelation(win=3)
    gcc = globalCrossCorrelation()

    lcc_val = lcc(x1, x2)
    gcc_val = gcc(x1, x2)
    print(gcc_val)
