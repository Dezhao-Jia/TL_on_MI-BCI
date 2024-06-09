import torch
import numpy as np
import torch.nn as nn

# 向数据中添加 白噪声
def add_white_noise(data, noise_level):
    """
    data : (1 * w)
    """
    noise = np.random.normal(0, noise_level, data.shape)
    noise_data = data + noise

    return noise_data

# 向数据中添加 高斯噪声
def add_gaussian_noise(data, mean=0, std=1):
    """
        data : (1 * w)
        The default parameters corresponds to the case of Gaussian white noise.
    """
    noise = np.random.normal(mean, std, data.shape)
    noise_data = data + noise

    return noise_data

# 向数据中添加 高斯噪声
class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0, std=0.1):
        super(AddGaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, data):
        noise = torch.normal(mean=self.mean, std=self.std, size=data.shape)
        noise_data = noise + data

        return noise_data
