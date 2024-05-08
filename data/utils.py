import numpy as np
from einops import rearrange
from scipy.linalg import fractional_matrix_power

def EA(x):
    if len(x.shape) == 4:
        x = rearrange(x, 'n c h w -> (n c) h w')
    xt = np.transpose(x, axes=(0, 2, 1))
    E = np.matmul(x, xt)
    R = np.mean(E, axis=0)
    R_mat = fractional_matrix_power(R, -0.5)
    x_ = np.einsum('n c s,r c -> n r s', x, R_mat).astype('float32')

    x_ = rearrange(x_, 'n h w -> n 1 h w')

    return x_

def zero_score(data_array):
    """
    To normalize the input data by zero_score method.
    data : ( n * c * l )
    """
    res = []
    for data in data_array:
        tmp = []
        mean = np.mean(data, axis=-1)
        std = np.std(data, axis=-1)

        for i in range(len(data)):
            tmp.append((data[i] - mean[i]) / std[i])
        tmp = np.stack(tmp, axis=0)

        res.append(tmp)
    res = np.stack(res, axis=0)

    return res
