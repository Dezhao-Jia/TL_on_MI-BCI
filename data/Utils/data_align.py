import numpy as np
from einops import rearrange
from scipy.linalg import fractional_matrix_power

def EA(x):
    """
    The Euclidean space alignment approach for EEG data from Wen chao Liu.

    Arg:
        x:The input data,shape of NxCxS

    Return:
        The aligned data.
    """
    if len(x.shape) == 4:
        x = rearrange(x, 'n c h w -> (n c) h w')
    xt = np.transpose(x, axes=(0, 2, 1))
    E = np.matmul(x, xt)
    R = np.mean(E, axis=0)
    R_mat = fractional_matrix_power(R, -0.5)
    x_ = np.einsum('n c s,r c -> n r s', x, R_mat).astype('float32')

    x_ = rearrange(x_, 'n h w -> n 1 h w')

    return x_
