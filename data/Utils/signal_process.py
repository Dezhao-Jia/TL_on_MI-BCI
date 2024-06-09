import numpy as np
import pandas as pd

from scipy import signal

def standardize_data(datasets, factor_new=0.001, init_block_size=None, eps=1e-4):
    for i in range(datasets.shape[0]):
        datasets[i] = exponential_running_standardize(datasets[i].T, factor_new=0.001,
                                                      init_block_size=init_block_size, eps=1e-4).T

    return datasets

def exponential_running_standardize(data, factor_new=0.001, init_block_size=None, eps=1e-4):
    """
    Perform exponential running standardization.
    Parameters
    ----------
    data: 2d_array (time, channels)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.

    Returns
    -------
    standardized: 2d_array (time, channels)
        Standardized data.
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_block_standardized = (
                                          data[0:init_block_size] - init_mean
                                  ) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized

    return standardized

# 带通滤波
def butter_bandpass(low_cut, high_cut, fs):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = signal.butter(N=3, Wn=[low, high], btype='bandpass')

    return b, a

def butter_bandpass_filter(data, low_cut, high_cut, fs):
    b, a = butter_bandpass(low_cut, high_cut, fs)
    y = signal.filtfilt(b, a, data, axis=2)

    return y

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
