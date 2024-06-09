import torch
import numpy as np

def cls_split(in_data, in_labels):
    d01 = [in_data[i] for i, k in enumerate(in_labels) if k == 0]
    d01 = np.stack(d01, axis=0)
    l01 = np.ones(d01.shape[0]) * 0
    d02 = [in_data[i] for i, k in enumerate(in_labels) if k == 1]
    d02 = np.stack(d02, axis=0)
    l02 = np.ones(d02.shape[0]) * 1
    d03 = [in_data[i] for i, k in enumerate(in_labels) if k == 2]
    d03 = np.stack(d03, axis=0)
    l03 = np.ones(d03.shape[0]) * 2
    d04 = [in_data[i] for i, k in enumerate(in_labels) if k == 3]
    d04 = np.stack(d04, axis=0)
    l04 = np.ones(d04.shape[0]) * 3

    data = [d01, d02, d03, d04]
    labels = [l01, l02, l03, l04]

    return data, labels


def exchange_length(in_data):
    data = []
    length = in_data.shape[-1]
    stride = length // 3
    for i in range(len(in_data) // 3):
        d01 = [in_data[i * 3, :, :, 0:stride], in_data[i * 3 + 1, :, :, stride:2 * stride],
               in_data[i * 3 + 2, :, :, 2 * stride:]]
        d01 = np.concatenate(d01, axis=-1)
        data.append(d01)
        d02 = [in_data[i * 3 + 1, :, :, 0:stride], in_data[i * 3, :, :, stride:2 * stride],
               in_data[i * 3 + 2, :, :, 2 * stride:]]
        d02 = np.concatenate(d02, axis=-1)
        data.append(d02)
        d03 = [in_data[i * 3 + 2, :, :, 0:stride], in_data[i * 3, :, :, stride:2 * stride],
               in_data[i * 3 + 1, :, :, 2 * stride:]]
        d03 = np.concatenate(d03, axis=-1)
        data.append(d03)

    data = np.stack(data, axis=0)

    return data

# 数据增强 - length_change(分割重组)
def length_change(in_data, in_labels):
    res_data = []
    res_labels = []
    data, labels = cls_split(in_data, in_labels)
    for i in range(len(data)):
        tmp_d = exchange_length(data[i])
        tmp_l = np.ones(tmp_d.shape[0]) * i
        res_data.append(tmp_d)
        res_labels.append(tmp_l)
    res_data.append(in_data)
    res_labels.append(in_labels)
    res_data = np.concatenate(res_data, axis=0)
    res_labels = np.concatenate(res_labels, axis=0)

    return res_data, res_labels


def exchange_band(in_data):
    data = []
    for i in range(len(in_data) // 2):
        d01 = [in_data[i * 2, 0, :, :], in_data[i * 2 + 1, 1, :, :], in_data[i * 2, 2, :, :],
               in_data[i * 2 + 1, 3, :, :]]
        d01 = np.stack(d01, axis=0)
        data.append(d01)
        d02 = [in_data[i * 2 + 1, 0, :, :], in_data[i * 2, 1, :, :], in_data[i * 2 + 1, 2, :, :],
               in_data[i * 2, 3, :, :]]
        d02 = np.stack(d02, axis=0)
        data.append(d02)
    data = np.stack(data, axis=0)

    return data

# 数据增强 - 交换频带
def band_change(in_data, in_labels):
    new_data = []
    new_labels = []
    data, labels = cls_split(in_data, in_labels)
    for i in range(len(data)):
        tmp_d = exchange_band(data[i])
        tmp_l = np.ones(tmp_d.shape[0]) * i
        new_data.append(tmp_d)
        new_labels.append(tmp_l)
    new_data.append(in_data)
    new_labels.append(in_labels)
    new_data = np.concatenate(new_data, axis=0)
    new_labels = np.concatenate(new_labels, axis=0)

    return new_data, new_labels

# 数据增强 - 滑动窗口
def length_slide(in_data, in_labels):
    res_data = []
    res_labels = []
    for i in range(2):
        res_data.append(in_data[:, :, :, i * 500:i * 500 + 500])
        res_labels.append(in_labels)

    res_data = np.concatenate(res_data, axis=0)
    res_data = torch.from_numpy(res_data)
    res_labels = np.concatenate(res_labels, axis=0)
    res_data = res_data.detach().numpy()

    return res_data, res_labels
