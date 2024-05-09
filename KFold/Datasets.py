import mne
import torch
import numpy as np

from scipy.io import loadmat
from torch.utils.data import Dataset
from data.utils import zero_score
from data.DataSetsAugment import length_change
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')


class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __getitem__(self, item):
        data = self.data[item]
        label = self.labels[item]
        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return self.data.shape[0]


class LoadData:
    def __init__(self, sub_id, windows, k_fold=6):
        self.sub_id = sub_id
        self.windows = windows
        self.k_fold = k_fold
        self.train_stimcodes = ['769', '770', '771', '772']
        self.test_stimcodes = '783'
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']

    def get_train_data(self, window, baseline=None):
        d_p = '../data/GDF/A0' + str(self.sub_id) + 'T.gdf'
        raw_data = mne.io.read_raw_gdf(d_p, preload=True)
        events, events_id = mne.events_from_annotations(raw_data)
        stims = [value for key, value in events_id.items() if key in self.train_stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=window[0], tmax=window[-1],
                            event_repeated='drop', baseline=None, preload=True,
                            proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        data = epochs.get_data() * 1e6
        data = data[:, :, :-1]
        data = zero_score(data)
        print('*' * 40, 'loading session 01 data over', '*' * 40)

        return data

    def get_train_label(self):
        l_p = '../data/Labels/A0' + str(self.sub_id) + 'T.mat'
        labels = loadmat(l_p).get('classlabel')
        labels = labels.reshape(-1) - 1

        return labels

    def load_data(self):
        raw_data = []
        raw_labels = []

        for window in self.windows:
            raw_labels.append(self.get_train_label())
            raw_data.append(self.get_train_data(window))

        raw_data = np.stack(raw_data, axis=1)
        raw_labels = np.concatenate(raw_labels, axis=0)

        skf = StratifiedKFold(n_splits=self.k_fold)
        for train_index, test_index in skf.split(raw_data, raw_labels):
            train_data, train_labels = raw_data[train_index], raw_labels[train_index]
            test_data, test_labels = raw_data[test_index], raw_labels[test_index]
            train_eeg = MyDataset(train_data, train_labels)
            test_eeg = MyDataset(test_data, test_labels)

            print('=' * 90)
            print('shape of train data :', train_data.shape, '\tshape of train labels :', train_labels.shape)
            print('shape of test data :', test_data.shape, '\tshape of test labels :', test_labels.shape)
            print('=' * 90)

            yield [train_eeg, test_eeg]
