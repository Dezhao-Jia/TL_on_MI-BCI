import mne
import numpy as np

from scipy.io import loadmat
from data.utils import zero_score, EA
from data.myDataset import MyDataset

import warnings
warnings.filterwarnings('ignore')


class LoadData:
    def __init__(self, sub_id, windows):
        self.sub_id = sub_id
        self.windows = windows
        self.test_stimcodes = '783'
        self.train_stimcodes = ['769', '770', '771', '772']
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']

    def get_train_data(self, window):
        path = '../data/GDF/A0' + str(self.sub_id) + 'T.gdf'
        raw_data = mne.io.read_raw_gdf(path, preload=True)
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
        path = '../data/Labels/A0' + str(self.sub_id) + 'T.mat'
        labels = loadmat(path).get('classlabel')
        labels = labels.reshape(-1) - 1

        return labels

    def get_test_data(self, window):
        path = '../data/GDF/A0' + str(self.sub_id) + 'E.gdf'
        raw_data = mne.io.read_raw_gdf(path, preload=True)
        events, events_id = mne.events_from_annotations(raw_data)
        stims = [value for key, value in events_id.items() if key in self.test_stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=window[0], tmax=window[-1],
                            event_repeated='drop', baseline=None, preload=True,
                            proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        data = epochs.get_data() * 1e6
        data = data[:, :, :-1]
        data = zero_score(data)
        print('*' * 40, 'loading session 02 data over', '*' * 40)

        return data

    def get_test_label(self):
        path = '../data/Labels/A0' + str(self.sub_id) + 'E.mat'
        labels = loadmat(path).get('classlabel')
        labels = labels.reshape(-1) - 1

        return labels

    def load_data(self):
        train_data, train_labels = [], []
        test_data, test_labels = [], []
        for window in self.windows:
            train_labels.append(self.get_train_label())
            test_labels.append(self.get_test_label())
            train_data.append(self.get_train_data(window))
            test_data.append(self.get_test_data(window))

        train_data = EA(np.stack(train_data, axis=1))
        train_labels = np.concatenate(train_labels, axis=0)
        test_data = EA(np.stack(test_data, axis=1))
        test_labels = np.concatenate(test_labels, axis=0)
        train_eeg = MyDataset(train_data, train_labels)
        test_eeg = MyDataset(test_data, test_labels)

        print('=' * 90)
        print('shape of train data :', train_data.shape, '\tshape of train labels :', train_labels.shape)
        print('shape of test data :', test_data.shape, '\tshape of test labels :', test_labels.shape)
        print('=' * 90)

        return [train_eeg, test_eeg]
