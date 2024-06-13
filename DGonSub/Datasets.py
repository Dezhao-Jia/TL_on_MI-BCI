import mne
import numpy as np

from scipy.io import loadmat
from data.myDataset import MyDataset
from data.utils import zero_score, EA

import warnings
warnings.filterwarnings("ignore")


class LoadData:
    def __init__(self, tag_id, windows):
        self.tag_id = tag_id
        self.windows = windows
        self.test_stimcodes = '783'
        self.train_stimcodes = ['769', '770', '771', '772']
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']

    def get_train_data(self, sub_id, window):
        data_path = 'data/GDF/A0' + str(sub_id) + 'T.gdf'
        raw_data = mne.io.read_raw_gdf(data_path, preload=True)
        events, events_id = mne.events_from_annotations(raw_data)
        stims = [value for key, value in events_id.items() if key in self.train_stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=window[0], tmax=window[-1],
                            event_repeated='drop', baseline=None, preload=True,
                            proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        data = epochs.get_data()
        data = zero_score(data)
        data = data[:, :, :-1]

        return data

    def get_train_labels(self, sub_id):
        labels_path = 'data/Labels/A0' + str(sub_id) + 'T.mat'
        labels = loadmat(labels_path).get('classlabel')
        labels = labels.reshape(-1) - 1

        return labels

    def get_eval_data(self, sub_id, window):
        data_path = 'data/GDF/A0' + str(sub_id) + 'E.gdf'
        raw_data = mne.io.read_raw_gdf(data_path, preload=True)
        events, events_id = mne.events_from_annotations(raw_data)
        stims = [value for key, value in events_id.items() if key in self.test_stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=window[0], tmax=window[-1],
                            event_repeated='drop', baseline=None, preload=True,
                            proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        data = epochs.get_data()
        data = zero_score(data)
        data = data[:, :, :-1]

        return data

    def get_eval_labels(self, sub_id):
        labels_path = 'data/Labels/A0' + str(sub_id) + 'E.mat'
        labels = loadmat(labels_path).get('classlabel')
        labels = labels.reshape(-1) - 1

        return labels

    def get_single_datasets(self, sub_id):
        data = []
        labels = []

        for window in self.windows:
            l1 = self.get_train_labels(sub_id)
            l2 = self.get_eval_labels(sub_id)
            d1 = np.stack(self.get_train_data(sub_id, window), axis=1)
            d2 = np.stack(self.get_eval_data(sub_id, window), axis=1)

            data.append(np.concatenate([d1, d2], axis=0))
            labels.append(np.concatenate([l1, l2], axis=0))

        data = np.stack(data, axis=1)
        labels = np.concatenate(labels, axis=0)

        return data, labels

    def get_dataSets(self):
        src_data, src_labels = [], []
        tag_data, tag_labels = [], []
        for i in range(1):
            if i + 1 != self.tag_id:
                print('#'*40, 'Loading source domain dataset', '#'*40)
                data, labels = self.get_single_datasets(i + 1)
                src_data.append(data)
                src_labels.append(labels)
            else:
                print('#'*40, 'Loading target domain dataset', '#'*40)
                data, labels = self.get_single_datasets(i + 1)
                tag_data.append(data)
                tag_labels.append(labels)

        src_data = EA(np.concatenate(src_data, axis=0))
        src_labels = np.concatenate(src_labels, axis=0)
        tag_data = EA(np.concatenate(tag_data, axis=0))
        tag_labels = np.concatenate(tag_labels, axis=0)

        return [src_data, src_labels], [tag_data, tag_labels]

    def load_data(self):
        src_datasets, tag_datasets = self.get_dataSets()
        src_eeg = MyDataset(src_datasets[0], src_datasets[-1])
        tag_eeg = MyDataset(tag_datasets[0], tag_datasets[-1])

        print('=' * 90)
        print('shape of source data :', src_datasets[0].shape, '\tshape of source labels :', src_datasets[-1].shape)
        print('shape of target data :', tag_datasets[0].shape, '\tshape of target labels :', tag_datasets[-1].shape)
        print('=' * 90)

        return [src_eeg, tag_eeg]
