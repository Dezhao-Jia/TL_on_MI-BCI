import mne
import numpy as np

from scipy.io import loadmat
from data.myDataset import MyDataset
from data.data_augment import length_change
from sklearn.model_selection import StratifiedKFold


import warnings
warnings.filterwarnings('ignore')


class LoadData:
    def __init__(self, sub_id, windows, k_fold=6):
        self.sub_id = sub_id
        self.k_fold = k_fold
        self.windows = windows
        self.train_stimcodes = ['769', '770', '771', '772']
        self.test_stimcodes = '783'
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']

    def get_train_data(self, window):
        d_p = 'data/GDF/A0' + str(self.sub_id) + 'T.gdf'
        raw_data = mne.io.read_raw_gdf(d_p, preload=True)
        events, events_id = mne.events_from_annotations(raw_data)
        stims = [value for key, value in events_id.items() if key in self.train_stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=window[0], tmax=window[-1],
                            event_repeated='drop', baseline=None, preload=True,
                            proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        data = epochs.get_data() * 1e6
        data = data[:, :, :-1]
        print('*'*40, 'loading session 01 data over', '*'*40)

        return data

    def get_train_label(self):
        l_p = 'data/Labels/A0' + str(self.sub_id) + 'T.mat'
        labels = loadmat(l_p).get('classlabel')
        labels = labels.reshape(-1) - 1

        return labels

    def get_test_data(self, window):
        d_p = 'data/GDF/A0' + str(self.sub_id) + 'E.gdf'
        raw_data = mne.io.read_raw_gdf(d_p, preload=True)
        events, events_id = mne.events_from_annotations(raw_data)
        stims = [value for key, value in events_id.items() if key in self.test_stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=window[0], tmax=window[-1],
                            event_repeated='drop', baseline=None, preload=True,
                            proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        data = epochs.get_data() * 1e6
        data = data[:, :, :-1]
        print('*' * 40, 'loading session 02 data over', '*' * 40)

        return data

    def get_test_label(self):
        l_p = 'data/Labels/A0' + str(self.sub_id) + 'E.mat'
        labels = loadmat(l_p).get('classlabel')
        labels = labels.reshape(-1) - 1

        return labels

    def load_data(self):
        src_data = []
        src_labels = []
        raw_data = []
        raw_labels = []
        for window in self.windows:
            src_labels.append(self.get_train_label())
            raw_labels.append(self.get_test_label())
            src_data.append(self.get_train_data(window))
            raw_data.append(self.get_test_data(window))

        src_data = np.stack(src_data, axis=1)
        raw_data = np.stack(raw_data, axis=1)
        src_labels = np.concatenate(src_labels, axis=0)
        raw_labels = np.concatenate(raw_labels, axis=0)

        src_data, src_labels = length_change(src_data, src_labels)

        skf = StratifiedKFold(n_splits=self.k_fold)
        for tag_index, aux_index in skf.split(raw_data, raw_labels):
            tag_data = raw_data[tag_index]
            tag_labels = raw_labels[tag_index]
            aux_data = raw_data[aux_index]
            aux_labels = raw_labels[aux_index]

            aux_data, aux_labels = length_change(aux_data, aux_labels)

            src_eeg = MyDataset(src_data, src_labels)
            aux_eeg = MyDataset(aux_data, aux_labels)
            tag_eeg = MyDataset(tag_data, tag_labels)

            print('=' * 90)
            print('shape of source data :', src_data.shape, '\tshape of source labels :', src_labels.shape)
            print('shape of auxiliary data :', aux_data.shape, '\tshape of auxiliary labels :', aux_labels.shape)
            print('shape of target data :', tag_data.shape, '\tshape of target labels :', tag_labels.shape)
            print('=' * 90)

            yield [src_eeg, aux_eeg, tag_eeg]
