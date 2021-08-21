import os
from easydict import EasyDict as edict
import yaml
import utils
import joblib
import torch
from torch.nn.utils.rnn import pad_sequence
from utils import collect_files
import numpy as np
import espnet
import espnet.transform as trans
import espnet.transform.spec_augment as SPEC
import matplotlib.pyplot as plt
import random
import librosa
import soundfile as sf
import copy
import string
from tqdm import tqdm


class DiCOVA_Dataset(object):
    def __init__(self, config, params):
        """Get the data and supporting files"""
        self.config = config
        'Initialization'
        self.list_IDs = params['files']
        self.mode = params["mode"]
        self.metadata = params['metadata_object']
        self.class2index, self.index2class = utils.get_class2index_and_index2class()
        self.incorrect_scaler = self.config.post_pretraining_classifier.incorrect_scaler
        self.specaug_probability = params['specaugment']
        self.time_warp = params['time_warp']
        self.input_type = params['input_type']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]
        if self.mode == 'train':
            metadata = self.metadata.get_feature_metadata(file, dataset='DiCOVA')
            label = self.class2index[metadata['Covid_status']]
            label = self.to_GPU(torch.from_numpy(np.asarray(label)))
        elif self.mode == 'val':
            metadata = self.metadata.get_feature_metadata(file, dataset='DiCOVA')
            label = self.class2index[metadata['Covid_status']]
            label = self.to_GPU(torch.from_numpy(np.asarray(label)))
        else:
            metadata = None
            label = None

        """We want to load the audio file. Then we want to perform specaugment."""
        feats = utils.load(file, type='pickle')
        x = random.uniform(0, 1)
        if x <= self.specaug_probability and self.mode != 'test':
            time_width = round(feats.shape[0]*0.1)
            if self.time_warp:
                max_time_warp = 80
            else:
                max_time_warp = 1  # can't be zero
            aug_feats = SPEC.spec_augment(feats, resize_mode='PIL', max_time_warp=max_time_warp,
                                                                   max_freq_width=40, n_freq_mask=1,
                                                                   max_time_width=time_width, n_time_mask=2,
                                                                   inplace=False, replace_with_zero=True)
            # plt.subplot(211)
            # plt.imshow(feats.T)
            # plt.subplot(212)
            # plt.imshow(aug_feats.T)
            # plt.show()
            feats = aug_feats
        else:
            """"""

        if self.input_type == 'energy':
            """Take the mean along the feature dimension"""
            energy = np.mean(feats, axis=1)
            # plt.subplot(211)
            # plt.imshow(feats.T)
            # plt.subplot(212)
            # plt.plot(energy)
            # plt.show()
            feats = energy

        feats = self.to_GPU(torch.from_numpy(feats))
        feats = feats.to(torch.float32)

        """Get incorrect_scaler value"""
        if self.mode != 'test':
            if metadata['Covid_status'] == 'p':
                scaler = self.incorrect_scaler
            else:
                scaler = 1
            scaler = self.to_GPU(torch.from_numpy(np.asarray(scaler)))
            scaler = scaler.to(torch.float32)
            scaler.requires_grad = True
        else:
            scaler = None
        return file, feats, label, scaler

    def to_GPU(self, tensor):
        if self.config.use_gpu == True:
            tensor = tensor.cuda()
            return tensor
        else:
            return tensor

    def collate(self, data):
        files = [item[0] for item in data]
        spects = [item[1] for item in data]
        labels = [item[2] for item in data]
        scalers = [item[3] for item in data]
        spects = pad_sequence(spects, batch_first=True, padding_value=0)
        if self.mode != 'test':
            labels = torch.stack([x for x in labels])
            scalers = torch.stack([x for x in scalers])
        if self.input_type == 'energy':
            spects = torch.unsqueeze(spects, dim=2)
        return {'files': files, 'spects': spects, 'labels': labels, 'scalers': scalers}


def main():
    """"""

if __name__ == "__main__":
    main()