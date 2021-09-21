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
        self.mf_class2index, self.mf_index2class = utils.get_mf_class2index_and_index2class()
        self.incorrect_scaler = self.config.post_pretraining_classifier.incorrect_scaler
        self.specaug_probability = params['specaugment']
        self.time_warp = params['time_warp']
        self.input_type = params['input_type']
        self.args = params['args']

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
            mf = self.mf_class2index[metadata['Gender']]
            mf = self.to_GPU(torch.from_numpy(np.asarray(mf)))
        elif self.mode == 'val':
            metadata = self.metadata.get_feature_metadata(file, dataset='DiCOVA')
            label = self.class2index[metadata['Covid_status']]
            label = self.to_GPU(torch.from_numpy(np.asarray(label)))
            mf = self.mf_class2index[metadata['Gender']]
            mf = self.to_GPU(torch.from_numpy(np.asarray(mf)))
        else:
            metadata = None
            label = None

        """We want to load the audio file. Then we want to perform specaugment."""
        feats = utils.load(file, type='pickle')
        x = random.uniform(0, 1)
        if x <= self.specaug_probability and self.mode != 'test':
            time_width = round(feats.shape[0]*0.1)
            if self.time_warp:
                max_time_warp = 250
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

        """NEW: speech clips are VERY long so for training speedup/augmentation we take small window as training example"""
        if self.mode == 'train':
            old_feats = feats
            clip_length = feats.shape[0]
            section_length = int(self.args.TRAIN_CLIP_FRACTION * clip_length)
            new_start = int(random.uniform(0, clip_length-section_length))
            feats = feats[new_start: new_start+section_length]
        # plt.subplot(211)
        # plt.imshow(feats.T)
        # plt.subplot(212)
        # plt.imshow(old_feats.T)
        # plt.show()

        if self.input_type == 'energy':
            """Take the mean along the feature dimension"""
            energy = np.mean(feats, axis=1)
            # plt.subplot(211)
            # plt.imshow(feats.T)
            # plt.subplot(212)
            # plt.plot(energy)
            # plt.show()
            feats = energy

        elif self.input_type == 'mfcc':
            """Get the mfccs from the normalized spectrogram"""
            mfcc = librosa.feature.mfcc(S=feats.T, n_mfcc=self.config.data.num_mfccs)
            # plt.subplot(211)
            # plt.imshow(feats.T)
            # plt.subplot(212)
            # plt.imshow(mfcc)
            # plt.show()
            feats = mfcc.T

        # plt.subplot(211)
        # plt.plot(feats)
        # plt.subplot(212)
        # plt.plot(old_feats)
        # plt.show()

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
        return file, feats, label, scaler, mf

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
        mf = [item[4] for item in data]
        spects = pad_sequence(spects, batch_first=True, padding_value=0)
        if self.mode != 'test':
            labels = torch.stack([x for x in labels])
            scalers = torch.stack([x for x in scalers])
            mf = torch.stack([x for x in mf])
        if self.input_type == 'energy':
            spects = torch.unsqueeze(spects, dim=2)
        return {'files': files, 'spects': spects, 'labels': labels, 'scalers': scalers, 'mf': mf}


class DiCOVA_Dataset_Fusion(object):
    def __init__(self, config, params):
        """Get the data and supporting files"""
        self.config = config
        'Initialization'
        self.list_IDs = params['files']
        self.mode = params["mode"]
        self.metadata = params['metadata_object']
        self.class2index, self.index2class = utils.get_class2index_and_index2class()
        self.mf_class2index, self.mf_index2class = utils.get_mf_class2index_and_index2class()
        self.incorrect_scaler = self.config.post_pretraining_classifier.incorrect_scaler
        self.specaug_probability = params['specaugment']
        self.time_warp = params['time_warp']
        self.input_type = params['input_type']
        self.args = params['args']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]  # triple of three modality files, need to adjust code below
        speech = file['speech']
        # cough = file['cough']
        # breathing = file['breathing']
        if self.mode == 'train':
            metadata = self.metadata.get_feature_metadata(speech, dataset='DiCOVA')  # all modalities have the same metadata
            label = self.class2index[metadata['Covid_status']]
            label = self.to_GPU(torch.from_numpy(np.asarray(label)))
            mf = self.mf_class2index[metadata['Gender']]
            mf = self.to_GPU(torch.from_numpy(np.asarray(mf)))
        elif self.mode == 'val':
            metadata = self.metadata.get_feature_metadata(speech, dataset='DiCOVA')
            label = self.class2index[metadata['Covid_status']]
            label = self.to_GPU(torch.from_numpy(np.asarray(label)))
            mf = self.mf_class2index[metadata['Gender']]
            mf = self.to_GPU(torch.from_numpy(np.asarray(mf)))
        else:
            metadata = None
            label = None

        """We want to load the audio files. Then we want to perform specaugment."""
        feats_all_modalities = {}
        for modality in ['speech', 'cough', 'breathing']:
            feats = utils.load(file[modality], type='pickle')
            # adjust to modality-specific TRAIN_CLIP_FRACTION
            if modality == 'speech':
                TRAIN_CLIP_FRACTION = self.args.TRAIN_CLIP_FRACTION_SPEECH
            elif modality == 'cough':
                TRAIN_CLIP_FRACTION = self.args.TRAIN_CLIP_FRACTION_COUGH
            elif modality == 'breathing':
                TRAIN_CLIP_FRACTION = self.args.TRAIN_CLIP_FRACTION_BREATHING
            x = random.uniform(0, 1)
            if x <= self.specaug_probability and self.mode != 'test':
                time_width = round(feats.shape[0]*0.1)
                if self.time_warp:
                    max_time_warp = 250
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

            """NEW: speech clips are VERY long so for training speedup/augmentation we take small window as training example"""
            if self.mode == 'train':
                old_feats = feats
                clip_length = feats.shape[0]
                section_length = int(TRAIN_CLIP_FRACTION * clip_length)
                new_start = int(random.uniform(0, clip_length-section_length))
                feats = feats[new_start: new_start+section_length]
            # plt.subplot(211)
            # plt.imshow(feats.T)
            # plt.subplot(212)
            # plt.imshow(old_feats.T)
            # plt.show()

            if self.input_type == 'energy':
                """Take the mean along the feature dimension"""
                energy = np.mean(feats, axis=1)
                # plt.subplot(211)
                # plt.imshow(feats.T)
                # plt.subplot(212)
                # plt.plot(energy)
                # plt.show()
                feats = energy

            elif self.input_type == 'mfcc':
                """Get the mfccs from the normalized spectrogram"""
                mfcc = librosa.feature.mfcc(S=feats.T, n_mfcc=self.config.data.num_mfccs)
                # plt.subplot(211)
                # plt.imshow(feats.T)
                # plt.subplot(212)
                # plt.imshow(mfcc)
                # plt.show()
                feats = mfcc.T

            # plt.subplot(211)
            # plt.plot(feats)
            # plt.subplot(212)
            # plt.plot(old_feats)
            # plt.show()

            feats = self.to_GPU(torch.from_numpy(feats))
            feats = feats.to(torch.float32)

            feats_all_modalities[modality] = feats

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
        return file, feats_all_modalities, label, scaler, mf

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
        mf = [item[4] for item in data]
        """Need to grab each modality's features and pad those by batch size"""
        speech = [x['speech'] for x in spects]
        cough = [x['cough'] for x in spects]
        breathing = [x['breathing'] for x in spects]
        speech = pad_sequence(speech, batch_first=True, padding_value=0)
        cough = pad_sequence(cough, batch_first=True, padding_value=0)
        breathing = pad_sequence(breathing, batch_first=True, padding_value=0)
        if self.mode != 'test':
            labels = torch.stack([x for x in labels])
            scalers = torch.stack([x for x in scalers])
            mf = torch.stack([x for x in mf])
        if self.input_type == 'energy':
            speech = torch.unsqueeze(speech, dim=2)
            cough = torch.unsqueeze(cough, dim=2)
            breathing = torch.unsqueeze(breathing, dim=2)
        spects = {'speech': speech, 'cough': cough, 'breathing': breathing}
        return {'files': files, 'spects': spects, 'labels': labels, 'scalers': scalers, 'mf': mf}



class DiCOVA_Test_Dataset(object):
    def __init__(self, config, params):
        """Get the data and supporting files"""
        self.config = config
        'Initialization'
        self.list_IDs = params['files']
        self.mode = params["mode"]
        self.metadata = params['metadata_object']
        self.class2index, self.index2class = utils.get_class2index_and_index2class()
        self.mf_class2index, self.mf_index2class = utils.get_mf_class2index_and_index2class()
        self.incorrect_scaler = self.config.post_pretraining_classifier.incorrect_scaler
        self.specaug_probability = params['specaugment']
        self.time_warp = params['time_warp']
        self.input_type = params['input_type']
        self.args = params['args']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]

        if self.mode == 'test':
            metadata = self.metadata.get_feature_metadata(file, dataset='DiCOVA')
            mf = self.mf_class2index[metadata['Gender']]
            mf = self.to_GPU(torch.from_numpy(np.asarray(mf)))
        else:
            metadata = None

        """We want to load the audio file. Then we want to perform specaugment."""
        feats = utils.load(file, type='pickle')
        x = random.uniform(0, 1)
        if x <= self.specaug_probability and self.mode != 'test':
            time_width = round(feats.shape[0]*0.1)
            if self.time_warp:
                max_time_warp = 250
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

        """NEW: speech clips are VERY long so for training speedup/augmentation we take small window as training example"""
        if self.mode == 'train':
            old_feats = feats
            clip_length = feats.shape[0]
            section_length = int(self.args.TRAIN_CLIP_FRACTION * clip_length)
            new_start = int(random.uniform(0, clip_length-section_length))
            feats = feats[new_start: new_start+section_length]
        # plt.subplot(211)
        # plt.imshow(feats.T)
        # plt.subplot(212)
        # plt.imshow(old_feats.T)
        # plt.show()

        if self.input_type == 'energy':
            """Take the mean along the feature dimension"""
            energy = np.mean(feats, axis=1)
            # plt.subplot(211)
            # plt.imshow(feats.T)
            # plt.subplot(212)
            # plt.plot(energy)
            # plt.show()
            feats = energy

        elif self.input_type == 'mfcc':
            """Get the mfccs from the normalized spectrogram"""
            mfcc = librosa.feature.mfcc(S=feats.T, n_mfcc=self.config.data.num_mfccs)
            # plt.subplot(211)
            # plt.imshow(feats.T)
            # plt.subplot(212)
            # plt.imshow(mfcc)
            # plt.show()
            feats = mfcc.T

        # plt.subplot(211)
        # plt.plot(feats)
        # plt.subplot(212)
        # plt.plot(old_feats)
        # plt.show()

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
        label = None
        return file, feats, label, scaler, mf

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
        mf = [item[4] for item in data]
        spects = pad_sequence(spects, batch_first=True, padding_value=0)
        if self.mode != 'test':
            labels = torch.stack([x for x in labels])
            scalers = torch.stack([x for x in scalers])
            mf = torch.stack([x for x in mf])
        elif self.mode == 'test':
            mf = torch.stack([x for x in mf])
        if self.input_type == 'energy':
            spects = torch.unsqueeze(spects, dim=2)
        return {'files': files, 'spects': spects, 'labels': labels, 'scalers': scalers, 'mf': mf}

class DiCOVA_Dataset_Margin(object):
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
        self.args = params['args']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]
        positive = file['positive']
        negative = file['negative']
        paired_feats = []
        """We want to load the audio files. Then we want to perform specaugment."""
        for file in [positive, negative]:
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

            """NEW: speech clips are VERY long so for training speedup/augmentation we take small window as training example"""
            if self.mode == 'train':
                old_feats = feats
                clip_length = feats.shape[0]
                section_length = int(self.args.TRAIN_CLIP_FRACTION * clip_length)
                new_start = int(random.uniform(0, clip_length-section_length))
                feats = feats[new_start: new_start+section_length]
            # plt.subplot(211)
            # plt.imshow(feats.T)
            # plt.subplot(212)
            # plt.imshow(old_feats.T)
            # plt.show()

            if self.input_type == 'energy':
                """Take the mean along the feature dimension"""
                energy = np.mean(feats, axis=1)
                # plt.subplot(211)
                # plt.imshow(feats.T)
                # plt.subplot(212)
                # plt.plot(energy)
                # plt.show()
                feats = energy
            paired_feats.append(feats)
        pos_feats, neg_feats = paired_feats
        stop = None


        # plt.subplot(211)
        # plt.plot(feats)
        # plt.subplot(212)
        # plt.plot(old_feats)
        # plt.show()

        pos_feats = self.to_GPU(torch.from_numpy(pos_feats))
        pos_feats = pos_feats.to(torch.float32)
        neg_feats = self.to_GPU(torch.from_numpy(neg_feats))
        neg_feats = neg_feats.to(torch.float32)

        return file, pos_feats, neg_feats

    def to_GPU(self, tensor):
        if self.config.use_gpu == True:
            tensor = tensor.cuda()
            return tensor
        else:
            return tensor

    def collate(self, data):
        files = [item[0] for item in data]
        pos_spects = [item[1] for item in data]
        neg_spects = [item[2] for item in data]
        pos_spects = pad_sequence(pos_spects, batch_first=True, padding_value=0)
        neg_spects = pad_sequence(neg_spects, batch_first=True, padding_value=0)
        if self.input_type == 'energy':
            pos_spects = torch.unsqueeze(pos_spects, dim=2)
            neg_spects = torch.unsqueeze(neg_spects, dim=2)
        return {'files': files, 'pos_spects': pos_spects, 'neg_spects': neg_spects}

class LibriSpeech_Dataset(object):
    def __init__(self, config, params):
        """Get the data and supporting files"""
        self.config = config
        'Initialization'
        self.list_IDs = params['files']
        self.mode = params["mode"]
        self.input_type = params['input_type']
        self.args = params['args']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]

        """We want to load the audio file. Then we want to perform specaugment."""
        feats = utils.load(file, type='pickle')

        """NEW: speech clips are VERY long so for training speedup/augmentation we take small window as training example"""
        if self.mode == 'train':
            old_feats = feats
            clip_length = feats.shape[0]
            section_length = int(self.args.TRAIN_CLIP_FRACTION * clip_length)
            new_start = int(random.uniform(0, clip_length-section_length))
            feats = feats[new_start: new_start+section_length]
        # plt.subplot(211)
        # plt.imshow(feats.T)
        # plt.subplot(212)
        # plt.imshow(old_feats.T)
        # plt.show()
        if self.input_type == 'mfcc':
            """Get the mfccs from the normalized spectrogram"""
            mfcc = librosa.feature.mfcc(S=feats.T, n_mfcc=self.config.data.num_mfccs)
            # plt.subplot(211)
            # plt.imshow(feats.T)
            # plt.subplot(212)
            # plt.imshow(mfcc)
            # plt.show()
            feats = mfcc.T

        feats = self.to_GPU(torch.from_numpy(feats))
        feats = feats.to(torch.float32)

        return file, feats

    def to_GPU(self, tensor):
        if self.config.use_gpu == True:
            tensor = tensor.cuda()
            return tensor
        else:
            return tensor

    def collate(self, data):
        files = [item[0] for item in data]
        spects = [item[1] for item in data]
        spects = pad_sequence(spects, batch_first=True, padding_value=0)
        return {'files': files, 'spects': spects}

class COUGHVID_Dataset(object):
    def __init__(self, config, params):
        """Get the data and supporting files"""
        self.config = config
        'Initialization'
        self.list_IDs = params['files']
        self.mode = params["mode"]
        self.input_type = params['input_type']
        self.args = params['args']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]

        """We want to load the audio file. Then we want to perform specaugment."""
        feats = utils.load(file, type='pickle')

        """NEW: speech clips are VERY long so for training speedup/augmentation we take small window as training example"""
        if self.mode == 'train':
            old_feats = feats
            clip_length = feats.shape[0]
            section_length = int(self.args.TRAIN_CLIP_FRACTION * clip_length)
            new_start = int(random.uniform(0, clip_length-section_length))
            feats = feats[new_start: new_start+section_length]
        # plt.subplot(211)
        # plt.imshow(feats.T)
        # plt.subplot(212)
        # plt.imshow(old_feats.T)
        # plt.show()
        if self.input_type == 'mfcc':
            """Get the mfccs from the normalized spectrogram"""
            mfcc = librosa.feature.mfcc(S=feats.T, n_mfcc=self.config.data.num_mfccs)
            # plt.subplot(211)
            # plt.imshow(feats.T)
            # plt.subplot(212)
            # plt.imshow(mfcc)
            # plt.show()
            feats = mfcc.T

        feats = self.to_GPU(torch.from_numpy(feats))
        feats = feats.to(torch.float32)

        return file, feats

    def to_GPU(self, tensor):
        if self.config.use_gpu == True:
            tensor = tensor.cuda()
            return tensor
        else:
            return tensor

    def collate(self, data):
        files = [item[0] for item in data]
        spects = [item[1] for item in data]
        spects = pad_sequence(spects, batch_first=True, padding_value=0)
        return {'files': files, 'spects': spects}


def main():
    """"""

if __name__ == "__main__":
    main()