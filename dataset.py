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
        self.data_object = params['data_object']
        self.class2index, self.index2class = utils.get_class2index_and_index2class()
        self.incorrect_scaler = self.config.post_pretraining_classifier.incorrect_scaler
        self.specaugment = params['specaugment']
        self.specaug_probability = self.config.train.specaug_probability
        self.dataloader_temp_wavs = self.config.directories.dataloader_temp_wavs
        if not os.path.isdir(self.dataloader_temp_wavs):
            os.mkdir(self.dataloader_temp_wavs)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]
        # """This may be a dumb way to do this but we want to change the filename here to go to .wav.
        #    It's coming from the .pkl files but I don't wanna change the pipeline so we fix it here."""
        # file = self.get_wav_from_pkl(file=file)
        if self.mode == 'train':
            metadata = self.data_object.get_augmented_file_metadata(file)
            label = self.class2index[metadata['Covid_status']]
            label = self.to_GPU(torch.from_numpy(np.asarray(label)))
        elif self.mode == 'val':
            metadata = self.data_object.get_file_metadata(file)
            label = self.class2index[metadata['Covid_status']]
            label = self.to_GPU(torch.from_numpy(np.asarray(label)))
        else:
            metadata = None
            label = None

        """We want to load the audio file. Then we want to perform the specaugment-esque transformations
           in the time domain. Then (1) compute spectrogram (2) compute OpenSMILE features."""

        if self.mode == 'train':
            feats = joblib.load(file)
            spectrogram = feats['spectrogram']
            opensmile_feats = feats['opensmile']
        elif self.mode == 'val':
            spectrogram = joblib.load(file)
            """Now we must also load the opensmile feats so change the filename"""
            filename = file.split('/')[-1][:-4]
            opensmile_path = os.path.join(self.config.directories.dicova_opensmile_feats, filename + '.pkl')
            opensmile_feats = joblib.load(opensmile_path)

        # audio, _ = librosa.core.load(file, sr=self.config.data.sr)
        # if self.specaugment and self.mode != 'test':
        #     new_audio = self.time_domain_spec_aug(audio=audio)
        # else:
        #     new_audio = copy.deepcopy(audio)

        # """Compute spectrogram"""
        # feature_processor = utils.Mel_log_spect()
        # spectrogram = feature_processor.get_Mel_log_spect(new_audio)
        #
        # """Get OpenSMILE features"""
        # # Generate random filename of length 8 to write to disk
        # temp_write_name = ''.join(random.choices(string.ascii_uppercase +
        #                              string.digits, k=8))
        # dump_path = os.path.join(self.dataloader_temp_wavs, temp_write_name + '.wav')
        # sf.write(dump_path, new_audio, self.config.data.sr, subtype='PCM_16')
        # output = self.opensmile.process_file(dump_path)
        # os.remove(dump_path)  # don't want to keep all those temporary files!
        # opensmile_feats = np.squeeze(output.to_numpy())

        """If we had the augmented dataset ahead of time, we could normalize each feature,
           but let's just normalize the feature vector. It should be similar."""
        opensmile_feats = opensmile_feats/self.opensmile_norm_factors

        spectrogram = self.to_GPU(torch.from_numpy(spectrogram))
        opensmile_feats = self.to_GPU(torch.from_numpy(opensmile_feats))
        spectrogram = spectrogram.to(torch.float32)
        opensmile_feats = opensmile_feats.to(torch.float32)

        # if self.specaugment:
        #     feats = joblib.load(file)
        #     x = random.uniform(0, 1)
        #     if x <= self.specaug_probability and self.mode != 'test':
        #         time_width = round(feats.shape[0]*0.1)
        #         aug_feats = SPEC.spec_augment(feats, resize_mode='PIL', max_time_warp=80,
        #                                                                max_freq_width=20, n_freq_mask=1,
        #                                                                max_time_width=time_width, n_time_mask=2,
        #                                                                inplace=False, replace_with_zero=True)
        #         # plt.subplot(211)
        #         # plt.imshow(feats.T)
        #         # plt.subplot(212)
        #         # plt.imshow(aug_feats.T)
        #         # plt.show()
        #         feats = self.to_GPU(torch.from_numpy(aug_feats))
        #     else:
        #         feats = self.to_GPU(torch.from_numpy(feats))
        # else:
        #     feats = self.to_GPU(torch.from_numpy(joblib.load(file)))
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
        return file, spectrogram, opensmile_feats, label, scaler

    def to_GPU(self, tensor):
        if self.config.use_gpu == True:
            tensor = tensor.cuda()
            return tensor
        else:
            return tensor

    def get_wav_from_pkl(self, file):
        name = file.split('/')[-1][:-4]
        new_path = os.path.join(self.config.directories.dicova_wavs, name + '.wav')
        return new_path

    def collate(self, data):
        files = [item[0] for item in data]
        spects = [item[1] for item in data]
        opensmile = [item[2] for item in data]
        labels = [item[3] for item in data]
        scalers = [item[4] for item in data]
        spects = pad_sequence(spects, batch_first=True, padding_value=0)
        opensmile = torch.stack([x for x in opensmile])
        if self.mode != 'test':
            labels = torch.stack([x for x in labels])
            scalers = torch.stack([x for x in scalers])
        return {'files': files, 'spects': spects, 'opensmile': opensmile,
                'labels': labels, 'scalers': scalers}


def main():
    config = get_config.get()
    dicova = DiCOVA(config=config)
    # dicova.get_features()
    # dicova.get_test_files_and_feats()
    dicova.get_opensmile_feats()

if __name__ == "__main__":
    main()