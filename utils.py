import os
from tqdm import tqdm
import numpy as np
import joblib
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict
import pandas as pd
import shutil
from torch.utils import data
from itertools import groupby
import json
import random
import librosa
import matplotlib.pyplot as plt
import time
import multiprocessing
import concurrent.futures
import json
import pandas as pd
import soundfile as sf
import pickle
from sklearn.metrics import auc
import sys
import copy
import string
from matplotlib import rc
import matplotlib


def collect_files(directory, verbose=False):
    all_files = []
    if verbose:
        print('Collecting files in ' + directory + '...')
    for path, subdirs, files in os.walk(directory):
        for name in files:
            filename = os.path.join(path, name)
            all_files.append(filename)
    if verbose:
        print('Done.')
    return all_files

def get_config():
    config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))
    return config

def keep_wavs(files):
    new_list = []
    for file in files:
        if file[-4:] == '.wav':
            new_list.append(file)
    return new_list

def keep_flac(files):
    new_list = []
    for file in files:
        if file[-5:] == '.flac':
            new_list.append(file)
    return new_list

def dump(value, filename, type='joblib', verbose=False):
    if verbose:
        print('Dumping file to disk...')
    if type == 'pickle':
        with open(filename, 'wb') as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif type == 'joblib':
        joblib.dump(value, filename)
    else:
        raise ValueError('Dump type not implemented...')
    if verbose:
        print('Done.')

class Mel_log_spect(object):
    def __init__(self):
        self.config = get_config()
        self.nfft = self.config.data.fftl
        self.num_mels = self.config.data.num_mels
        self.hop_length = self.config.data.hop_length
        self.top_db = self.config.data.top_db
        self.sr = self.config.data.sr

    def feature_normalize(self, x):
        log_min = np.min(x)
        x = x - log_min
        x = x / self.top_db
        x = x.T
        return x

    def get_Mel_log_spect(self, y):
        y = librosa.util.normalize(S=y)
        spect = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.nfft,
                                               hop_length=self.hop_length, n_mels=self.num_mels)
        log_spect = librosa.core.amplitude_to_db(spect, ref=1.0, top_db=self.top_db)
        log_spect = self.feature_normalize(log_spect)
        return log_spect

    def norm_Mel_log_spect_to_amplitude(self, feature):
        feature = feature * self.top_db
        spect = librosa.core.db_to_amplitude(feature, ref=1.0)
        return spect

    def audio_from_spect(self, feature):
        spect = self.norm_Mel_log_spect_to_amplitude(feature)
        audio = librosa.feature.inverse.mel_to_audio(spect.T, sr=self.sr, n_fft=self.nfft, hop_length=self.hop_length)
        return audio

    def convert_and_write(self, load_path, write_path):
        y, sr = librosa.core.load(path=load_path, sr=self.sr)
        feature = self.get_Mel_log_spect(y, n_mels=self.num_mels)
        audio = self.audio_from_spect(feature)
        librosa.output.write_wav(write_path, y=audio, sr=self.sr, norm=True)