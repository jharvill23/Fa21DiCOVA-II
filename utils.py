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
import csv
import argparse


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

def keep_webm(files):
    new_list = []
    for file in files:
        if file[-5:] == '.webm':
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

def load(filename, type='joblib', verbose=False):
    if verbose:
        print('Loading file from disk...')
    if type == 'pickle':
        with open(filename, 'rb') as handle:
            value = pickle.load(handle)
    elif type == 'joblib':
        value = joblib.load(filename)
    else:
        raise ValueError('Load type not implemented...')
    if verbose:
        print('Done.')
    return value

def dump_audio(audio, write_path, sr):
    sf.write(write_path, audio, sr, "PCM_16")

def get_class2index_and_index2class():
    class2index = {'p': 0, 'n': 1}
    index2class = {0: 'p', 1: 'n'}
    return class2index, index2class

def get_mf_class2index_and_index2class():
    class2index = {'m': 0, 'f': 1}
    index2class = {0: 'm', 1: 'f'}
    return class2index, index2class

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def scoring(refs, sys_outs, out_file):
    """
        inputs::
        refs: a txt file with a list of labels for each wav-fileid in the format: <id> <label>
        sys_outs: a txt file with a list of scores (probability of being covid positive) for each wav-fileid in the format: <id> <score>
        threshold (optional): a np.array(), like np.arrange(0,1,.01), sweeping for AUC
        outputs::
        """

    thresholds = np.arange(0, 1, 0.01)
    # Read the ground truth labels into a dictionary
    data = open(refs).readlines()
    reference_labels = {}
    categories = ['n', 'p']
    for line in data:
        key, val = line.strip().split()
        reference_labels[key] = categories.index(val)

    # Read the system scores into a dictionary
    data = open(sys_outs).readlines()
    sys_scores = {}
    for line in data:
        key, val = line.strip().split()
        sys_scores[key] = float(val)
    del data

    # Ensure all files in the reference have system scores and vice-versa
    if len(sys_scores) != len(reference_labels):
        print("Expected the score file to have scores for all files in reference and no duplicates/extra entries")
        return None
    # %%

    # Arrays to store true positives, false positives, true negatives, false negatives
    TP = np.zeros((len(reference_labels), len(thresholds)))
    TN = np.zeros((len(reference_labels), len(thresholds)))
    keyCnt = -1
    for key in sys_scores:  # Repeat for each recording
        keyCnt += 1
        sys_labels = (sys_scores[key] >= thresholds) * 1  # System label for a range of thresholds as binary 0/1
        gt = reference_labels[key]

        ind = np.where(sys_labels == gt)  # system label matches the ground truth
        if gt == 1:  # ground-truth label=1: True positives
            TP[keyCnt, ind] = 1
        else:  # ground-truth label=0: True negatives
            TN[keyCnt, ind] = 1

    total_positives = sum(reference_labels.values())  # Total number of positive samples
    total_negatives = len(reference_labels) - total_positives  # Total number of negative samples

    TP = np.sum(TP, axis=0)  # Sum across the recordings
    TN = np.sum(TN, axis=0)

    TPR = TP / total_positives  # True positive rate: #true_positives/#total_positives
    TNR = TN / total_negatives  # True negative rate: #true_negatives/#total_negatives

    AUC = auc(1 - TNR, TPR)  # AUC

    ind = np.where(TPR >= 0.8)[0]
    sensitivity = TPR[ind[-1]]
    specificity = TNR[ind[-1]]

    # pack the performance metrics in a dictionary to save & return
    # Each performance metric (except AUC) is a array for different threshold values
    # Specificity at 90% sensitivity
    scores = {'TPR': TPR,
              'FPR': 1 - TNR,
              'AUC': AUC,
              'sensitivity': sensitivity,
              'specificity': specificity,
              'thresholds': thresholds}

    with open(out_file, "wb") as f:
        pickle.dump(scores, f)

def summary(folname, scores, iterations):
    # folname = sys.argv[1]
    num_files = 1
    R = []
    for i in range(num_files):
        # res = pickle.load(open(folname + "/fold_{}/val_results.pkl".format(i + 1), 'rb'))
        # res = pickle.load(open(scores))
        res = joblib.load(scores)
        R.append(res)

    # Plot ROC curves
    clr_1 = 'tab:green'
    clr_2 = 'tab:green'
    clr_3 = 'k'
    data_x, data_y, data_auc = [], [], []
    for i in range(num_files):
        data_x.append(R[i]['FPR'].tolist())
        data_y.append(R[i]['TPR'].tolist())
        data_auc.append(R[i]['AUC'] * 100)
        plt.plot(data_x[i], data_y[i], label='V-' + str(i + 1) + ', auc=' + str(np.round(data_auc[i], 2)), c=clr_1,
                 alpha=0.2)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    plt.plot(np.mean(data_x, axis=0), np.mean(data_y, axis=0),
             label='AVG, auc=' + str(np.round(np.mean(np.array(data_auc)), 2)), c=clr_2, alpha=1, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', label='chance', c=clr_3, alpha=.5)
    plt.legend(loc='lower right', frameon=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
    plt.text(0, 1, 'PATIENT-LEVEL ROC', color='gray', fontsize=12)

    plt.gca().set_xlabel('FALSE POSITIVE RATE')
    plt.gca().set_ylabel('TRUE POSITIVE RATE')
    plt.savefig(os.path.join(folname, 'val_roc_plot_' + str(iterations) + '.pdf'), bbox_inches='tight')
    plt.close()

    sensitivities = [R[i]['sensitivity'] * 100 for i in range(num_files)]
    specificities = [R[i]['specificity'] * 100 for i in range(num_files)]

    with open(os.path.join(folname, 'val_summary_metrics.txt'), 'w') as f:
        f.write("Sensitivities: " + " ".join([str(round(item, 2)) for item in sensitivities]) + "\n")
        f.write("Specificities: " + " ".join([str(round(item, 2)) for item in specificities]) + "\n")
        f.write("AUCs: " + " ".join([str(round(item, 2)) for item in data_auc]) + "\n")
        f.write(
            "Average sensitivity: " + str(np.round(np.mean(np.array(sensitivities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(sensitivities)), 2)) + "\n")
        f.write(
            "Average specificity: " + str(np.round(np.mean(np.array(specificities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(specificities)), 2)) + "\n")
        f.write("Average AUC: " + str(np.round(np.mean(np.array(data_auc)), 2)) + " standard deviation:" + str(
            np.round(np.std(np.array(data_auc)), 2)) + "\n")
    return np.round(np.mean(np.array(data_auc)), 2)

def eval_summary(folname, outfiles):
    # folname = sys.argv[1]
    # matplotlib.rc('pdf', fonttype=42)
    num_files = len(outfiles)
    R = []
    for file in outfiles:
        # res = pickle.load(open(folname + "/fold_{}/val_results.pkl".format(i + 1), 'rb'))
        # res = pickle.load(open(scores))
        res = joblib.load(file)
        R.append(res)

    # Plot ROC curves
    clr_1 = 'tab:green'
    clr_2 = 'tab:green'
    clr_3 = 'k'
    data_x, data_y, data_auc = [], [], []
    for i in range(num_files):
        data_x.append(R[i]['FPR'].tolist())
        data_y.append(R[i]['TPR'].tolist())
        data_auc.append(R[i]['AUC'] * 100)
        plt.plot(data_x[i], data_y[i], label='V-' + str(i + 1) + ', auc=' + str(np.round(data_auc[i], 2)), c=clr_1,
                 alpha=0.2)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    plt.plot(np.mean(data_x, axis=0), np.mean(data_y, axis=0),
             label='AVG, auc=' + str(np.round(np.mean(np.array(data_auc)), 2)), c=clr_2, alpha=1, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', label='chance', c=clr_3, alpha=.5)
    plt.legend(loc='lower right', frameon=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
    plt.text(0, 1, 'PATIENT-LEVEL ROC', color='gray', fontsize=12)

    plt.gca().set_xlabel('FALSE POSITIVE RATE')
    plt.gca().set_ylabel('TRUE POSITIVE RATE')
    plt.savefig(os.path.join(folname, 'val_roc_plot.pdf'), bbox_inches='tight')
    plt.close()

    sensitivities = [R[i]['sensitivity'] * 100 for i in range(num_files)]
    specificities = [R[i]['specificity'] * 100 for i in range(num_files)]

    with open(os.path.join(folname, 'val_summary_metrics.txt'), 'w') as f:
        f.write("Sensitivities: " + " ".join([str(round(item, 2)) for item in sensitivities]) + "\n")
        f.write("Specificities: " + " ".join([str(round(item, 2)) for item in specificities]) + "\n")
        f.write("AUCs: " + " ".join([str(round(item, 2)) for item in data_auc]) + "\n")
        f.write(
            "Average sensitivity: " + str(np.round(np.mean(np.array(sensitivities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(sensitivities)), 2)) + "\n")
        f.write(
            "Average specificity: " + str(np.round(np.mean(np.array(specificities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(specificities)), 2)) + "\n")
        f.write("Average AUC: " + str(np.round(np.mean(np.array(data_auc)), 2)) + " standard deviation:" + str(
            np.round(np.std(np.array(data_auc)), 2)) + "\n")

def get_test_filename_converter():
    metadata = open('DiCOVA_Test/metadata.csv', 'r')
    Lines = metadata.readlines()
    full_dict = {'breathing': {}, 'cough': {}, 'speech': {}, 'fusion': {}}
    for line in Lines[1:]:  # first line is the category labels
        line = line[:-1]
        pieces = line.split(' ')
        breathing_ID = pieces[0]
        cough_ID = pieces[1]
        speech_ID = pieces[2]
        fusion_ID = pieces[3]
        datum = {'breathing': breathing_ID, 'cough': cough_ID, 'speech': speech_ID, 'fusion': fusion_ID}
        full_dict['breathing'][breathing_ID] = datum
        full_dict['cough'][cough_ID] = datum
        full_dict['speech'][speech_ID] = datum
        full_dict['fusion'][fusion_ID] = datum
    return full_dict


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

class Metadata(object):
    def __init__(self, test=False, clinical=False):
        """"""
        self.config = get_config()
        self.load_dicova()
        if test:
            self.load_TEST_dicova()
        if clinical:
            self.load_CLINICAL_dicova()

    def load_dicova(self):
        lines = []
        with open(os.path.join(self.config.directories.dicova_root, 'metadata.csv'), newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            for i, row in enumerate(reader):
                if i > 0:
                    lines.append(row)
        metadata = {}
        for line in lines:
            metadata[line[0]] = {'Covid_status': line[1], 'Gender': line[2]}
        self.dicova_metadata = metadata

    def load_TEST_dicova(self):
        lines = []
        with open(os.path.join(self.config.directories.dicova_test_root, 'metadata.csv'), newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            for i, row in enumerate(reader):
                if i > 0:
                    lines.append(row)
        metadata = {}
        for line in lines:
            breathing_ID = line[0]
            cough_ID = line[1]
            speech_ID = line[2]
            fusion_ID = line[3]
            covid_data = {'Covid_status': line[4], 'Gender': line[5]}
            metadata[breathing_ID] = covid_data
            metadata[cough_ID] = covid_data
            metadata[speech_ID] = covid_data
            metadata[fusion_ID] = covid_data
        self.dicova_metadata = metadata

    def load_CLINICAL_dicova(self):
        self.clinical_feats = {}
        for part in ['Test', 'Train']:
            lines = []
            temp_dict = {}
            with open(os.path.join('Clinical_feats', 'Concat_Prob_' + part + '.csv'), newline='') as f:
                reader = csv.reader(f, delimiter=' ')
                for i, row in enumerate(reader):
                    if i > 0:
                        lines.append(row)
                    else:
                        row_headers = row
            for line in lines:
                line = line[0].split(',')
                filename = line[0].split('.')[0]
                dry = float(line[1])
                long = float(line[2])
                short = float(line[3])
                wet = float(line[4])
                wheeze = float(line[5])
                temp_dict[filename] = np.asarray([dry, long, short, wet, wheeze])

            self.clinical_feats[part] = temp_dict

    def get_metadata(self, file, dataset='DiCOVA'):
        if dataset == 'DiCOVA':
            name = file.split('/')[-1]
            name = name.split('.')[0]
            metadata = self.dicova_metadata[name]
            return metadata

    def get_feature_metadata(self, file, dataset='DiCOVA'):
        if dataset == 'DiCOVA':
            name = file.split('/')[-1]
            name = name.split('_')[0]
            metadata = self.dicova_metadata[name]
            return metadata

    def get_clinical_metadata(self, file, Train=True):
        name = file.split('/')[-1]
        name = name.split('_')[0]
        if Train:
            metadata = self.clinical_feats['Train'][name]
        else:
            metadata = self.clinical_feats['Test'][name]
        return metadata

class Partition(object):
    def __init__(self):
        """"""
        self.config = get_config()
        self.metadata = Metadata()
        self.load_dicova()

    def load_dicova(self):
        print('Loading DiCOVA partition information...')
        files = collect_files(os.path.join(self.config.directories.dicova_root, 'LISTS'))
        folds = {}
        for file in files:
            name = file.split('/')[-1][:-4]
            pieces = name.split('_')
            train_val = pieces[0]
            fold = pieces[1]
            if fold in folds:
                folds[fold][train_val] = file
            else:
                folds[fold] = {train_val: file}
        fold_files = {}
        for fold, partition in folds.items():
            train = partition['train']
            with open(train) as f:
                train_files = f.readlines()
            train_files = [x.strip() for x in train_files]
            """Get train positives and train negatives"""
            train_pos = []
            train_neg = []
            for file in train_files:
                meta = self.metadata.get_metadata(file, dataset='DiCOVA')
                if meta['Covid_status'] == 'p':
                    train_pos.append(file)
                elif meta['Covid_status'] == 'n':
                    train_neg.append(file)
            val = partition['val']
            with open(val) as f:
                val_files = f.readlines()
            val_files = [x.strip() for x in val_files]
            val_pos = []
            val_neg = []
            for file in val_files:
                meta = self.metadata.get_metadata(file, dataset='DiCOVA')
                if meta['Covid_status'] == 'p':
                    val_pos.append(file)
                elif meta['Covid_status'] == 'n':
                    val_neg.append(file)
            fold_files[fold] = {'train_pos': train_pos, 'train_neg': train_neg,
                                'val_pos': val_pos, 'val_neg': val_neg}
        self.dicova_partition = fold_files
        print('Done.')

def main():
    """"""
    # files = collect_files('DiCOVA/AUDIO/breathing')
    # meta = Metadata()
    # for file in files:
    #     metadata = meta.get_metadata(file, dataset='DiCOVA')
    # partition = Partition()





if __name__ == "__main__":
    main()