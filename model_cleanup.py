"""This script REMOVES models it chooses not to keep based on evaluation metrics.
   BE VERY CAREFUL SINCE THE MODELS ARE REMOVED PERMANENTLY AND CANNOT BE RECOVERED."""

# code to read tensorboard logfile from:
# https://stackoverflow.com/questions/36700404/tensorflow-opening-log-data-written-by-summarywriter/45899735#45899735

import os
import random

from tqdm import tqdm
import numpy as np
import joblib
import torch
import torch.nn as nn
import model
import shutil
from utils import collect_files
import utils
from torch.utils import data
import json
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from dataset import DiCOVA_Dataset, DiCOVA_Dataset_Margin, LibriSpeech_Dataset, COUGHVID_Dataset
import torch.nn.functional as F
import copy
from scipy.special import softmax
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(args):
    tb_file = utils.collect_files(os.path.join(args.EXP_DIR, 'logs'))
    assert len(tb_file) == 1
    tb_file = tb_file[0]
    # path = 'dummy_exps/speech_yespretrain_notimewarp_yesspecaug_spect_crossent_fold1/logs/events.out.tfevents.1629766099.ifp-52.22672.0'
    event_acc = EventAccumulator(tb_file)
    event_acc.Reload()
    # Show all tags in the log file
    print(event_acc.Tags())

    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    w_times, step_nums, vals = zip(*event_acc.Scalars(args.SAVE_METRIC))
    """Make a dictionary with step_nums as keys and vals as values"""
    data_dict = {}
    for i, _ in enumerate(w_times):
        data_dict[step_nums[i]] = vals[i]
    if args.MAXIMIZE:
        reverse = True
    else:
        reverse = False
    sorted_data_dict = {k: v for k, v in sorted(data_dict.items(), key=lambda item: item[1], reverse=reverse)}
    model_count = 0
    keep_models = []
    for timestep, metric_value in sorted_data_dict.items():
        if model_count < args.NUM_SAVE_MODELS:
            keep_models.append({'timestep': timestep, 'metric_value': metric_value})
        model_count += 1
    """Convert timesteps into model paths"""
    keep_model_paths = []
    for pair in keep_models:
        timestep = pair['timestep']
        filename = str(timestep) + '-G.ckpt'
        model_path = os.path.join(args.EXP_DIR, 'models', filename)
        assert os.path.exists(model_path)
        keep_model_paths.append(model_path)
    """Now we want to collect all the models in the 'models' directory, remove all elements of 'keep_model_paths' from
       the list, then delete all models left in the list."""
    all_models = utils.collect_files(os.path.join(args.EXP_DIR, 'models'))
    delete_models = list(set(all_models).difference(set(keep_model_paths)))
    print('Deleting worst-performing models...')
    for model in tqdm(delete_models):
        assert os.path.exists(model)
        os.remove(model)
    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to train classifier')
    parser.add_argument('--EXP_DIR', type=str, default='exps/speech_MF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_fold0')
    parser.add_argument('--NUM_SAVE_MODELS', type=int, default=5)
    parser.add_argument('--SAVE_METRIC', type=str, default='AUC')  # Be very careful with this!!!
    parser.add_argument('--MAXIMIZE', type=utils.str2bool, default=True)  # Be very careful with this!!!
    args = parser.parse_args()
    main(args)