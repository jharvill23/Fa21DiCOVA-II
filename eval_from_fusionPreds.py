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
from dataset import DiCOVA_Dataset, DiCOVA_Dataset_Fusion, DiCOVA_Dataset_Margin, LibriSpeech_Dataset, COUGHVID_Dataset
import torch.nn.functional as F
import copy
from scipy.special import softmax
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

test_preds_folder = os.path.join('evals', 'fusion_from_preds2_bestvalloss_confidence_boosting_0dot6_and_0dot4')
if not os.path.exists(test_preds_folder):
    os.mkdir(test_preds_folder)

### Using best validation LOSS as the metric ###
# Manually put the best ones here from tensorboard
val_score_paths = ['exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold1/val_scores/scores_30',
                   'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold2/val_scores/scores_30',
                   'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold3/val_scores/scores_40',
                   'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold4/val_scores/scores_110',
                   'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold0/val_scores/scores_30']


# Manually put the best ones here from tensorboard
test_score_paths = ['exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold1/val_scores/test_scores_30.pkl',
                   'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold2/val_scores/test_scores_30.pkl',
                   'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold3/val_scores/test_scores_40.pkl',
                   'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold4/val_scores/test_scores_110.pkl',
                   'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold0/val_scores/test_scores_30.pkl']

# ### Using best validation AUC as the metric ###
#
# val_score_paths = ['exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold1/val_scores/scores_110',
#                    'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold2/val_scores/scores_70',
#                    'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold3/val_scores/scores_20',
#                    'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold4/val_scores/scores_180',
#                    'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold0/val_scores/scores_80']
#
# # Manually put the best ones here from tensorboard
# test_score_paths = ['exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold1/val_scores/test_scores_110.pkl',
#                    'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold2/val_scores/test_scores_70.pkl',
#                    'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold3/val_scores/test_scores_20.pkl',
#                    'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold4/val_scores/test_scores_180.pkl',
#                    'exps/fusion_SAVE_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy_frompreds_fold0/val_scores/test_scores_80.pkl']

"""Put all the val scores into one file"""
all_val_scores_dict = {}
for score_path in val_score_paths:
    file1 = open(score_path, 'r')
    Lines = file1.readlines()
    for line in Lines:
        line = line[:-1]
        """Need to remove _MODALITY from filename"""
        pieces = line.split(' ')
        filename = pieces[0]
        filename = filename.split('_')[0]
        score = pieces[1]
        # all_val_scores.append(filename + ' ' + score)
        all_val_scores_dict[filename] = score
"""Need to reorder based on provided order..."""
# Reorder files
all_val_scores = []
example_val_order = open("example_submission_track4/val_answer.csv", 'r')
Lines = example_val_order.readlines()
for line in Lines:
    line = line[:-1]
    pieces = line.split(' ')
    filename = pieces[0]
    my_score = all_val_scores_dict[filename]
    all_val_scores.append(filename + ' ' + my_score)
# all_val_scores = sorted(all_val_scores)  # Don't do this, they want the files in order of fold 0 to fold 4
val_score_path = os.path.join(test_preds_folder, 'val_answer.csv')
with open(val_score_path, 'w') as f:
    for item in all_val_scores:
        f.write("%s\n" % item)

"""Test data now"""
test_scores = {}
for file in test_score_paths:
    scores = utils.load(file)
    for key, value in scores.items():
        if key not in test_scores:
            test_scores[key] = [value]
        else:
            test_scores[key].append(value)
"""Need to take the mean of each list"""
total_upcounts = 0
total_downcounts = 0
for key, value in test_scores.items():
    """Let's push the more confident ones to the extreme"""
    push_factor = 0.9
    up_threshold = 0.6  # was 0.8
    down_threshold = 0.4  # was 0.2
    upcount = 0
    downcount = 0
    upcount_scores = []
    downcount_scores = []
    for x in value:
        if x >= up_threshold:
            upcount += 1
            upcount_scores.append(x)
        elif x <= down_threshold:
            downcount += 1
            downcount_scores.append(x)
    if upcount >= 4:
        original_prob = np.mean(np.asarray(upcount_scores))
        new_prob = original_prob + (1-original_prob)*push_factor
        value = new_prob
        total_upcounts += 1
    elif downcount >= 4:
        original_prob = np.mean(np.asarray(downcount_scores))
        new_prob = original_prob - (original_prob) * push_factor
        value = new_prob
        total_downcounts += 1
    else:
        value = np.mean(np.asarray(value))
    test_scores[key] = float(value)

"""Let's analyze the distribution of scores"""
distribution = {'confident_no': [], 'confident_yes': [], 'unsure_no': [], 'unsure_yes': []}
for key, value in test_scores.items():
    if 0 <= value <= 0.1:
        distribution['confident_no'].append({key: value})
    elif 0.1 < value < 0.5:
        distribution['unsure_no'].append({key: value})
    elif 0.5 <= value < 0.9:
        distribution['unsure_yes'].append({key: value})
    else:
        distribution['confident_yes'].append({key: value})



file_FINAL_scores_dict = {}
for key, mean in test_scores.items():
    # file_FINAL_scores.append(key + ' ' + str(sum))
    file_FINAL_scores_dict[key] = str(mean)
"""Need to reorder based on provided order..."""
file_FINAL_scores = []
example_val_order = open("example_submission_track4/test_answer.csv", 'r')
Lines = example_val_order.readlines()
file_converter = utils.get_test_filename_converter()
for line in Lines:
    line = line[:-1]
    pieces = line.split(' ')
    filename = pieces[0]
    # """The example given is the breathing one, convert to the other modality name"""
    # modality_filename = file_converter['breathing'][filename][args.MODALITY]
    my_score = file_FINAL_scores_dict[filename]
    file_FINAL_scores.append(filename + ' ' + my_score)
# Reorder files
FINAL_score_path = os.path.join(test_preds_folder, 'test_answer.csv')
with open(FINAL_score_path, 'w') as f:
    for item in file_FINAL_scores:
        f.write("%s\n" % item)