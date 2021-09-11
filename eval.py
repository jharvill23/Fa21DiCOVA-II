import os
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
from dataset import DiCOVA_Dataset
import torch.nn.functional as F
import copy
from scipy.special import softmax
import argparse
from train import Solver
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

config = utils.get_config()

if not os.path.exists(config.directories.exps):
    os.mkdir(config.directories.exps)

def get_best_models(tb_file, args, exp_dir):
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
        if model_count < args.ENSEMBLE_NUM_MODELS:
            keep_models.append({'timestep': timestep, 'metric_value': metric_value})
        model_count += 1
    """Convert timesteps into model paths"""
    keep_model_paths = []
    for pair in keep_models:
        timestep = pair['timestep']
        filename = str(timestep) + '-G.ckpt'
        model_path = os.path.join(exp_dir, 'models', filename)
        assert os.path.exists(model_path)
        keep_model_paths.append(model_path)
    return keep_model_paths

def get_model_predictions(self):
    """Something is very, very wrong here and I don't know what it is..."""
    self.evaluation_dir = os.path.join(self.EVAL_DIR)
    if not os.path.exists(self.evaluation_dir):
        os.mkdir(self.evaluation_dir)
    self.trial_dir = os.path.join(self.EVAL_DIR, self.input_TRIAL)
    if not os.path.exists(self.trial_dir):
        os.mkdir(self.trial_dir)
    self.fold_dir = os.path.join(self.trial_dir, self.args.FOLD)
    if not os.path.isdir(self.fold_dir):
        os.mkdir(self.fold_dir)
    self.eval_type_dir = os.path.join(self.fold_dir, 'val')
    if not os.path.isdir(self.eval_type_dir):
        os.mkdir(self.eval_type_dir)
    self.specific_model_dir = os.path.join(self.eval_type_dir, self.model_number)
    if not os.path.isdir(self.specific_model_dir):
        os.mkdir(self.specific_model_dir)
    """Get train/test"""
    train, val = self.get_train_test()
    train_gen, val_gen = self.get_train_val_generators(train, val)

    self.restore_model(self.args.RESTORE_PATH)

    # val_files_list = val['positive'] + val['negative']

    ground_truth = []
    pred_scores = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    val_loss = 0
    for batch_number, batch_data in tqdm(enumerate(val_gen)):
        # try:
        files = batch_data['files']
        self.G = self.G.eval()
        predictions = self.forward_pass(batch_data=batch_data, margin_config=False)
        loss = self.compute_loss(predictions=predictions, batch_data=batch_data, crossentropy_overwrite=True)
        val_loss += loss.sum().item()

        if not self.args.PRETRAINING:
            predictions = predictions.detach().cpu().numpy()
            max_preds = np.argmax(predictions, axis=1)
            scores = softmax(predictions, axis=1)
            pred_value = [self.index2class[x] for x in max_preds]

            info = [self.metadata.get_feature_metadata(x) for x in files]

            for i, file in enumerate(files):
                filekey = file.split('/')[-1][:-4]
                gt = info[i]['Covid_status']
                score = scores[i, self.class2index['p']]
                ground_truth.append(filekey + ' ' + gt)
                pred_scores.append(filekey + ' ' + str(score))

            for i, entry in enumerate(info):
                if entry['Covid_status'] == 'p':
                    if pred_value[i] == 'p':
                        TP += 1
                    elif pred_value[i] == 'n':
                        FN += 1
                elif entry['Covid_status'] == 'n':
                    if pred_value[i] == 'n':
                        TN += 1
                    elif pred_value[i] == 'p':
                        FP += 1

    # except:
    #     """"""
    if not self.args.PRETRAINING:
        """Sort the lists in alphabetical order"""
        ground_truth.sort()
        pred_scores.sort()

        """Write the files"""
        gt_path = os.path.join(self.specific_model_dir, 'val_labels')
        score_path = os.path.join(self.specific_model_dir, 'scores')

        for path in [gt_path, score_path]:
            with open(path, 'w') as f:
                if path == gt_path:
                    for item in ground_truth:
                        f.write("%s\n" % item)
                elif path == score_path:
                    for item in pred_scores:
                        f.write("%s\n" % item)
        try:
            out_file_path = os.path.join(self.specific_model_dir, 'outfile.pkl')
            utils.scoring(refs=gt_path, sys_outs=score_path, out_file=out_file_path)
            auc = utils.summary(folname=self.specific_model_dir, scores=out_file_path, iterations=0)
        except:
            auc = 0

        if TP + FP > 0:
            Prec = TP / (TP + FP)
        else:
            Prec = 0
        if TP + FN > 0:
            Rec = TP / (TP + FN)
        else:
            Rec = 0

        acc = (TP + TN) / (TP + TN + FP + FN)

        return val_loss, Prec, Rec, acc, auc
    else:
        return val_loss, 0, 0, 0, 0

def main(args):
    if not os.path.isdir(args.EVAL_DIR):
        os.mkdir(args.EVAL_DIR)

    dump_root = os.path.join(args.EVAL_DIR, args.TRIAL)
    if not os.path.isdir(dump_root):
        os.mkdir(dump_root)
    """First collect the best models automatically (hand-picked them for first DiCOVA challenge)"""
    best_models = {}
    for fold in ['0', '1', '2', '3', '4']:
        exp_dir = os.path.join('exps', args.TRIAL + '_fold' + fold)
        tb_file = utils.collect_files(os.path.join(exp_dir, 'logs'))
        assert len(tb_file) == 1
        tb_file = tb_file[0]
        best_models_fold = get_best_models(tb_file=tb_file, args=args, exp_dir=exp_dir)
        best_models[fold] = best_models_fold

    outfiles = []
    input_TRIAL = args.TRIAL  # need to modify this variable for use in solver
    for fold in ['1', '0', '2', '3', '4']:
        paths = []
        eval_paths = []
        for checkpoint in best_models[fold]:
            """"""
            model_number = checkpoint.split('/')[-1][:-7]
            """"""
            args.RESTORE_PATH = checkpoint
            args.FOLD = fold
            args.TRIAL = os.path.join(input_TRIAL + '_fold' + fold)
            solver = Solver(config=config, args=args)
            solver.EVAL_DIR = args.EVAL_DIR
            solver.model_number = model_number
            solver.input_TRIAL = input_TRIAL
            _ = get_model_predictions(solver)

            specific_path, fold_score_path, eval_type_dir = solver.val_scores_ensemble(model_num=model_number)
            # eval_score_path, eval_ensemb_type_dir = solver.eval_ensemble(model_num=model_number)

            paths.append(specific_path)
            # eval_paths.append(eval_score_path)
        """We get scores from individual models. Need to load those scores and take mean"""
        file_scores = {}
        for dictionary in paths:
            score_path = dictionary['score_path']
            file1 = open(score_path, 'r')
            Lines = file1.readlines()
            for line in Lines:
                line = line[:-1]
                pieces = line.split(' ')
                filename = pieces[0]
                score = pieces[1]
                if filename not in file_scores:
                    file_scores[filename] = [score]
                else:
                    file_scores[filename].append(score)
        file_final_scores = []
        for key, score_list in file_scores.items():
            sum = 0
            for score in score_list:
                sum += float(score)
            sum = sum / len(score_list)
            file_final_scores.append(key + ' ' + str(sum))
        with open(fold_score_path, 'w') as f:
            for item in file_final_scores:
                f.write("%s\n" % item)

        # eval_file_scores = {}
        # for x in eval_paths:
        #     score_path = x
        #     file1 = open(score_path, 'r')
        #     Lines = file1.readlines()
        #     for line in Lines:
        #         line = line[:-1]
        #         pieces = line.split(' ')
        #         filename = pieces[0]
        #         score = pieces[1]
        #         if filename not in eval_file_scores:
        #             eval_file_scores[filename] = [score]
        #         else:
        #             eval_file_scores[filename].append(score)
        # eval_file_final_scores = []
        # for key, score_list in eval_file_scores.items():
        #     sum = 0
        #     for score in score_list:
        #         sum += float(score)
        #     sum = sum / len(score_list)
        #     eval_file_final_scores.append(key + ' ' + str(sum))
        # with open(os.path.join(eval_ensemb_type_dir, 'scores'), 'w') as f:
        #     for item in eval_file_final_scores:
        #         f.write("%s\n" % item)

        outfile_path = os.path.join(eval_type_dir, 'outfile.pkl')
        utils.scoring(refs=paths[0]['gt_path'], sys_outs=fold_score_path, out_file=outfile_path)
        # outfile_path = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', fold, 'val', 'outfile.pkl')
        outfiles.append(outfile_path)
    folder = os.path.join(config.directories.exps, args.TRIAL, 'evaluations')
    utils.eval_summary(folname=folder, outfiles=outfiles)

    # """Take mean of probability for each fold on test data"""
    # val_scores = []
    # test_scores = {}
    #
    # # test_folds_to_include = ['1', '2', '4']
    # # test_folds_to_include = ['1']
    # test_folds_to_include = ['1', '2', '3', '4', '5']
    # for fold in ['1', '2', '3', '4', '5']:
    #     val_file = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', fold, 'val', 'scores')
    #     test_file = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', fold, 'blind', 'scores')
    #
    #     file1 = open(val_file, 'r')
    #     Lines = file1.readlines()
    #     for line in Lines:
    #         line = line[:-1]
    #         pieces = line.split(' ')
    #         filename = pieces[0]
    #         score = pieces[1]
    #         val_scores.append(filename + ' ' + str(score))
    #
    #     if fold in test_folds_to_include:
    #         file1 = open(test_file, 'r')
    #         Lines = file1.readlines()
    #         for line in Lines:
    #             line = line[:-1]
    #             pieces = line.split(' ')
    #             filename = pieces[0]
    #             score = pieces[1]
    #             if filename not in test_scores:
    #                 test_scores[filename] = [score]
    #             else:
    #                 test_scores[filename].append(score)
    # test_final_scores = []
    # for key, score_list in test_scores.items():
    #     sum = 0
    #     for score in score_list:
    #         sum += float(score)
    #     sum = sum/len(score_list)
    #     test_final_scores.append(key + ' ' + str(sum))
    # test_score_path = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', 'test_scores.txt')
    # val_score_path = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', 'val_scores.txt')
    # with open(test_score_path, 'w') as f:
    #     for item in test_final_scores:
    #         f.write("%s\n" % item)
    # with open(val_score_path, 'w') as f:
    #     for item in val_scores:
    #         f.write("%s\n" % item)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to train classifier')
    parser.add_argument('--EVAL_DIR', type=str, default='evals')  # dump everything to this new folder
    parser.add_argument('--TRIAL', type=str, default='speech_MF_CNN_yespretrainCNN_notimewarp_yesspecaug_mfcc_crossentropy')  # add the folds in the loop
    parser.add_argument('--TRAIN', action='store_true', default=False)
    parser.add_argument('--LOAD_MODEL', action='store_true', default=True)
    parser.add_argument('--FOLD', type=str, default='None')  # you must set it later so you want error if it's None
    parser.add_argument('--ENSEMBLE_NUM_MODELS', type=str, default=1)
    parser.add_argument('--SAVE_METRIC', type=str, default='AUC')
    parser.add_argument('--MAXIMIZE', type=utils.str2bool, default=True)
    parser.add_argument('--VAL_OR_TEST', type=str, default='VAL')  # VAL, TEST
    parser.add_argument('--RESTORE_PATH', type=str, default='')  # you must set it later so you want error if it's empty
    parser.add_argument('--RESTORE_PRETRAINER_PATH', type=str, default='exps/speech_pretrain_20ff_mfcc_APC_CNN/models/90000-G.ckpt')
    parser.add_argument('--PRETRAINING', type=utils.str2bool, default=False)
    parser.add_argument('--FROM_PRETRAINING', type=utils.str2bool, default=True)
    parser.add_argument('--LOSS', type=str, default='crossentropy')  # doesn't matter for eval
    parser.add_argument('--MODALITY', type=str, default='speech')
    parser.add_argument('--FEAT_DIR', type=str, default='feats/DiCOVA')  # will put test data in separate directory
    parser.add_argument('--POS_NEG_SAMPLING_RATIO', type=float, default=1.0)  # doesn't matter for eval
    parser.add_argument('--TIME_WARP', type=utils.str2bool, default=False)
    parser.add_argument('--MODEL_INPUT_TYPE', type=str, default='mfcc')  # spectrogram, energy, mfcc
    parser.add_argument('--MODEL_TYPE', type=str, default='CNN')  # CNN, LSTM
    parser.add_argument('--TRAIN_DATASET', type=str, default='DiCOVA')  # DiCOVA, COUGHVID, LibriSpeech
    parser.add_argument('--TRAIN_CLIP_FRACTION', type=float,
                        default=0.3)  # randomly shorten clips during training (speech, breathing)
    parser.add_argument('--INCLUDE_MF', type=utils.str2bool, default=True)  # include male/female metadata
    parser.add_argument('--USE_TENSORBOARD', type=utils.str2bool, default=False)  # don't make tb file in this script
    args = parser.parse_args()
    main(args)