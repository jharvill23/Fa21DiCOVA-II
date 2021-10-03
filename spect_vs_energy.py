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
from dataset import DiCOVA_Dataset, DiCOVA_Dataset_Fusion, DiCOVA_Dataset_Fusion_from_Preds, DiCOVA_Dataset_Margin, LibriSpeech_Dataset, COUGHVID_Dataset
import torch.nn.functional as F
import copy
from scipy.special import softmax
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


config = utils.get_config()

if not os.path.exists(config.directories.exps):
    os.mkdir(config.directories.exps)


class Solver(object):
    """Solver"""

    def __init__(self, config, args):
        """Initialize configurations."""

        self.config = config
        self.args = args
        self.fold = self.args.FOLD
        self.val_folds = os.path.join('val_folds', 'fold_' + self.fold, 'val_labels')

        # Training configurations.
        # if self.args.PRETRAINING:
        #     self.model_hyperparameters = self.config.pretraining2
        # else:
        #     if self.args.FROM_PRETRAINING:
        #         self.model_hyperparameters = self.config.post_pretraining_classifier
        #     else:
        #         self.model_hyperparameters = self.config.classifier
        self.model_hyperparameters = self.config.fusion

        self.g_lr = self.model_hyperparameters.lr
        self.torch_type = torch.float32

        # Miscellaneous.
        self.use_tensorboard = args.USE_TENSORBOARD
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(0) if self.use_cuda else 'cpu')

        # Directories.
        # trial = 'finetuning_trial_10_with_scaling_and_auc_plots_10_ff_pretraining_coughvid_specaug_prob_0dot7'
        trial = self.args.TRIAL
        self.exp_dir = os.path.join(self.config.directories.exps, trial)
        if not os.path.isdir(self.exp_dir):
            os.mkdir(self.exp_dir)
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        self.model_save_dir = os.path.join(self.exp_dir, 'models')
        # self.train_data_dir = self.config.directories.features
        self.predict_dir = os.path.join(self.exp_dir, 'predictions')
        self.images_dir = os.path.join(self.exp_dir, 'images')
        self.val_scores_dir = os.path.join(self.exp_dir, 'val_scores')

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        if not os.path.isdir(self.predict_dir):
            os.mkdir(self.predict_dir)
        if not os.path.isdir(self.images_dir):
            os.mkdir(self.images_dir)
        if not os.path.isdir(self.val_scores_dir):
            os.mkdir(self.val_scores_dir)

        """Training Data"""
        self.metadata = utils.Metadata()
        self.partition = utils.Partition()

        """Partition file"""
        if self.args.TRAIN:
            # copy config
            shutil.copy(src='config.yml', dst=os.path.join(self.exp_dir, 'config.yml'))
            # save args to file
            args_dump_path = os.path.join(self.exp_dir, 'args.txt')
            with open(args_dump_path, 'w') as f:
                json.dump(args.__dict__, f, indent=2)

        # Step size.
        self.log_step = self.model_hyperparameters.log_step
        self.model_save_step = self.model_hyperparameters.model_save_step

        # Get the paths for the best model from each fold based on the tensorboard log files
        self.get_best_modality_models()

    def tensorboard_best_models(self, tb_file, exp_root):
        event_acc = EventAccumulator(tb_file)
        event_acc.Reload()
        # Show all tags in the log file
        print(event_acc.Tags())

        # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
        w_times, step_nums, vals = zip(*event_acc.Scalars(self.args.SAVE_METRIC))
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
            if model_count < 1:
                keep_models.append({'timestep': timestep, 'metric_value': metric_value})
            model_count += 1
        """Convert timesteps into model paths"""
        keep_model_paths = []
        for pair in keep_models:
            timestep = pair['timestep']
            filename = str(timestep) + '-G.ckpt'
            model_path = os.path.join(exp_root, 'models', filename)
            assert os.path.exists(model_path)
            metric_value = pair['metric_value']
            datum = {'model_path': model_path, 'metric_value': metric_value}
            keep_model_paths.append(datum)
        return keep_model_paths

    def get_best_modality_models(self):
        self.best_modality_models = {}
        self.best_metrics = {}
        for modality in ['speech', 'cough', 'breathing']:
            if modality == 'speech':
                tb_dir = os.path.join(self.args.RESTORE_SPEECH_FINETUNED_EXP_PATH + '_fold' + self.args.FOLD, 'logs')
                exp_root = os.path.join(self.args.RESTORE_SPEECH_FINETUNED_EXP_PATH + '_fold' + self.args.FOLD)
            elif modality == 'cough':
                tb_dir = os.path.join(self.args.RESTORE_COUGH_FINETUNED_EXP_PATH + '_fold' + self.args.FOLD, 'logs')
                exp_root = os.path.join(self.args.RESTORE_COUGH_FINETUNED_EXP_PATH + '_fold' + self.args.FOLD)
            elif modality == 'breathing':
                tb_dir = os.path.join(self.args.RESTORE_BREATHING_FINETUNED_EXP_PATH + '_fold' + self.args.FOLD, 'logs')
                exp_root = os.path.join(self.args.RESTORE_BREATHING_FINETUNED_EXP_PATH + '_fold' + self.args.FOLD)
            tb_path = utils.collect_files(tb_dir)
            assert len(tb_path) == 1
            tb_path = tb_path[0]
            best_model_ = self.tensorboard_best_models(tb_file=tb_path, exp_root=exp_root)[0]
            best_model = best_model_['model_path']
            best_metric = best_model_['metric_value']
            self.best_modality_models[modality] = best_model
            self.best_metrics[modality] = best_metric

    def get_absolute_filepaths(self, files):
        new_files = []
        for file in files:
            new_name = os.path.join(self.args.FEAT_DIR, file + '_' + self.args.MODALITY + '.pkl')
            assert os.path.exists(new_name)
            new_files.append(new_name)
        return new_files

    def get_absolute_filepaths_fusion(self, files):
        new_files = []
        for file in files:
            modality_triple = {}
            for modality in ['speech', 'cough', 'breathing']:
                new_name = os.path.join(self.args.FEAT_DIR, file + '_' + modality + '.pkl')
                assert os.path.exists(new_name)
                modality_triple[modality] = new_name
            new_files.append(modality_triple)
        return new_files

    def upsample_positive_class(self, negative, positive):
        """Balance the data by seeing positive COVID examples more often"""
        random.shuffle(negative)
        random.shuffle(positive)
        long_negative = []
        long_positive = []
        for i in range(30):
            long_negative += negative
            long_positive += positive
        negative_iter = iter(long_negative)
        positive_iter = iter(long_positive)
        resampled_files = []
        positive_probability = self.args.POS_NEG_SAMPLING_RATIO / (1.0 + self.args.POS_NEG_SAMPLING_RATIO)
        pos_file_count = 0
        neg_file_count = 0
        for i in range(len(negative)*3):
            random_draw = random.uniform(0, 1)
            if random_draw < positive_probability:
                resampled_files.append(next(positive_iter))
                pos_file_count += 1
            else:
                resampled_files.append(next(negative_iter))
                neg_file_count += 1
        return resampled_files

    def all_pairs(self, negative, positive):
        """Make all possible pairs of positive and negative samples"""
        random.shuffle(negative)
        random.shuffle(positive)
        pairs = []
        for neg in negative:
            for pos in positive:
                pairs.append({'positive': pos, 'negative': neg})
        random.shuffle(pairs)
        return pairs

    def get_model_num_from_path(self, path):
        model_name = path.split('/')[-1]
        model_num = model_name.split('-')[0]
        return model_num

    def get_val_folder_from_model_path(self, path):
        path_list = path.split('/')[:-2]
        exp_dir = os.path.join(*path_list)
        return os.path.join(exp_dir, 'val_scores')

    def get_train_test(self):
        if self.args.TRAIN_DATASET == 'DiCOVA':
            partition = self.partition.dicova_partition
            partition = partition[self.fold]

            # Get absolute filepaths depending on modality
            partition['train_pos'] = self.get_absolute_filepaths_fusion(partition['train_pos'])
            partition['train_neg'] = self.get_absolute_filepaths_fusion(partition['train_neg'])
            partition['val_pos'] = self.get_absolute_filepaths_fusion(partition['val_pos'])
            partition['val_neg'] = self.get_absolute_filepaths_fusion(partition['val_neg'])

            hidden_states = {'speech': {}, 'breathing': {}, 'cough': {}}

            """Need to load appropriate hidden state vectors here for each modality (lots of stuff)
               All hidden states are pickled. Intermediate probabilities are as follows:
               Train/test are pickled and val is in form of text file"""
            for modality in ['speech', 'breathing', 'cough']:
                val_scores_path = self.get_val_folder_from_model_path(self.best_modality_models[modality])
                model_num = self.get_model_num_from_path(self.best_modality_models[modality])
                hidden_states[modality]['train'] = utils.load(os.path.join(val_scores_path, 'train_hidden_states_' + model_num + '.pkl'))
                hidden_states[modality]['val'] = utils.load(os.path.join(val_scores_path, 'val_hidden_states_' + model_num + '.pkl'))
                hidden_states[modality]['test'] = utils.load(os.path.join(val_scores_path, 'test_hidden_states_' + model_num + '.pkl'))

            self.hidden_states = hidden_states

            train_files = {'positive': partition['train_pos'], 'negative': partition['train_neg']}
            # test_files = {'positive': partition['test_positive'], 'negative': partition['test_negative']}
            val_files = {'positive': partition['val_pos'], 'negative': partition['val_neg']}
            return train_files, val_files
        elif self.args.TRAIN_DATASET == 'COUGHVID':
            """"""
            training_files = utils.collect_files(self.args.FEAT_DIR)
            train_val_split = int(0.985 * len(training_files))
            train_files = training_files[0:train_val_split]
            val_files = training_files[train_val_split:]
            return train_files, val_files
        elif self.args.TRAIN_DATASET == 'LibriSpeech':
            """"""
            training_files = utils.collect_files(self.args.FEAT_DIR)
            train_val_split = int(0.985 * len(training_files))
            train_files = training_files[0:train_val_split]
            val_files = training_files[train_val_split:]
            return train_files, val_files
        elif self.args.TRAIN_DATASET == 'Cambridge':
            """"""
            training_files = utils.collect_files(self.args.FEAT_DIR)
            train_val_split = int(0.97 * len(training_files))
            train_files = training_files[0:train_val_split]
            val_files = training_files[train_val_split:]
            return train_files, val_files

    def forward_pass(self, batch_data, margin_config=False):
        if not margin_config:
            spects = batch_data['spects']
            files = batch_data['files']
            # labels = batch_data['labels']
            # scalers = batch_data['scalers']
            """Get the pretrained intermediate states first for each modality"""
            intermediate = {}
            for modality in ['speech', 'cough', 'breathing']:
                intermediate[modality] = spects[modality]
                if self.args.INCLUDE_MF:
                    mf = batch_data['mf']
                    intermediate[modality] = {'intermediate': intermediate[modality], 'mf': mf}
            """Concatenate the fixed-length representations of all three modalities and pass to fusion network"""
            fusion_input = torch.cat((intermediate['speech'],
                                      intermediate['cough'],
                                      intermediate['breathing']), dim=1)
            predictions = self.G(fusion_input)
            return predictions
        else:
            neg_spects = batch_data['neg_spects']
            pos_spects = batch_data['pos_spects']
            if self.args.FROM_PRETRAINING:
                _, intermediate_neg = self.pretrained(neg_spects)
                _, intermediate_pos = self.pretrained(pos_spects)
            else:
                intermediate_neg = neg_spects
                intermediate_pos = pos_spects

            predictions_neg = self.G(intermediate_neg)
            predictions_pos = self.G(intermediate_pos)
            return predictions_pos, predictions_neg

    def compute_loss(self, predictions, batch_data, crossentropy_overwrite=False):
        if self.args.LOSS == 'APC':
            """Compute Autoregressive Predictive Coding Loss between output features and shifted input features"""
            model_output, intermediate_state = predictions
            input_features = batch_data['spects']
            input_length = input_features.shape[1]
            """Now we need to trim the input features for MSE error between output"""
            input_features = input_features[:, self.config.pretraining2.future_frames:, :]
            model_output = model_output[:, 0: input_length - self.config.pretraining2.future_frames, :]
            loss = F.mse_loss(input=input_features, target=model_output)
        elif self.args.LOSS == 'crossentropy' or crossentropy_overwrite:
            """Compute Cross Entropy Loss"""
            loss_function = nn.CrossEntropyLoss(reduction='none')
            labels = batch_data['labels']
            loss = loss_function(predictions, labels)
            scalers = batch_data['scalers']
            """Multiply loss of positive labels by incorrect scaler"""
            loss = loss * scalers
        elif self.args.LOSS == 'margin':
            pos, neg = predictions
            pos = torch.softmax(pos, dim=1)
            neg = torch.softmax(neg, dim=1)
            # pos_np = pos.detach().cpu().numpy()
            pos_index = utils.get_class2index_and_index2class()[0]['p']
            loss = 0
            for i in range(self.model_hyperparameters.batch_size):
                pos_value = pos[i][pos_index]
                neg_value = neg[i][pos_index]
                zero = torch.zeros(size=(1,)).to(pos_value.device)
                loss_term = torch.max(zero, 0.8 + neg_value - pos_value)  # values range from 0 to 1 so margin 0.8
                loss += loss_term
        return loss

    def val_loss(self, val, iterations):
        val_loss = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        ground_truth = []
        pred_scores = []
        for batch_number, batch_data in tqdm(enumerate(val)):
            # try:
                files = batch_data['files']
                self.G = self.G.eval()
                predictions = self.forward_pass(batch_data=batch_data, margin_config=False)
                loss = self.compute_loss(predictions=predictions, batch_data=batch_data, crossentropy_overwrite=True)
                val_loss += loss.sum().item()

                if not self.args.PRETRAINING:
                    predictions = np.squeeze(predictions.detach().cpu().numpy())
                    max_preds = np.argmax(predictions, axis=1)
                    scores = softmax(predictions, axis=1)
                    pred_value = [self.index2class[x] for x in max_preds]

                    info = [self.metadata.get_feature_metadata(x['speech']) for x in files]

                    for i, file in enumerate(files):
                        filekey = file['speech'].split('/')[-1][:-4]
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
            gt_path = os.path.join(self.val_scores_dir, 'val_labels_' + str(iterations))
            score_path = os.path.join(self.val_scores_dir, 'scores_' + str(iterations))

            for path in [gt_path, score_path]:
                with open(path, 'w') as f:
                    if path == gt_path:
                        for item in ground_truth:
                            f.write("%s\n" % item)
                    elif path == score_path:
                        for item in pred_scores:
                            f.write("%s\n" % item)
            try:
                out_file_path = os.path.join(self.val_scores_dir, 'outfile_' + str(iterations) + '.pkl')
                utils.scoring(refs=gt_path, sys_outs=score_path, out_file=out_file_path)
                auc = utils.summary(folname=self.val_scores_dir, scores=out_file_path, iterations=iterations)
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

    def save_train_test_hidden_states(self, gen, iterations, Train=True):
        # saved_hidden_states = {}
        saved_scores = {}  # probabilities!!!
        if Train:
            mode = 'train'
        else:
            mode = 'test'
        for batch_number, batch_data in tqdm(enumerate(gen)):
            # try:
            files = batch_data['files']
            self.G = self.G.eval()
            with torch.no_grad():
                predictions = self.forward_pass(batch_data=batch_data, margin_config=False)
                # fus = fus.detach().cpu().numpy()
                scores = softmax(predictions.detach().cpu().numpy(), axis=1)

            if not self.args.PRETRAINING:
                for i, file in enumerate(files):
                    filekey = file['fusion']
                    # if self.args.FUSION_SETUP:
                    #     saved_hidden_states[filekey] = fus[i]
                    score = scores[i, self.class2index['p']]
                    saved_scores[filekey] = score
        # if self.args.FUSION_SETUP:
        #     dump_path_fus = os.path.join(self.val_scores_dir, mode + '_hidden_states_' + str(iterations) + '.pkl')
        #     utils.dump(saved_hidden_states, dump_path_fus)
        """Also save the probabilities!!! Remember the val probabilities are saved as scores_'iterations' in text format"""
        dump_path_scores = os.path.join(self.val_scores_dir, mode + '_scores_' + str(iterations) + '.pkl')
        utils.dump(saved_scores, dump_path_scores)
        stop = None

    def get_save_hidden_state_generators(self, train, test):
        """"""
        if self.args.TRAIN_DATASET == 'DiCOVA' and self.args.LOSS != 'margin':
            # Adjust how often we see positive examples
            train_files_list = train['positive'] + train['negative']
            test_files_list = test
            """Make dataloader"""
            train_data = DiCOVA_Dataset_Fusion_from_Preds(config=self.config, params={'files': train_files_list,
                                                                    'mode': 'val',
                                                                    'metadata_object': self.metadata,
                                                                    'best_modality_models': self.best_modality_models,
                                                                    'hidden_states': self.hidden_states,
                                                                    'specaugment': self.model_hyperparameters.specaug_probability,
                                                                    'time_warp': self.args.TIME_WARP,
                                                                    'input_type': self.args.MODEL_INPUT_TYPE,
                                                                    'args': self.args})
            train_gen = data.DataLoader(train_data, batch_size=self.model_hyperparameters.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=False)
            self.index2class = train_data.index2class
            self.class2index = train_data.class2index
            test_data = DiCOVA_Dataset_Fusion_from_Preds(config=self.config, params={'files': test_files_list,
                                                                  'mode': 'test',
                                                                  'metadata_object': self.metadata,
                                                                  'best_modality_models': self.best_modality_models,
                                                                  'hidden_states': self.hidden_states,
                                                                  'specaugment': 0.0,
                                                                  'time_warp': self.args.TIME_WARP,
                                                                  'input_type': self.args.MODEL_INPUT_TYPE,
                                                                  'args': self.args})
            test_gen = data.DataLoader(test_data, batch_size=self.model_hyperparameters.batch_size,
                                      shuffle=True, collate_fn=test_data.collate, drop_last=False)
            return train_gen, test_gen
        stop = None

    def get_train_val_generators(self, train, val):
        """Return generators for training with different datasets"""
        if self.args.TRAIN_DATASET == 'DiCOVA' and self.args.LOSS != 'margin':
            # Adjust how often we see positive examples
            train_files_list = self.upsample_positive_class(negative=train['negative'], positive=train['positive'])
            val_files_list = val['positive'] + val['negative']
            """Make dataloader"""
            train_data = DiCOVA_Dataset_Fusion_from_Preds(config=self.config, params={'files': train_files_list,
                                                                    'mode': 'train',
                                                                    'metadata_object': self.metadata,
                                                                    'best_modality_models': self.best_modality_models,
                                                                    'hidden_states': self.hidden_states,
                                                                    'specaugment': self.model_hyperparameters.specaug_probability,
                                                                    'time_warp': self.args.TIME_WARP,
                                                                    'input_type': self.args.MODEL_INPUT_TYPE,
                                                                    'args': self.args})
            train_gen = data.DataLoader(train_data, batch_size=self.model_hyperparameters.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=False)
            self.index2class = train_data.index2class
            self.class2index = train_data.class2index
            val_data = DiCOVA_Dataset_Fusion_from_Preds(config=self.config, params={'files': val_files_list,
                                                                  'mode': 'val',
                                                                  'metadata_object': self.metadata,
                                                                  'best_modality_models': self.best_modality_models,
                                                                  'hidden_states': self.hidden_states,
                                                                  'specaugment': 0.0,
                                                                  'time_warp': self.args.TIME_WARP,
                                                                  'input_type': self.args.MODEL_INPUT_TYPE,
                                                                  'args': self.args})
            val_gen = data.DataLoader(val_data, batch_size=self.model_hyperparameters.batch_size,
                                      shuffle=True, collate_fn=val_data.collate, drop_last=False)
            return train_gen, val_gen
        elif self.args.TRAIN_DATASET == 'DiCOVA' and self.args.LOSS == 'margin':
            # Make all pairs of positive and negative samples
            train_files_list = self.all_pairs(negative=train['negative'], positive=train['positive'])
            val_files_list = val['positive'] + val['negative']  # NOT PAIRS!!!
            """Make dataloader"""
            train_data = DiCOVA_Dataset_Margin(config=self.config, params={'files': train_files_list,
                                                                    'mode': 'train',
                                                                    'metadata_object': self.metadata,
                                                                    'specaugment': self.model_hyperparameters.specaug_probability,
                                                                    'time_warp': self.args.TIME_WARP,
                                                                    'input_type': self.args.MODEL_INPUT_TYPE,
                                                                    'args': self.args})
            train_gen = data.DataLoader(train_data, batch_size=self.model_hyperparameters.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=True)
            self.index2class = train_data.index2class
            self.class2index = train_data.class2index
            val_data = DiCOVA_Dataset(config=self.config, params={'files': val_files_list,
                                                                  'mode': 'val',
                                                                  'metadata_object': self.metadata,
                                                                  'specaugment': 0.0,
                                                                  'time_warp': self.args.TIME_WARP,
                                                                  'input_type': self.args.MODEL_INPUT_TYPE,
                                                                  'args': self.args})
            val_gen = data.DataLoader(val_data, batch_size=self.model_hyperparameters.batch_size,
                                      shuffle=True, collate_fn=val_data.collate, drop_last=True)
            return train_gen, val_gen
        elif self.args.TRAIN_DATASET == 'COUGHVID':
            """"""
            train_files_list = train
            val_files_list = val
            """Make dataloader"""
            train_data = COUGHVID_Dataset(config=self.config, params={'files': train_files_list,
                                                                      'mode': 'train',
                                                                      'input_type': self.args.MODEL_INPUT_TYPE,
                                                                      'args': self.args})
            train_gen = data.DataLoader(train_data, batch_size=self.model_hyperparameters.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=True)
            val_data = COUGHVID_Dataset(config=self.config, params={'files': val_files_list,
                                                                    'mode': 'val',
                                                                    'input_type': self.args.MODEL_INPUT_TYPE,
                                                                    'args': self.args})
            val_gen = data.DataLoader(val_data, batch_size=self.model_hyperparameters.batch_size,
                                      shuffle=True, collate_fn=val_data.collate, drop_last=True)
            return train_gen, val_gen
        elif self.args.TRAIN_DATASET == 'LibriSpeech' or self.args.TRAIN_DATASET == 'Cambridge':
            """"""
            train_files_list = train
            val_files_list = val
            """Make dataloader"""
            train_data = LibriSpeech_Dataset(config=self.config, params={'files': train_files_list,
                                                                         'mode': 'train',
                                                                         'input_type': self.args.MODEL_INPUT_TYPE,
                                                                         'args': self.args})
            train_gen = data.DataLoader(train_data, batch_size=self.model_hyperparameters.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=True)
            val_data = LibriSpeech_Dataset(config=self.config, params={'files': val_files_list,
                                                                       'mode': 'val',
                                                                       'input_type': self.args.MODEL_INPUT_TYPE,
                                                                       'args': self.args})
            val_gen = data.DataLoader(val_data, batch_size=self.model_hyperparameters.batch_size,
                                      shuffle=True, collate_fn=val_data.collate, drop_last=True)
            return train_gen, val_gen

    def get_test_data(self):
        filename_converter = utils.get_test_filename_converter()
        """Hard-code the directories here. Sloppy software engineering but it's the end stretch."""
        all_test_files = utils.collect_files('feats/DiCOVA_Test')
        test_dict = {}
        for file in all_test_files:
            filename = file.split('/')[-1]
            fileID = filename.split('_')[0]
            modality = filename.split('_')[1].split('.')[0]
            fusionID = filename_converter[modality][fileID]['fusion']
            if fusionID not in test_dict:
                test_dict[fusionID] = {}
                test_dict[fusionID][modality] = file
            else:
                test_dict[fusionID][modality] = file
        """Need to turn dictionary into list. We also want to keep the fusion name so first add that as a value"""
        for key, value in test_dict.items():
            value['fusion'] = key
        test_list = []
        for key, value in test_dict.items():
            test_list.append(value)

        return test_list

    def text_to_score_dict(self, path, labels=False):
        file1 = open(path, 'r')
        Lines = file1.readlines()
        score_dict = {}
        for line in Lines:
            line = line[:-1]  # remove newline character
            pieces = line.split(' ')
            filekey = pieces[0]
            if not labels:
                score = float(pieces[1])
            else:
                score = pieces[1]
            score_dict[filekey] = score
        return score_dict

    def train(self):
        iterations = 0
        """Get train/test"""
        train, val = self.get_train_test()
        for epoch in range(self.model_hyperparameters.num_epochs):
            train_gen, val_gen = self.get_train_val_generators(train, val)
            for batch_number, batch_data in enumerate(train_gen):
                # try:
                    self.G = self.G.train()
                    if self.args.LOSS == 'margin':
                        predictions = self.forward_pass(batch_data=batch_data, margin_config=True)
                    else:
                        predictions = self.forward_pass(batch_data=batch_data, margin_config=False)
                    loss = self.compute_loss(predictions=predictions, batch_data=batch_data)
                    # Backward and optimize.
                    self.reset_grad()
                    loss.sum().backward()
                    self.g_optimizer.step()

                    if iterations % self.log_step == 0:
                        normalized_loss = loss.sum().item()
                        print(str(iterations) + ', loss: ' + str(normalized_loss))
                        if self.use_tensorboard:
                            self.logger.add_scalar('loss', normalized_loss, iterations)
                    if iterations % self.model_save_step == 0:
                        test = self.get_test_data()
                        save_train_gen, save_test_gen = self.get_save_hidden_state_generators(train, test)
                        self.save_train_test_hidden_states(save_test_gen, iterations=iterations, Train=False)
                        # self.save_train_test_hidden_states(save_train_gen, iterations=iterations, Train=True)
                        """Calculate validation loss"""
                        val_loss, Prec, Rec, acc, auc = self.val_loss(val=val_gen, iterations=iterations)
                        print(str(iterations) + ', val_loss: ' + str(val_loss))
                        if self.use_tensorboard:
                            self.logger.add_scalar('val_loss', val_loss, iterations)
                            self.logger.add_scalar('Prec', Prec, iterations)
                            self.logger.add_scalar('Rec', Rec, iterations)
                            self.logger.add_scalar('Accuracy', acc, iterations)
                            self.logger.add_scalar('AUC', auc, iterations)
                    """Save model checkpoints."""
                    if iterations % self.model_save_step == 0:
                        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(iterations))
                        torch.save({'model': self.G.state_dict(),
                                    'optimizer': self.g_optimizer.state_dict()}, G_path)
                        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                    iterations += 1
                # except:
                #     print('GPU out of memory or other training error...')

    def to_gpu(self, tensor):
        tensor = tensor.to(self.torch_type)
        tensor = tensor.to(self.device)
        return tensor

    def fix_tensor(self, x):
        x.requires_grad = True
        x = x.to(self.torch_type)
        x = x.cuda()
        return x

    def dump_json(self, dict, path):
        a_file = open(path, "w")
        json.dump(dict, a_file, indent=2)
        a_file.close()

def TPR_FPR(gt, pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for key, prediction in pred.items():
        if gt[key] == 'p':
            if prediction == 'p':
                TP += 1
            elif prediction == 'n':
                FN += 1
        elif gt[key] == 'n':
            if prediction == 'n':
                TN += 1
            elif prediction == 'p':
                FP += 1
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    eps = 0.00000000001
    Prec = TP / (TP + FP + eps)
    Rec = TPR
    return TPR, FPR, Prec, Rec


def get_agreement_TPR_FPR(temp_labels, modalities, ground_truth_scores):
    """First get subset of samples where all modalities agree"""
    subset_samples = {}
    for key, label in ground_truth_scores.items():
        label_list = []
        for mod in modalities:
            label_list.append(temp_labels[mod][key])
        label_list = list(set(label_list))
        if len(label_list) == 2:
            """Modalities did not agree, ignore this sample"""
        elif len(label_list) == 1:
            subset_samples[key] = label_list[0]
    fraction_kept = len(subset_samples) / len(ground_truth_scores)
    TPR, FPR , Prec, Rec = TPR_FPR(gt=ground_truth_scores, pred=subset_samples)
    return TPR, FPR, fraction_kept, Prec, Rec

def main(args):
    # dum = utils.load('exps/dummy_fusion_spect_from_preds_fold1_2/val_scores/test_scores_20.pkl')
    # solver = Solver(config=config, args=args)
    fold_probs = {'speech': {}, 'cough': {}, 'breathing': {}}
    ground_truth_scores = {}
    for fold in tqdm(['0', '1', '2', '3', '4']):
        args.FOLD = fold
        solver = Solver(config=config, args=args)
        """Collect the probabilties of validation scores for each fold based on solver.best_modality_models"""
        for modality in ['speech', 'cough', 'breathing']:
            """Get the scores file"""
            exp_path = os.path.join(*solver.best_modality_models[modality].split('/')[:-2])
            model_num = solver.best_modality_models[modality].split('/')[-1].split('-')[0]
            score_path = os.path.join(exp_path, 'val_scores', 'scores_' + model_num)
            assert os.path.exists(score_path)
            gt_path = os.path.join(exp_path, 'val_scores', 'val_labels_' + model_num)
            assert os.path.exists(gt_path)
            scores = solver.text_to_score_dict(score_path)
            gt_scores = solver.text_to_score_dict(gt_path, labels=True)
            for key, score in scores.items():
                key = key.split('_')[0]
                if key not in fold_probs[modality]:
                    fold_probs[modality][key] = score
                else:
                    """Overlap in validation issue, hopefully not present in data... ---> confirmed this doesn't happen"""
                if key not in ground_truth_scores:
                    ground_truth_scores[key] = gt_scores[key + '_' + modality]
                    stop = None

            stop = None
            # fold_probs[modality].append(solver.text_to_score_dict(score_path))
    """Now we want to go through 100 thresholds and see what the TPR and FPR are for the agreement sets"""
    agreement_sets = {'S+B+C': ['speech', 'breathing', 'cough'],
                      'S': ['speech'],
                      'B': ['breathing'],
                      'C': ['cough'],
                      'S+B': ['speech', 'breathing'],
                      'S+C': ['speech', 'cough'],
                      'C+B': ['cough', 'breathing'],
                      }
    agreement_TPR_FPR = {}
    for threshold in np.linspace(start=0, stop=1, num=101):
        agreement_TPR_FPR[threshold] = {'S+B+C': {},
                                        'S': {},
                                        'B': {},
                                        'C': {},
                                        'S+B': {},
                                        'S+C': {},
                                        'C+B': {}
                                        }
        """Convert probabilities to labels at given threshold"""
        temp_labels = {'speech': {}, 'cough': {}, 'breathing': {}}
        for modality in ['speech', 'cough', 'breathing']:
            for key, value in fold_probs[modality].items():
                if value >= threshold:
                    temp_labels[modality][key] = 'p'
                else:
                    temp_labels[modality][key] = 'n'
        for key, modalities in agreement_sets.items():
            TPR, FPR, fraction_kept, Prec, Rec = get_agreement_TPR_FPR(temp_labels, modalities, ground_truth_scores)
            plot_points = [FPR, TPR]
            sensitivity = TPR
            specificity = 1 - FPR
            eps = 0.000000000001
            F1 = 2 * (Prec * Rec) / (Prec + Rec + eps)
            agreement_TPR_FPR[threshold][key] = {'TPR': TPR, 'FPR': FPR, 'fraction_kept': fraction_kept,
                                                 'plot_points': plot_points, 'sens': sensitivity, 'spec': specificity,
                                                 'F1': F1, 'Prec': Prec, 'Rec': Rec}

            stop = None
        stop = None
    """Print numbers for Table 2"""
    T33 = agreement_TPR_FPR[0.33]
    T66 = agreement_TPR_FPR[0.66]
    T25 = agreement_TPR_FPR[0.1]
    T50 = agreement_TPR_FPR[0.2]
    T75 = agreement_TPR_FPR[0.3]
    for modality_agreement_key, _ in T33.items():
        print(modality_agreement_key + ' & ' + str(round(T33[modality_agreement_key]['sens'], 2)) + ' & ' + \
              str(round(T33[modality_agreement_key]['spec'], 2)) + ' & ' + str(round(T33[modality_agreement_key]['fraction_kept'], 2)) + \
              ' & ' + str(round(T66[modality_agreement_key]['sens'], 2)) + ' & ' + \
              str(round(T66[modality_agreement_key]['spec'], 2)) + ' & ' + str(round(T66[modality_agreement_key]['fraction_kept'], 2)) + ' \\\ \\hline'
              )
    print ('*****************************')
    keys = ['S+B+C', 'S+B', 'S+C', 'C+B', 'S', 'B', 'C']
    for modality_agreement_key in keys:
        print(modality_agreement_key + ' & ' + str(round(T25[modality_agreement_key]['F1'], 2)) + ' & ' + \
              str(int(round(T25[modality_agreement_key]['fraction_kept'], 2)*100)) + ' & ' + str(round(T50[modality_agreement_key]['F1'], 2)) + \
              ' & ' + str(int(round(T50[modality_agreement_key]['fraction_kept'], 2) * 100)) + ' & ' + \
              str(round(T75[modality_agreement_key]['F1'], 2)) + ' & ' + str(int(round(T75[modality_agreement_key]['fraction_kept'], 2) * 100)) + ' \\\ \\hline'
              )
    """Let's do a scatter plot of the S+B+C scenario of all TPR and FPR points"""
    for modality_agreement_key, _ in agreement_TPR_FPR[0].items():
        point_set = []
        fractions = []
        for threshold, agreement_dict in agreement_TPR_FPR.items():
            point_set.append(agreement_dict[modality_agreement_key]['plot_points'])
            fractions.append([threshold, agreement_dict[modality_agreement_key]['fraction_kept']])
        point_set = np.asarray(point_set)
        # plt.scatter(point_set[:, 0], point_set[:, 1])
        # plt.show()
        # fractions = np.asarray(fractions)
        # plt.plot(fractions[:, 0], fractions[:, 1])
        # plt.show()
        stop = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to train classifier')
    parser.add_argument('--TRIAL', type=str, default='spect_vs_energy')
    parser.add_argument('--TRAIN', type=utils.str2bool, default=False)
    parser.add_argument('--LOAD_MODEL', type=utils.str2bool, default=False)
    parser.add_argument('--FOLD', type=str, default=None)
    parser.add_argument('--RESTORE_PATH', type=str, default='')
    parser.add_argument('--RESTORE_SPEECH_PRETRAINER_PATH', type=str, default='exps/speech_pretrain_10ff_spect_APC/models/170000-G.ckpt')
    parser.add_argument('--RESTORE_COUGH_PRETRAINER_PATH', type=str, default='exps/cough_pretrain_10ff_spect_APC/models/100000-G.ckpt')
    parser.add_argument('--RESTORE_BREATHING_PRETRAINER_PATH', type=str, default='exps/breathing_pretrain_10ff_spect_APC/models/75000-G.ckpt')
    parser.add_argument('--RESTORE_SPEECH_FINETUNED_EXP_PATH', type=str, default='exps/speech_ablation_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy')
    parser.add_argument('--RESTORE_COUGH_FINETUNED_EXP_PATH', type=str, default='exps/cough_ablation_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy')
    parser.add_argument('--RESTORE_BREATHING_FINETUNED_EXP_PATH', type=str, default='exps/breathing_ablation_noMF_LSTM_yespretrain_notimewarp_yesspecaug_spect_crossentropy')
    parser.add_argument('--PRETRAINING', type=utils.str2bool, default=False)
    parser.add_argument('--FROM_PRETRAINING', type=utils.str2bool, default=True)
    parser.add_argument('--LOSS', type=str, default='crossentropy')  # crossentropy, APC, margin
    parser.add_argument('--MODALITY', type=str, default='fusion')
    parser.add_argument('--FEAT_DIR', type=str, default='feats/DiCOVA')
    parser.add_argument('--POS_NEG_SAMPLING_RATIO', type=float, default=1.0)
    parser.add_argument('--TIME_WARP', type=utils.str2bool, default=False)
    parser.add_argument('--MODEL_INPUT_TYPE', type=str, default='spectrogram')  # spectrogram, energy, mfcc
    parser.add_argument('--MODEL_TYPE', type=str, default='LSTM')  # CNN, LSTM
    parser.add_argument('--TRAIN_DATASET', type=str, default='DiCOVA')  # DiCOVA, COUGHVID, LibriSpeech
    parser.add_argument('--TRAIN_CLIP_FRACTION_SPEECH', type=float, default=0.3)  # randomly shorten clips during training (speech, breathing)
    parser.add_argument('--TRAIN_CLIP_FRACTION_COUGH', type=float, default=0.85)
    parser.add_argument('--TRAIN_CLIP_FRACTION_BREATHING', type=float, default=0.3)
    parser.add_argument('--INCLUDE_MF', type=utils.str2bool, default=False)  # include male/female metadata
    parser.add_argument('--USE_TENSORBOARD', type=utils.str2bool, default=True)  # whether to make tb file
    parser.add_argument('--SAVE_METRIC', type=str, default='AUC')  # change to val_loss for this attempt
    parser.add_argument('--MAXIMIZE', type=utils.str2bool, default=True)
    args = parser.parse_args()
    main(args)