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
        if self.args.PRETRAINING:
            self.model_hyperparameters = self.config.pretraining2
        else:
            if self.args.FROM_PRETRAINING:
                self.model_hyperparameters = self.config.post_pretraining_classifier
            else:
                self.model_hyperparameters = self.config.classifier

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

        # Build the model
        self.build_model()
        if self.args.LOAD_MODEL:
            self.restore_model(self.args.RESTORE_PATH)
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        if self.args.FROM_PRETRAINING:
            """Build the model"""
            pretrain_config = copy.deepcopy(self.config)
            if self.args.MODEL_TYPE == 'LSTM':
                pretrain_config.model.name = 'PreTrainer2'
            elif self.args.MODEL_TYPE == 'CNN':
                pretrain_config.model.name = 'PreTrainerCNN'
            """Load the weights"""
            self.pretrained = model.Model(pretrain_config, self.args)
            pretrain_checkpoint = self._load(self.args.RESTORE_PRETRAINER_PATH)
            print('Restoring pretrainer from ' + self.args.RESTORE_PRETRAINER_PATH)
            self.pretrained.load_state_dict(pretrain_checkpoint['model'])
            """Freeze pretrainer"""
            for param in self.pretrained.parameters():
                param.requires_grad = False
            self.pretrained.to(self.device)
            """Make trainer have input to take pretrained feature output dimension"""
            train_config = copy.deepcopy(self.config)
            if self.args.MODEL_TYPE == 'LSTM':
                train_config.model.name = 'PostPreTrainClassifier'
            elif self.args.MODEL_TYPE == 'CNN':
                train_config.model.name = 'PostPreTrainClassifierCNN'
        elif self.args.PRETRAINING:
            train_config = copy.deepcopy(self.config)
            if self.args.MODEL_TYPE == 'LSTM':
                train_config.model.name = 'PreTrainer2'
            elif self.args.MODEL_TYPE == 'CNN':
                train_config.model.name = 'PreTrainerCNN'
        else:
            train_config = copy.deepcopy(self.config)
            train_config.model.name = 'Classifier'
        self.G = model.Model(train_config, self.args)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr)
        self.print_network(self.G, 'G')
        self.G.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def print_optimizer(self, opt, name):
        print(opt)
        print(name)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = SummaryWriter(log_dir=self.log_dir)

    def _load(self, checkpoint_path):
        if self.use_cuda:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def restore_model(self, G_path):
        """Restore the model"""
        print('Loading the trained model from ' + G_path)
        g_checkpoint = self._load(G_path)
        self.G.load_state_dict(g_checkpoint['model'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']

    def update_lr(self, g_lr):
        """Decay learning rates of g"""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def get_absolute_filepaths(self, files):
        new_files = []
        for file in files:
            new_name = os.path.join(self.args.FEAT_DIR, file + '_' + self.args.MODALITY + '.pkl')
            assert os.path.exists(new_name)
            new_files.append(new_name)
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

    def get_train_test(self):
        if self.args.TRAIN_DATASET == 'DiCOVA':
            partition = self.partition.dicova_partition
            partition = partition[self.fold]

            # Get absolute filepaths depending on modality
            partition['train_pos'] = self.get_absolute_filepaths(partition['train_pos'])
            partition['train_neg'] = self.get_absolute_filepaths(partition['train_neg'])
            partition['val_pos'] = self.get_absolute_filepaths(partition['val_pos'])
            partition['val_neg'] = self.get_absolute_filepaths(partition['val_neg'])

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

    def forward_pass(self, batch_data, margin_config=False):
        if not margin_config:
            spects = batch_data['spects']
            files = batch_data['files']
            # labels = batch_data['labels']
            # scalers = batch_data['scalers']
            if self.args.FROM_PRETRAINING:
                _, intermediate = self.pretrained(spects)
            else:
                intermediate = spects
            if self.args.INCLUDE_MF:
                mf = batch_data['mf']
                intermediate = {'intermediate': intermediate, 'mf': mf}
            predictions = self.G(intermediate)
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

    def get_train_val_generators(self, train, val):
        """Return generators for training with different datasets"""
        if self.args.TRAIN_DATASET == 'DiCOVA' and self.args.LOSS != 'margin':
            # Adjust how often we see positive examples
            train_files_list = self.upsample_positive_class(negative=train['negative'], positive=train['positive'])
            val_files_list = val['positive'] + val['negative']
            """Make dataloader"""
            train_data = DiCOVA_Dataset(config=self.config, params={'files': train_files_list,
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
        elif self.args.TRAIN_DATASET == 'LibriSpeech':
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

    def val_scores(self):
        self.evaluation_dir = os.path.join(self.exp_dir, 'evaluations')
        if not os.path.exists(self.evaluation_dir):
            os.mkdir(self.evaluation_dir)
        self.fold_dir = os.path.join(self.evaluation_dir, self.test_fold)
        if not os.path.isdir(self.fold_dir):
            os.mkdir(self.fold_dir)
        self.eval_type_dir = os.path.join(self.fold_dir, 'val')
        if not os.path.isdir(self.eval_type_dir):
            os.mkdir(self.eval_type_dir)
        """Get train/test"""
        train, val = self.get_train_test()
        train_files_list = train['positive'] + train['negative']
        val_files_list = val['positive'] + val['negative']

        ground_truth = []
        pred_scores = []

        """Make dataloader"""
        train_data = DiCOVA_Dataset(config=config, params={'files': train_files_list,
                                                    'mode': 'train',
                                                    'data_object': self.training_data,
                                                    'specaugment': self.config.train.specaugment})
        train_gen = data.DataLoader(train_data, batch_size=config.train.batch_size,
                                    shuffle=True, collate_fn=train_data.collate, drop_last=True)
        self.index2class = train_data.index2class
        self.class2index = train_data.class2index
        val_data = DiCOVA_Dataset(config=config, params={'files': val_files_list,
                                                  'mode': 'train',
                                                  'data_object': self.training_data,
                                                  'specaugment': False})
        val_gen = data.DataLoader(val_data, batch_size=1, shuffle=True, collate_fn=val_data.collate, drop_last=False)
        for batch_number, features in tqdm(enumerate(val_gen)):
            feature = features['features']
            files = features['files']
            self.G = self.G.eval()
            _, intermediate = self.pretrained(feature)
            predictions = self.G(intermediate)
            predictions = predictions.detach().cpu().numpy()
            scores = softmax(predictions, axis=1)
            info = [self.training_data.get_file_metadata(x) for x in files]
            file = files[0]  # batch size 1

            filekey = file.split('/')[-1][:-4]
            gt = info[0]['Covid_status']
            score = scores[0, self.class2index['p']]
            ground_truth.append(filekey + ' ' + gt)
            pred_scores.append(filekey + ' ' + str(score))
        """Sort the lists in alphabetical order"""
        ground_truth.sort()
        pred_scores.sort()

        """Write the files"""
        gt_path = os.path.join(self.eval_type_dir, 'val_labels')
        score_path = os.path.join(self.eval_type_dir, 'scores')
        for path in [gt_path, score_path]:
            with open(path, 'w') as f:
                if path == gt_path:
                    for item in ground_truth:
                        f.write("%s\n" % item)
                elif path == score_path:
                    for item in pred_scores:
                        f.write("%s\n" % item)
        out_file_path = os.path.join(self.eval_type_dir, 'outfile.pkl')
        utils.scoring(refs=gt_path, sys_outs=score_path, out_file=out_file_path)

    def eval(self):
        self.evaluation_dir = os.path.join(self.exp_dir, 'evaluations')
        if not os.path.exists(self.evaluation_dir):
            os.mkdir(self.evaluation_dir)
        self.fold_dir = os.path.join(self.evaluation_dir, self.test_fold)
        if not os.path.isdir(self.fold_dir):
            os.mkdir(self.fold_dir)
        self.eval_type_dir = os.path.join(self.fold_dir, 'blind')
        if not os.path.isdir(self.eval_type_dir):
            os.mkdir(self.eval_type_dir)
        """Get test files"""
        test_files = utils.collect_files(self.config.directories.dicova_test_logspect_feats)
        """Make dataloader"""
        test_data = Dataset(config=config, params={'files': test_files,
                                                    'mode': 'test',
                                                    'data_object': None,
                                                    'specaugment': 0.0})
        test_gen = data.DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=test_data.collate, drop_last=False)
        self.index2class = test_data.index2class
        self.class2index = test_data.class2index
        pred_scores = []
        for batch_number, features in tqdm(enumerate(test_gen)):
            feature = features['features']
            files = features['files']
            self.G = self.G.eval()
            _, intermediate = self.pretrained(feature)
            predictions = self.G(intermediate)

            file = files[0]  # batch size is 1 for evaluation
            filekey = file.split('/')[-1][:-4]
            predictions = predictions.detach().cpu().numpy()
            scores = softmax(predictions, axis=1)
            score = scores[0, self.class2index['p']]
            pred_scores.append(filekey + ' ' + str(score))
        pred_scores.sort()
        score_path = os.path.join(self.eval_type_dir, 'scores')
        with open(score_path, 'w') as f:
            for item in pred_scores:
                f.write("%s\n" % item)

    def val_scores_ensemble(self, model_num):
        """Something is very, very wrong here and I don't know what it is..."""
        self.evaluation_dir = os.path.join(self.exp_dir, 'evaluations')
        if not os.path.exists(self.evaluation_dir):
            os.mkdir(self.evaluation_dir)
        self.fold_dir = os.path.join(self.evaluation_dir, self.args.FOLD)
        if not os.path.isdir(self.fold_dir):
            os.mkdir(self.fold_dir)
        self.eval_type_dir = os.path.join(self.fold_dir, 'val')
        if not os.path.isdir(self.eval_type_dir):
            os.mkdir(self.eval_type_dir)
        self.specific_model_dir = os.path.join(self.eval_type_dir, model_num)
        if not os.path.isdir(self.specific_model_dir):
            os.mkdir(self.specific_model_dir)
        """Get train/test"""
        train, val = self.get_train_test()
        # train_files_list = train['positive'] + train['negative']
        val_files_list = val['positive'] + val['negative']

        ground_truth = []
        pred_scores = []

        """Make dataloader"""
        val_data = DiCOVA_Dataset(config=self.config, params={'files': val_files_list,
                                                              'mode': 'train',
                                                              'metadata_object': self.metadata,
                                                              'specaugment': 1.0,
                                                              'time_warp': False,
                                                              'input_type': self.args.MODEL_INPUT_TYPE,
                                                              'args': self.args})
        val_gen = data.DataLoader(val_data, batch_size=3,
                                  shuffle=True, collate_fn=val_data.collate, drop_last=True)

        self.index2class = val_data.index2class
        self.class2index = val_data.class2index
        correct = 0
        val_loss = 0
        for batch_number, features in tqdm(enumerate(val_gen)):
            files = features['files']
            self.G = self.G.eval()
            predictions = self.forward_pass(batch_data=features, margin_config=False)
            loss = self.compute_loss(predictions=predictions, batch_data=features, crossentropy_overwrite=True)
            val_loss += loss.sum().item()

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

            if gt == 'n' and score < 0.5:
                correct += 1
            if gt == 'p' and score >= 0.5:
                correct += 1
        """Sort the lists in alphabetical order"""
        ground_truth.sort()
        pred_scores.sort()
        print('Dummy accuracy: ' + str(correct/len(ground_truth)))

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
        paths = {'gt_path': gt_path, 'score_path': score_path}

        out_file_path = os.path.join(self.eval_type_dir, 'outfile.pkl')
        utils.scoring(refs=gt_path, sys_outs=score_path, out_file=out_file_path)
        auc = utils.summary(folname=self.eval_type_dir, scores=out_file_path, iterations=0)



        return paths, os.path.join(self.eval_type_dir, 'scores'), self.eval_type_dir
        # out_file_path = os.path.join(self.specific_model_dir, 'outfile.pkl')
        # utils.scoring(refs=gt_path, sys_outs=score_path, out_file=out_file_path)

    def eval_ensemble(self, model_num):
        self.evaluation_dir = os.path.join(self.exp_dir, 'evaluations')
        if not os.path.exists(self.evaluation_dir):
            os.mkdir(self.evaluation_dir)
        self.fold_dir = os.path.join(self.evaluation_dir, self.test_fold)
        if not os.path.isdir(self.fold_dir):
            os.mkdir(self.fold_dir)
        self.eval_type_dir = os.path.join(self.fold_dir, 'blind')
        if not os.path.isdir(self.eval_type_dir):
            os.mkdir(self.eval_type_dir)
        self.specific_model_dir = os.path.join(self.eval_type_dir, model_num)
        if not os.path.isdir(self.specific_model_dir):
            os.mkdir(self.specific_model_dir)
        """Get test files"""
        test_files = utils.collect_files(self.config.directories.dicova_test_logspect_feats)
        """Make dataloader"""
        test_data = Dataset(config=config, params={'files': test_files,
                                                    'mode': 'test',
                                                    'data_object': None,
                                                    'specaugment': 0.0})
        test_gen = data.DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=test_data.collate, drop_last=False)
        self.index2class = test_data.index2class
        self.class2index = test_data.class2index
        pred_scores = []
        for batch_number, features in tqdm(enumerate(test_gen)):
            feature = features['features']
            files = features['files']
            self.G = self.G.eval()
            _, intermediate = self.pretrained(feature)
            predictions = self.G(intermediate)

            file = files[0]  # batch size is 1 for evaluation
            filekey = file.split('/')[-1][:-4]
            predictions = predictions.detach().cpu().numpy()
            scores = softmax(predictions, axis=1)
            score = scores[0, self.class2index['p']]
            pred_scores.append(filekey + ' ' + str(score))
        pred_scores.sort()
        score_path = os.path.join(self.specific_model_dir, 'scores')
        with open(score_path, 'w') as f:
            for item in pred_scores:
                f.write("%s\n" % item)
        return score_path, self.eval_type_dir

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

def main(args):
    solver = Solver(config=config, args=args)
    if args.TRAIN:
        solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to train classifier')
    parser.add_argument('--TRIAL', type=str, default='dummy_cough_fold1')
    parser.add_argument('--TRAIN', type=utils.str2bool, default=True)
    parser.add_argument('--LOAD_MODEL', type=utils.str2bool, default=False)
    parser.add_argument('--FOLD', type=str, default='1')
    parser.add_argument('--RESTORE_PATH', type=str, default='')
    parser.add_argument('--RESTORE_PRETRAINER_PATH', type=str, default='exps/cough_pretrain_10ff_spect_APC/models/100000-G.ckpt')
    parser.add_argument('--PRETRAINING', type=utils.str2bool, default=False)
    parser.add_argument('--FROM_PRETRAINING', type=utils.str2bool, default=True)
    parser.add_argument('--LOSS', type=str, default='crossentropy')  # crossentropy, APC, margin
    parser.add_argument('--MODALITY', type=str, default='cough')
    parser.add_argument('--FEAT_DIR', type=str, default='feats/DiCOVA')
    parser.add_argument('--POS_NEG_SAMPLING_RATIO', type=float, default=1.0)
    parser.add_argument('--TIME_WARP', type=utils.str2bool, default=False)
    parser.add_argument('--MODEL_INPUT_TYPE', type=str, default='spectrogram')  # spectrogram, energy, mfcc
    parser.add_argument('--MODEL_TYPE', type=str, default='LSTM')  # CNN, LSTM
    parser.add_argument('--TRAIN_DATASET', type=str, default='DiCOVA')  # DiCOVA, COUGHVID, LibriSpeech
    parser.add_argument('--TRAIN_CLIP_FRACTION', type=float, default=0.85)  # randomly shorten clips during training (speech, breathing)
    parser.add_argument('--INCLUDE_MF', type=utils.str2bool, default=True)  # include male/female metadata
    parser.add_argument('--USE_TENSORBOARD', type=utils.str2bool, default=True)  # whether to make tb file
    args = parser.parse_args()
    main(args)