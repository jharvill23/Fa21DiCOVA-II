import torch.nn as nn
import torch.nn.functional as F
#import torch.autograd.function as Function
#from torch.autograd import function as Function
import torch
import utils
import torchvision.models as models
from torch.nn.utils.rnn import pad_sequence
import numpy as np


config = utils.get_config()

class Model(nn.Module):
    def __init__(self, config, args, fusion=False):
        super(Model, self).__init__()
        self.config = config
        self.args = args
        self.fusion = fusion
        model = eval(self.config.model.name)
        self.model = model(self.config, self.args, self.fusion)

    def forward(self, padded_input):
        return self.model(padded_input)


class PreTrainer2(nn.Module):
    def __init__(self, config, args, fusion=False):
        super(PreTrainer2, self).__init__()
        self.config = config
        self.args = args
        if self.args.MODEL_INPUT_TYPE == 'spectrogram':
            self.input_size = self.config.data.num_mels
        elif self.args.MODEL_INPUT_TYPE == 'energy':
            self.input_size = 1
        if self.args.MODEL_INPUT_TYPE == 'mfcc':
            self.input_size = self.config.data.num_mfccs
        self.num_mels = self.config.data.num_mels
        self.hidden_size = self.config.pretraining2.hidden_size
        self.linear_hidden_size = self.config.pretraining2.linear_hidden_size
        self.encoder1_num_layers = self.config.pretraining2.encoder1_num_layers
        self.encoder2_num_layers = self.config.pretraining2.encoder2_num_layers
        self.batch_first = self.config.pretraining2.batch_first
        self.dropout = self.config.pretraining2.dropout
        self.encoder_lstm_1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                             num_layers=self.encoder1_num_layers, batch_first=self.batch_first,
                             dropout=self.dropout, bidirectional=False)
        self.encoder_lstm_2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                      num_layers=self.encoder2_num_layers, batch_first=self.batch_first,
                                      dropout=self.dropout, bidirectional=False)
        self.full1 = nn.Linear(in_features=self.hidden_size,
                               out_features=self.linear_hidden_size)
        self.full2 = nn.Linear(in_features=self.linear_hidden_size, out_features=self.input_size)

    def forward(self, x):
        x, _ = self.encoder_lstm_1(x)
        intermediate_state = x  # take the intermediate layer after two layers of LSTM
        x, _ = self.encoder_lstm_2(x)
        x = self.full1(x)
        x = torch.tanh(x)
        x = self.full2(x)
        return x, intermediate_state

class PostPreTrainClassifier(nn.Module):
    def __init__(self, config, args, fusion=False):
        super(PostPreTrainClassifier, self).__init__()
        self.config = config
        self.args = args
        self.fusion = fusion  # whether we use this model for fusion training, changes output behavior!!!
        self.input_size = self.config.pretraining2.hidden_size
        self.hidden_size = self.config.post_pretraining_classifier.hidden_size
        self.linear_hidden_size = self.config.post_pretraining_classifier.linear_hidden_size
        self.encoder_num_layers = self.config.post_pretraining_classifier.encoder_num_layers
        self.batch_first = self.config.post_pretraining_classifier.batch_first
        self.dropout = self.config.post_pretraining_classifier.dropout
        self.output_dim = 2  # number of classes
        self.bidirectional = config.post_pretraining_classifier.bidirectional
        if self.args.INCLUDE_MF:
            """Add an embedding layer for male/female input embeddings"""
            self.embed = nn.Embedding(num_embeddings=2, embedding_dim=10)  # male/female classes only
            """Add the dimension of the embedding to the input size!!!"""
            self.input_size += self.embed.embedding_dim

        self.encoder_lstm_1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                      num_layers=self.encoder_num_layers, batch_first=self.batch_first,
                                      dropout=self.dropout, bidirectional=self.bidirectional)

        self.full1 = nn.Linear(in_features=self.hidden_size*2 if self.bidirectional else self.hidden_size,
                               out_features=self.linear_hidden_size)
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.full2 = nn.Linear(in_features=self.linear_hidden_size, out_features=self.linear_hidden_size)
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.full3 = nn.Linear(in_features=self.linear_hidden_size, out_features=self.output_dim)

    def forward(self, x):
        if self.args.INCLUDE_MF:
            data = x['intermediate']
            mf_indices = x['mf']
            """Need to get mf embedding and concatenate along intermediate representation's time axis"""
            mf_embeds = self.embed(mf_indices)
            concattable_embeds = torch.unsqueeze(mf_embeds, dim=1)
            concattable_embeds = concattable_embeds.repeat(1, data.shape[1], 1)
            x = torch.cat((data, concattable_embeds), dim=2)

        x, _ = self.encoder_lstm_1(x)
        out_forward = x[:, :, :self.hidden_size]
        out_backward = x[:, :, self.hidden_size:]
        x_forward = out_forward[:, -1]
        x_backward = out_backward[:, 0]
        summary = torch.cat((x_forward, x_backward), dim=1)
        x = self.full1(summary)
        x = self.dropout1(x)
        x = F.tanh(x)
        x = self.full2(x)
        x = self.dropout2(x)
        x = F.tanh(x)
        if self.fusion:
            return x
        x = self.full3(x)
        return x

class Classifier(nn.Module):
    def __init__(self, config, args, fusion=False):
        super(Classifier, self).__init__()
        self.config = config
        self.args = args
        if self.args.MODEL_INPUT_TYPE == 'spectrogram':
            self.input_size = self.config.data.num_mels
        elif self.args.MODEL_INPUT_TYPE == 'energy':
            self.input_size = 1
        if self.args.MODEL_INPUT_TYPE == 'mfcc':
            self.input_size = self.config.data.num_mfccs
        self.hidden_size = self.config.post_pretraining_classifier.hidden_size
        self.linear_hidden_size = self.config.post_pretraining_classifier.linear_hidden_size
        self.encoder_num_layers = self.config.post_pretraining_classifier.encoder_num_layers
        self.batch_first = self.config.post_pretraining_classifier.batch_first
        self.dropout = self.config.post_pretraining_classifier.dropout
        self.output_dim = 2  # number of classes
        self.bidirectional = config.post_pretraining_classifier.bidirectional

        self.encoder_lstm_1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                      num_layers=self.encoder_num_layers, batch_first=self.batch_first,
                                      dropout=self.dropout, bidirectional=self.bidirectional)

        self.full1 = nn.Linear(in_features=self.hidden_size*2 if self.bidirectional else self.hidden_size,
                               out_features=self.linear_hidden_size)
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.full2 = nn.Linear(in_features=self.linear_hidden_size, out_features=self.linear_hidden_size)
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.full3 = nn.Linear(in_features=self.linear_hidden_size, out_features=self.output_dim)

    def forward(self, x):
        x, _ = self.encoder_lstm_1(x)
        out_forward = x[:, :, :self.hidden_size]
        out_backward = x[:, :, self.hidden_size:]
        x_forward = out_forward[:, -1]
        x_backward = out_backward[:, 0]
        summary = torch.cat((x_forward, x_backward), dim=1)
        x = self.full1(summary)
        x = self.dropout1(x)
        x = F.tanh(x)
        x = self.full2(x)
        x = self.dropout2(x)
        x = F.tanh(x)
        x = self.full3(x)
        return x


class ClassifierCNN(nn.Module):  # ResNet50 +  Average Pool
    def __init__(self, config, args, fusion=False):
        super(ClassifierCNN, self).__init__()
        self.config = config
        self.args = args

        self.model = models.resnet50(pretrained=True)
        self.model.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        x = self.model(x)
        return x


class PreTrainerCNN(nn.Module):
    def __init__(self, config, args, fusion=False):
        super(PreTrainerCNN, self).__init__()
        # https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113
        self.config = config
        self.args = args
        if self.args.MODEL_INPUT_TYPE == 'spectrogram':
            self.input_size = self.config.data.num_mels
        elif self.args.MODEL_INPUT_TYPE == 'energy':
            self.input_size = 1
        if self.args.MODEL_INPUT_TYPE == 'mfcc':
            self.input_size = self.config.data.num_mfccs
        ### https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707
        # original_model = models.resnet18(pretrained=True)
        # self.high_feats = nn.Sequential(*list(original_model.children())[:-3])
        # self.conv1 = ConvNorm(in_channels=self.input_size, out_channels=self.input_size)

        self.upscale = self.config.pretraining2.upscale

        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(self.input_size,
                         self.input_size * self.upscale,
                         kernel_size=7, stride=1,
                         padding=int((7 - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(self.input_size * self.upscale))
        )
        """Lower layers"""
        for i in range(0, 3):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(self.input_size * self.upscale,
                             self.input_size * self.upscale,
                             kernel_size=7, stride=1,
                             padding=int((7 - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(self.input_size * self.upscale))
            )
        """Higher layers"""
        for i in range(0, 3):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(self.input_size * self.upscale,
                             self.input_size * self.upscale,
                             kernel_size=7, stride=1,
                             padding=int((7 - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(self.input_size * self.upscale))
            )
        """Last layer"""
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(self.input_size * self.upscale,
                         self.input_size,
                         kernel_size=7, stride=1,
                         padding=int((7 - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(self.input_size))
        )
        self.n_convs = len(self.convolutions)




    def forward(self, x):
        """Expand to three 'channels' for the resnet model"""
        # examples = []
        # for example in x:
        #     example = torch.unsqueeze(example, dim=0)
        #     example = example.repeat(3, 1, 1)
        #     examples.append(example)
        # examples = pad_sequence(examples, batch_first=True, padding_value=0)
        # x = examples
        x = x.permute(0, 2, 1)
        # if self.args.TRAIN:
        i = 0
        for conv in self.convolutions:
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
            if i == 3:
                intermediate = x
            i += 1
            # x = self.high_feats(x)
        x = x.permute(0, 2, 1)
        intermediate = intermediate.permute(0, 2, 1)
        return x, intermediate


class PostPreTrainClassifierCNN(nn.Module):
    def __init__(self, config, args, fusion=False):
        super(PostPreTrainClassifierCNN, self).__init__()
        # https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113
        self.config = config
        self.args = args
        if self.args.MODEL_INPUT_TYPE == 'spectrogram':
            self.input_size = self.config.data.num_mels
        elif self.args.MODEL_INPUT_TYPE == 'energy':
            self.input_size = 1
        if self.args.MODEL_INPUT_TYPE == 'mfcc':
            self.input_size = self.config.data.num_mfccs
        if self.args.INCLUDE_MF:
            """Add an embedding layer for male/female input embeddings"""
            self.embed = nn.Embedding(num_embeddings=2, embedding_dim=10)  # male/female classes only

        self.linear_hidden_size = self.config.post_pretraining_classifier.linear_hidden_size
        self.output_dim = 2  # number of classes
        self.dropout = self.config.post_pretraining_classifier.dropout
        ### https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707
        # original_model = models.resnet18(pretrained=True)
        # self.high_feats = nn.Sequential(*list(original_model.children())[:-3])
        # self.conv1 = ConvNorm(in_channels=self.input_size, out_channels=self.input_size)

        self.upscale = self.config.pretraining2.upscale

        self.convolutions = nn.ModuleList()

        """Lower layers"""
        if self.args.INCLUDE_MF:
            conv_input_size = self.input_size * self.upscale + self.embed.embedding_dim
        else:
            conv_input_size = self.input_size * self.upscale
        for i in range(0, 8):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(conv_input_size,
                             conv_input_size,
                             kernel_size=7, stride=1,
                             padding=int((7 - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(conv_input_size))
            )
        """Last layer"""
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(conv_input_size,
                         self.input_size,
                         kernel_size=7, stride=1,
                         padding=int((7 - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(self.input_size))
        )
        self.n_convs = len(self.convolutions)
        """Add an adaptive pooling layer"""
        self.pool = nn.AdaptiveAvgPool2d(self.config.post_pretraining_classifier.pooled_size)
        self.full1 = nn.Linear(in_features=self.config.post_pretraining_classifier.pooled_size**2,
                               out_features=self.linear_hidden_size)
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.full2 = nn.Linear(in_features=self.linear_hidden_size, out_features=self.linear_hidden_size)
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.full3 = nn.Linear(in_features=self.linear_hidden_size, out_features=self.output_dim)

    def forward(self, x):
        if self.args.INCLUDE_MF:
            data = x['intermediate']
            mf_indices = x['mf']
            """Need to get mf embedding and concatenate along intermediate representation's time axis"""
            mf_embeds = self.embed(mf_indices)
            concattable_embeds = torch.unsqueeze(mf_embeds, dim=1)
            concattable_embeds = concattable_embeds.repeat(1, data.shape[1], 1)
            x = torch.cat((data, concattable_embeds), dim=2)

        x = x.permute(0, 2, 1)
        if self.args.TRAIN:
            i = 0
            for conv in self.convolutions:
                if i < self.n_convs - 1:
                    x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
                else:
                    x = F.dropout(conv(x), 0.5, training=self.training)
                i += 1
            # x = self.high_feats(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        """Flatten along last two dimensions"""
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.full1(x)
        x = self.dropout1(x)
        x = F.tanh(x)
        x = self.full2(x)
        x = self.dropout2(x)
        x = F.tanh(x)
        x = self.full3(x)
        return x


class FusionClassifier(nn.Module):
    def __init__(self, config, args, fusion=False):
        super(FusionClassifier, self).__init__()
        self.config = config
        self.args = args
        self.input_size = 3 * self.config.post_pretraining_classifier.linear_hidden_size

        self.linear_hidden_size = self.config.fusion.linear_hidden_size
        self.batch_first = self.config.fusion.batch_first
        self.dropout = self.config.fusion.dropout
        self.output_dim = 2  # number of classes

        self.full1 = nn.Linear(in_features=self.input_size, out_features=self.linear_hidden_size)
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.full2 = nn.Linear(in_features=self.linear_hidden_size, out_features=self.linear_hidden_size)
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.full3 = nn.Linear(in_features=self.linear_hidden_size, out_features=self.output_dim)

    def forward(self, x):
        """x is a fixed-length concatenation of the three modality features"""
        x = self.full1(x)
        x = self.dropout1(x)
        x = F.tanh(x)
        x = self.full2(x)
        x = self.dropout2(x)
        x = F.tanh(x)
        x = self.full3(x)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, config, args, fusion=False):
        super(TransformerClassifier, self).__init__()
        self.config = config
        self.args = args
        if self.args.MODEL_INPUT_TYPE == 'spectrogram':
            self.input_size = self.config.data.num_mels
        elif self.args.MODEL_INPUT_TYPE == 'energy':
            self.input_size = 1
        if self.args.MODEL_INPUT_TYPE == 'mfcc':
            self.input_size = self.config.data.num_mfccs
        if self.args.MODEL_INPUT_TYPE == 'mfcc':
            self.input_size = 512  # Wav2Vec2 output size
        self.reduction_size = self.config.transformer.reduction_size  # reduce each Wav2Vec2 frame to this size to help avoid overfitting!!!
        self.model_dim = self.config.transformer.model_dim
        self.num_heads = self.config.transformer.num_heads
        self.num_layers = self.config.transformer.num_layers
        self.encoder_num_layers = self.config.transformer.num_layers_lstm
        self.hidden_size = self.config.transformer.hidden_size
        self.dim_feedforward = self.config.transformer.dim_feedforward
        self.bidirectional = self.config.transformer.bidirectional

        self.linear_hidden_size = self.config.transformer.linear_hidden_size
        self.batch_first = self.config.transformer.batch_first
        self.dropout = self.config.transformer.dropout
        self.output_dim = 2  # number of classes

        self.reduce_input = nn.Linear(in_features=self.input_size, out_features=self.reduction_size)
        self.inflate_input = nn.Linear(in_features=self.reduction_size, out_features=self.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.num_heads, dim_feedforward=self.dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.encoder_lstm_1 = nn.LSTM(input_size=self.model_dim, hidden_size=self.hidden_size,
                                      num_layers=self.encoder_num_layers, batch_first=self.batch_first,
                                      dropout=self.dropout, bidirectional=self.bidirectional)

        """Add an adaptive pooling layer"""
        # self.pool = nn.AdaptiveAvgPool2d(self.config.transformer.pooled_size)
        # self.full1 = nn.Linear(in_features=self.config.transformer.pooled_size ** 2,
        #                        out_features=self.linear_hidden_size)
        self.full1 = nn.Linear(in_features=self.config.transformer.hidden_size * 2,
                               out_features=self.linear_hidden_size)
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.full2 = nn.Linear(in_features=self.linear_hidden_size, out_features=self.linear_hidden_size)
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.full3 = nn.Linear(in_features=self.linear_hidden_size, out_features=self.output_dim)

    def forward(self, x):
        """Try with no positional encodings so that input acts like a set and not an ordered structure,
           all in the name of avoiding overfitting"""
        """Now we need to add positional encodings because it wasn't learning anything"""
        x = x.permute(1, 0, 2)
        x = add_positional_encodings(x)
        x = x.permute(1, 0, 2)
        x = self.reduce_input(x)
        x = self.inflate_input(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)

        x, _ = self.encoder_lstm_1(x)
        out_forward = x[:, :, :self.hidden_size]
        out_backward = x[:, :, self.hidden_size:]
        x_forward = out_forward[:, -1]
        x_backward = out_backward[:, 0]
        x = torch.cat((x_forward, x_backward), dim=1)


        # x = self.pool(x)
        # """Flatten along last two dimensions"""
        # x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.full1(x)
        x = self.dropout1(x)
        x = F.tanh(x)
        x = self.full2(x)
        x = self.dropout2(x)
        x = F.tanh(x)
        x = self.full3(x)
        return x

def positional_encodings(nframes, d_model):
    times = np.arange(nframes, dtype=np.float)
    frequencies = np.power(10000.0, -2 * np.arange(d_model / 2) / d_model)
    phases = torch.tensor(data=np.outer(times, frequencies), dtype=torch.float)
    return (torch.cat((torch.sin(phases), torch.cos(phases)), dim=1))

def add_positional_encodings(batched_data):
    """Add positional encodings to spectrograms in forward pass"""
    device = batched_data.device
    T = batched_data.shape[0]
    E = batched_data.shape[2]
    N = batched_data.shape[1]
    pos_enc = positional_encodings(nframes=T, d_model=E)
    pos_enc = torch.unsqueeze(pos_enc, dim=1)
    pos_enc = pos_enc.expand(T, N, E).to(device)
    # diff = pos_enc[:, 0, :] - pos_enc[:, 1, :]
    # assert diff == 0
    batched_data = batched_data + 1e-6 * pos_enc
    return batched_data

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)