import torch.nn as nn
import torch.nn.functional as F
#import torch.autograd.function as Function
#from torch.autograd import function as Function
import torch
import utils

config = utils.get_config()

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        model = eval(self.config.model.name)
        self.model = model(config)

    def forward(self, padded_input):
        return self.model(padded_input)


class PreTrainer2(nn.Module):
    def __init__(self, config):
        super(PreTrainer2, self).__init__()
        self.config = config
        self.num_mels = self.config.data.num_mels
        self.hidden_size = self.config.pretraining2.hidden_size
        self.linear_hidden_size = self.config.pretraining2.linear_hidden_size
        self.encoder1_num_layers = self.config.pretraining2.encoder1_num_layers
        self.encoder2_num_layers = self.config.pretraining2.encoder2_num_layers
        self.batch_first = self.config.pretraining2.batch_first
        self.dropout = self.config.pretraining.dropout
        self.encoder_lstm_1 = nn.LSTM(input_size=self.num_mels, hidden_size=self.hidden_size,
                             num_layers=self.encoder1_num_layers, batch_first=self.batch_first,
                             dropout=self.dropout, bidirectional=False)
        self.encoder_lstm_2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                      num_layers=self.encoder2_num_layers, batch_first=self.batch_first,
                                      dropout=self.dropout, bidirectional=False)
        self.full1 = nn.Linear(in_features=self.hidden_size,
                               out_features=self.linear_hidden_size)
        self.full2 = nn.Linear(in_features=self.linear_hidden_size, out_features=self.num_mels)

    def forward(self, x):
        x, _ = self.encoder_lstm_1(x)
        intermediate_state = x  # take the intermediate layer after two layers of LSTM
        x, _ = self.encoder_lstm_2(x)
        x = self.full1(x)
        x = torch.tanh(x)
        x = self.full2(x)
        return x, intermediate_state

class PostPreTrainClassifier(nn.Module):
    def __init__(self, config):
        super(PostPreTrainClassifier, self).__init__()
        self.config = config
        self.input_size = self.config.pretraining2.hidden_size
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
