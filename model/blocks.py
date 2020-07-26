import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, n_channels, hidden_size, encoder_dropout, gpu, bidirectional_encoder = True ):
        super(EncoderRNN, self).__init__()

        self.input_size = n_channels
        self.hidden_size = hidden_size
        self.layers =  1
        self.dropout = encoder_dropout
        self.bi = bidirectional_encoder

        # self.rnn = nn.GRU(
        #     self.input_size,
        #     self.hidden_size,
        #     self.layers,
        #     dropout=self.dropout,
        #     bidirectional=self.bi,
        #     batch_first=True)

        self.rnn = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True)

        self.gpu = gpu


    def forward(self, inputs, hidden, cell, input_lengths):

        x = pack_padded_sequence(inputs, input_lengths, batch_first=True)

        output, (hidden_state, cell_state) = self.rnn(x, (hidden, cell))
        output, _ = pad_packed_sequence(output, batch_first= True, padding_value=0.)

        # if self.bi:
        #     output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        #     print('bi output', output.shape)
        return output, hidden_state, cell_state

    def init_hidden(self, batch_size):
        #init hidden for gru
        first_dim = self.layers
        if self.bi:
            first_dim = first_dim*2
        # print("first_dim", first_dim)

        h0 = Variable(torch.zeros(first_dim, batch_size, self.hidden_size))
        # print("h0 shape", h0.shape)
        if self.gpu:
            h0 = h0.cuda()
        return h0

    def init_hidden_lstm(self, batch_size):
        #init hidden for lstm
        first_dim = self.layers
        if self.bi:
            first_dim = first_dim*2
        # print("first_dim", first_dim)

        h0 = Variable(torch.zeros(first_dim, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(first_dim, batch_size, self.hidden_size))

        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0



class AspectDetector():
    def __init__(self, hidden_size):
        super(AspectDetector, self).__init__()

        out_features = 50
        self.aspect = nn.Sequential(
            nn.Linear(hidden_size, out_features),
            nn.Linear(out_features, 12),
        )

    def forward(self, x):
        return self.aspect(x)

class SentimentDetector():
    def __init__(self, hidden_size, aspect_size):
        super(SentimentDetector, self).__init__()

        out_features = 50
        self.c_aspect = nn.Linear(aspect_size, out_features)
        self.h_features = nn.Linear(hidden_size, out_features)
        self.sent = nn.Linear(out_features*2, 3)

    def forward(self, ht, aspect):
        c = self.c_aspect(aspect)
        h = self.h_features(ht)
        merged = torch.cat((c,h),1)
        return self.sent(merged)
