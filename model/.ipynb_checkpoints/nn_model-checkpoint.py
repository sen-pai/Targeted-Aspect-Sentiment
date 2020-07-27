import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *


class LSTMModel(nn.Module):
    def __init__(self, n_channels, hidden_size, encoder_dropout, gpu, aspect_dim,  bidirectional_encoder = True ):
        super(Model, self).__init__()

        self.input_size = n_channels
        self.hidden_size = hidden_size
        self.dropout = encoder_dropout
        self.bi = bidirectional_encoder
        self.gpu = gpu
        self.aspect_dim = aspect_dim

        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.dropout, self.gpu)
        self.aspect_detector = AspectDetector(self.hidden_size)
        self.sentiment_detector = SentimentDetector(self.hidden_size, self.aspect_dim)

    def encode(self, batch_text, text_lens):

        batch_size = batch_text.size()[0]
        init_state, init_cell = self.encoder.init_hidden_lstm(batch_size)

        encoder_outputs, encoder_hidden_state, _ = self.encoder.forward(batch_text, init_state, init_cell, text_len)

        return encoder_outputs, encoder_hidden_state, encoder_cell_state

    # def decode(self, ):
