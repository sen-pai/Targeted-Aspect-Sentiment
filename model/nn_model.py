import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *


class LSTMModel(nn.Module):
    def __init__(
        self,
        n_channels,
        hidden_size,
        encoder_dropout,
        device,
        aspect_dim,
        bidirectional_encoder=True,
    ):
        super(LSTMModel, self).__init__()

        self.input_size = n_channels
        self.hidden_size = hidden_size
        self.dropout = encoder_dropout
        self.bi = bidirectional_encoder
        self.device = device
        self.aspect_dim = aspect_dim

        # if self.bi:
        #     self.hidden_size *= 2

        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.dropout, self.device)
        self.aspect_detector = AspectDetector(self.hidden_size * 2)
        self.sentiment_detector = SentimentDetector(self.hidden_size * 2, self.aspect_dim)

    def encode(self, batch_text, text_lens):

        batch_size = batch_text.size()[0]
        init_state, init_cell = self.encoder.init_hidden_lstm(batch_size)

        encoder_outputs, encoder_hidden_state, _ = self.encoder.forward(
            batch_text, init_state, init_cell, text_lens
        )

        return encoder_outputs

    def decode(self, encoder_outputs, target_index, c_aspects):
        batch_size = encoder_outputs.size()[0]
        indexed_outputs = self.get_indexed_encoder_outputs(encoder_outputs, target_index).to(
            self.device
        )

        aspect_pred_logits = self.aspect_detector(indexed_outputs)
        sentiment_pred = self.sentiment_detector(indexed_outputs, c_aspects)

        return aspect_pred_logits, sentiment_pred

    @classmethod
    def get_indexed_encoder_outputs(self, encoder_outputs, target_index):
        batch_size = encoder_outputs.size()[0]
        feed_in = torch.zeros((batch_size, encoder_outputs.size()[2]))
        for i, index in enumerate(target_index):
            feed_in[i] = encoder_outputs[i][index[0]]

        return feed_in
