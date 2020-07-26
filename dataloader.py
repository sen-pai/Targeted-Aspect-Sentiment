import numpy as np
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import transformers

from utils.data_util import parse_sentihood_json, convert_input, aspect2idx, all_aspects_logits

"""
Inputs:
Raw JSON file

Outputs:

"""

def pretrained_embeddings(text, pretrained_model , tokenizer):
    #takes text as input
    #returns bert embeddings and index of target
    tokenized = tokenizer.encode(text, add_special_tokens= False)
    input_ids = torch.tensor(tokenized).unsqueeze(0)  # Batch size 1
    outputs = pretrained_model(input_ids)
    last_hidden_states = outputs[0]
    #encoded LOCATION = 3295
    target_index = [i for i, j in enumerate(tokenized) if j == 3295]

    return last_hidden_states, target_index

def aspect_embeddings(aspect, pretrained_model, tokenizer):
    #takes text as input
    #returns bert embeddings and index of target
    tokenized = tokenizer.encode(aspect, add_special_tokens= False)
    input_ids = torch.tensor(tokenized).unsqueeze(0)  # Batch size 1
    outputs = pretrained_model(input_ids)
    last_hidden_states = outputs[0]

    return last_hidden_states

def sentiment2onehot(sentiment):
    one_hot = np.zeros(3)
    if sentiment == "Positive":
        one_hot[0] = 1
    if sentiment == "None":
        one_hot[1] = 1
    else:
        one_hot[2] = 1
    return one_hot

class SentihoodDataset(Dataset):
    def __init__(
        self,
        data_dir_path,
        dataset_type='train',
        transform=None,
        condition_on_number = True,
    ):
        # try to change back here everytime
        # self.current_path = os.getcwd()
        self.dataset_type = dataset_type
        self.transform = transform
        self.condition_on_number = condition_on_number

        self.embedding_pretrained_model = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        self.data_dir_path = data_dir_path  # data folder
        os.chdir(self.data_dir_path)
        if self.dataset_type == "train":
            self.data = parse_sentihood_json("sentihood-train.json")
        elif self.dataset_type == "dev":
            self.data = parse_sentihood_json("sentihood-dev.json")
        elif self.dataset_type == "test":
            self.data = parse_sentihood_json("sentihood-test.json")

        self.data = convert_input(self.data, aspect2idx)
        self.aspect_logits = all_aspects_logits(self.data, aspect2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        id, text, target, aspect, sentiment = self.data[i]
        embedded_text, target_index = pretrained_embeddings(text, self.embedding_pretrained_model , self.tokenizer)
        sentiment_one_hot = sentiment2onehot(sentiment)
        aspect_logit = self.aspect_logits[i//12]
        #condition of aspect number or aspect embeddings
        if self.condition_on_number:
            c_aspect = aspect2idx[aspect]
        else:
            c_aspect = aspect_embeddings(aspect,self.embedding_pretrained_model, self.tokenizer)

        return embedded_text.squeeze(0), target_index, aspect_logit, c_aspect.squeeze(0), sentiment_one_hot
