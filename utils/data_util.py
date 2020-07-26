# taken from https://github.com/firascherif/ABSA-BERT-pair/blob/master/generate/data_utils_sentihood.py
# Reference: https://github.com/liufly/delayed-memory-update-entnet

from __future__ import absolute_import
import json
import operator
import os
import re
import sys
import xml.etree.ElementTree

# import nltk
import numpy as np


def parse_sentihood_json(in_file):
    with open(in_file) as f:
        data = json.load(f)
    ret = []
    for d in data:
        text = d["text"]
        sent_id = d["id"]
        opinions = []
        targets = set()
        for opinion in d["opinions"]:
            sentiment = opinion["sentiment"]
            aspect = opinion["aspect"]
            target_entity = opinion["target_entity"]
            targets.add(target_entity)
            opinions.append((target_entity, aspect, sentiment))
        ret.append((sent_id, text, opinions))
    return ret


def convert_input(data, all_aspects):
    ret = []
    for sent_id, text, opinions in data:
        for target_entity, aspect, sentiment in opinions:
            if aspect not in all_aspects:
                continue
            ret.append((sent_id, text, target_entity, aspect, sentiment))
        assert "LOCATION1" in text
        targets = set(["LOCATION1"])
        if "LOCATION2" in text:
            targets.add("LOCATION2")
        for target in targets:
            aspects = set([a for t, a, _ in opinions if t == target])
            none_aspects = [a for a in all_aspects if a not in aspects]
            for aspect in none_aspects:
                ret.append((sent_id, text, target, aspect, "None"))
    return ret


# wrote the next part myself.

aspect2idx = {
    "general": 0,
    "price": 1,
    "transit-location": 2,
    "safety": 3,
    "live": 4,
    "quiet": 5,
    "dining": 6,
    "nightlife": 7,
    "touristy": 8,
    "shopping": 9,
    "green-culture": 10,
    "multicultural": 11,
}


def all_aspects_logits(data, aspect2idx):
    aspect_logits = []
    logit = np.zeros(12)  # taking care of 0 % 12
    for i, (sent_id, text, target, aspect, sentiment) in enumerate(data, start=1):
        if i % 12 == 0:
            aspect_logits.append(logit)
            logit = np.zeros(12)
        if sentiment != "None":
            logit[aspect2idx[aspect]] = 1
    return aspect_logits


from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch):
    batch_embedded_text, a, b, c, d = zip(*batch)
    lens = [x.shape[0] for x in batch_embedded_text]

    batch_embedded_text = pad_sequence(batch_embedded_text, batch_first=True, padding_value=0)

    return batch_embedded_text, lens, a, b, c, d
