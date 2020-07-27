import os, sys
import numpy as np
import time
import copy
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

from dataloader import SentihoodDataset
from model.nn_model import LSTMModel
from utils.data_util import pad_collate

# comment out warnings if you are testing it out
import warnings

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="LSTMModel")

parser.add_argument(
    "--exp-name", default="vanilla_lstm", help="Experiment name",
)

parser.add_argument(
    "--save-freq", type=int, default=1, help="every x epochs save weights",
)

args = parser.parse_args()

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = SentihoodDataset(
    data_dir, dataset_type="test", transform=None, condition_on_number=False
)

valid_dataset = SentihoodDataset(
    data_dir, dataset_type="dev", transform=None, condition_on_number=False
)


accumulation_steps = 10
batch_size = 5
val_batch_size = 1
dataloaders = {
    "train": DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=pad_collate
    ),
    "val": DataLoader(
        valid_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate,
    ),
}
exp_name = args.exp_name

experiments_path = os.path.join(current_dir, "experiments")
exp_name_path = os.path.join(experiments_path, exp_name)
os.chdir(experiments_path)

if os.path.exists(exp_name):
    shutil.rmtree(exp_name)

# make weights dir too
os.mkdir(exp_name)
os.chdir(exp_name)
os.mkdir("weights")
os.chdir(current_dir)


def calc_loss(
    pred_aspects, true_aspects, pred_sentiment, true_sentiment, metrics, aspect_weight=0.5
):
    aspect_bce = F.binary_cross_entropy_with_logits(pred_aspects, true_aspects)

    sent_bce = F.binary_cross_entropy_with_logits(pred_sentiment, true_sentiment)
    loss = aspect_bce * aspect_weight + sent_bce * (1 - aspect_weight)

    metrics["loss"] += loss.data.cpu().numpy() * true_aspects.size(0)
    metrics["aspect bce"] += sent_bce.data.cpu().numpy() * true_aspects.size(0)
    metrics["sent bce"] += aspect_bce.data.cpu().numpy() * true_sentiment.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase, epoch):
    # print(phase)
    outputs = []
    outputs.append("{}:".format(str(epoch)))
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    f = open("log.txt", "w")
    f.write("{}: {}".format(phase, ", ".join(outputs)))
    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, num_epochs=25):

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            count = 0
            for (
                embedded_text,
                lens,
                target_index,
                aspect_logit,
                c_aspect,
                sentiment_one_hot,
            ) in tqdm(dataloaders[phase]):
                count += 1
                aspect_logit, c_aspect, sentiment_one_hot = (
                    torch.stack(aspect_logit),
                    torch.stack(c_aspect),
                    torch.stack(sentiment_one_hot),
                )
                embedded_text, lens, aspect_logit, c_aspect, sentiment_one_hot = (
                    embedded_text.to(device),
                    lens.to(device),
                    aspect_logit.to(device),
                    c_aspect.to(device),
                    sentiment_one_hot.to(device),
                )

                # # zero the parameter gradients
                # optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    encoded = model.encode(embedded_text, lens)
                    pred_logits, sent_pred = model.decode(encoded, target_index, c_aspect)
                    loss = calc_loss(
                        pred_logits, aspect_logit, sent_pred, sentiment_one_hot, metrics,
                    )

                    loss = loss / accumulation_steps
                    if phase == "train":
                        loss.backward()

                    if phase == "train" and (count + 1) % accumulation_steps == 0:
                        optimizer.step()
                        # zero the parameter gradients
                        optimizer.zero_grad()

                # statistics
                epoch_samples += embedded_text.size(0)

            print_metrics(metrics, epoch_samples, phase, epoch)
            epoch_loss = metrics["mtl"] / epoch_samples

            # deep copy the model
            if phase == "val" and args.save_freq % epoch == 0:
                print("saving weights")
                best_model_wts = copy.deepcopy(model.state_dict())
                os.chdir(os.path.join(exp_name_path, "weights"))
                save_name = str(epoch) + "best_model_weights.pt"
                torch.save(best_model_wts, save_name)
                os.chdir(current_dir)

        time_elapsed = time.time() - since
        print("{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(768, 100, 0.0, device, 768).to(device)

# initialize weights
for name, param in model.named_parameters():
    if "bias" in name:
        torch.nn.init.constant_(param, 0.0)
    if "weight" in name:
        torch.nn.init.xavier_normal_(param)


# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)
train_model(model, optimizer_ft, num_epochs=100)
