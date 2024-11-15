import argparse
import os
import numpy as np
import torch
import json
from resnet import ResNet
import matplotlib.pyplot as plt
from glob import glob


loss_file_1 = "/Users/piyushchauhan/Desktop/col775/assignments/a1/2021CS11010/part1/results/sgd_0.0001/batch_norm/results/batch_norm/loss_tracker_batch_norm_20240326_0614.json"

loss_file_2 = "/Users/piyushchauhan/Desktop/col775/assignments/a1/2021CS11010/part1/results/sgd_0.0001/inbuilt_bn/results/inbuilt_bn/loss_tracker_inbuilt_bn_20240326_0613.json"


with open(loss_file_1, "r+") as file:
    loss_dict_1 = json.load(file)

with open(loss_file_2, "r+") as file:
    loss_dict_2 = json.load(file)

#both training loss
fig = plt.figure()
plt.plot(list(range(len(loss_dict_1['train']))), loss_dict_1['train'], c="tab:green", label="Batch Norm")
plt.plot(list(range(len(loss_dict_2['train']))), loss_dict_2['train'], c="tab:orange", label="Inbuilt BN")
plt.title(f"Train Loss with for Batch Norm and Inbuilt BN")
plt.xlabel("# Epochs")
plt.ylabel("Loss values")
plt.legend()
fig.savefig("train_loss_comp.png", dpi= 300, pad_inches=0.1, format="png")
fig.savefig("train_loss_comp.pdf", dpi= 300, pad_inches=0.1)
