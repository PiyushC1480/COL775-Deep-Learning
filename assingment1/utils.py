import argparse
import os
import numpy as np
import torch
import json
from resnet import ResNet
import matplotlib.pyplot as plt
from glob import glob

norm_type_mappings = {"inbuilt_bn": "Pytorch Batch Normalization", "no_norm": "No Normalization", 
"batch_norm": "Batch Normalization", "instance_norm": "Instance Normalization","layer_norm": "Layer Normalization", "batch_instance_norm": "Batch Instance Normalization",
"group_norm": "Group Normalization"}

def plot_train_val_stats(args,date_time):
    loss_file = os.path.join(args.result_dir, "loss_tracker_{}_{}.json".format(args.norm_type, date_time))
    accu_file = os.path.join(args.result_dir, "accuracy_tracker_{}_{}.json".format(args.norm_type, date_time))

    with open(loss_file, "r+") as file:
        loss_dict = json.load(file)
    
    with open(accu_file, "r+") as file:
        accu_dict = json.load(file)
    best_accu_epoch = np.argmax(accu_dict['val'])

    fig = plt.figure()
    plt.plot(list(range(len(loss_dict['train']))), loss_dict['train'], c="tab:green", label="Train Loss")
    plt.plot(list(range(len(loss_dict['val']))), loss_dict['val'], c="tab:orange", label="Val Loss")
    plt.axvline(best_accu_epoch, linestyle="dotted", label = f"Early Stopping (epoch={best_accu_epoch})")
    plt.title(f"Train & Val Loss with for {norm_type_mappings[args.norm_type]}")
    plt.xlabel("# Epochs")
    plt.ylabel("Loss values")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "train_val_loss.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "train_val_loss.pdf"), dpi= 300, pad_inches=0.1)


    fig = plt.figure()
    plt.plot(list(range(len(accu_dict['train']))), accu_dict['train'], c="tab:green", label="Train Accuracy")
    plt.plot(list(range(len(accu_dict['val']))), accu_dict['val'], c="tab:orange", label="Val Accuracy")
    plt.axvline(best_accu_epoch, linestyle="dotted", label = f"Early Stopping (epoch={best_accu_epoch})")
    plt.title(f"Train & Val Loss with for {norm_type_mappings[args.norm_type]}")
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "train_val_accu.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "train_val_accu.pdf"), dpi= 300, pad_inches=0.1)

    return


def plot_quantiles(args, date_time):
    quantile_file = os.path.join(args.result_dir, "ft_quantile_tracker_{}_{}.json".format(args.norm_type, date_time))
    with open(quantile_file, "r+") as file:
        quantile_dict = json.load(file)

    fig = plt.figure()
    plt.plot(list(range(100)), quantile_dict['1'], c="tab:blue", label="1$^{st}$ Quantile")
    plt.plot(list(range(100)), quantile_dict['20'], c="tab:orange", label="20$^{th}$ Quantile")
    plt.plot(list(range(100)), quantile_dict['80'], c="tab:green", label="80$^{th}$ Quantile")
    plt.plot(list(range(100)), quantile_dict['99'], c="tab:purple", label="99$^{th}$ Quantile")
    plt.title(f"Quantile plot for {norm_type_mappings[args.norm_type]}")
    plt.xlabel("# Epochs")
    plt.ylabel("Feature Value")
    plt.legend()
    
    fig.savefig(os.path.join(args.result_dir, "ft_quantile_plot.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "ft_quantile_plot.pdf"), dpi= 300, pad_inches=0.1)

    return


def plot_loss_comparison(args):

    fig = plt.figure()
    path = os.path.join(args.result_dir, "**/loss_tracker*.json")
    json_list = glob(path)
    json_list.sort()
    for file in json_list:
        with open(file, "r+") as f:
            json_dict = json.load(f)
        
        norm_type = file.split("/")[-2]
        if norm_type != "torch_bn":
            print(norm_type)
            plt.plot(list(range(100)), json_dict['train'], label=norm_type.upper())
        
    plt.title(f"Train Loss variation with Epochs")
    plt.xlabel("# Epochs")
    plt.ylabel("Loss Values")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "train_loss_all.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "train_loss_all.pdf"), dpi= 300, pad_inches=0.1)

    fig = plt.figure()
    json_list = glob(os.path.join(args.result_dir, "**/loss_tracker*.json"))
    json_list.sort()
    for file in json_list:
        with open(file, "r+") as f:
            json_dict = json.load(f)
        
        norm_type = file.split("/")[-2]
        if norm_type != "torch_bn":
            # print(norm_type)
            plt.plot(list(range(100)), json_dict['val'], label=norm_type.upper())
        
    plt.title(f"Validation Loss variation with Epochs")
    plt.xlabel("# Epochs")
    plt.ylabel("Loss Values")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "val_loss_all.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "val_loss_all.pdf"), dpi= 300, pad_inches=0.1)

    return

def plot_accu_comparison(args):

    fig = plt.figure()
    json_list = glob(os.path.join(args.result_dir, "**/accuracy_tracker*.json"))
    json_list.sort()
    for file in json_list:
        with open(file, "r+") as f:
            json_dict = json.load(f)
        
        norm_type = file.split("/")[-2]
        if norm_type != "torch_bn":
            print(norm_type)
            plt.plot(list(range(100)), json_dict['train'], label=norm_type.upper())
        
    plt.title(f"Train Accuracy variation with Epochs")
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "train_accu_all.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "train_accu_all.pdf"), dpi= 300, pad_inches=0.1)

    fig = plt.figure()
    json_list = glob(os.path.join(args.result_dir, "**/accuracy_tracker*.json"))
    json_list.sort()
    for file in json_list:
        with open(file, "r+") as f:
            json_dict = json.load(f)
        
        norm_type = file.split("/")[-2]
        if norm_type != "torch_bn":
            print(norm_type)
            plt.plot(list(range(100)), json_dict['val'], label=norm_type.upper())
        
    plt.title(f"Validation Accuracy variation with Epochs")
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "val_accu_all.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "val_accu_all.pdf"), dpi= 300, pad_inches=0.1)

    return

def plot_time_comparison(args):

    fig = plt.figure()
    json_list = glob(os.path.join(args.result_dir, "**/time_tracker*.json"))
    json_list.sort()
    for file in json_list:
        with open(file, "r+") as f:
            json_dict = json.load(f)
        
        norm_type = file.split("/")[-2]
        if norm_type != "torch_bn":
            print(norm_type)
            plt.plot(list(range(100)), json_dict['train'], label=norm_type.upper())
        
    plt.title(f"Training Time v/s Epochs")
    plt.xlabel("# Epochs")
    plt.ylabel("Time (Mins)")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "train_time_all.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "train_time_all.pdf"), dpi= 300, pad_inches=0.1)

    return