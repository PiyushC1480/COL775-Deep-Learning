import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score,accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import json
import datetime
from resnet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_loaders(args, data_dir):
    """
    input : args : arguments from the user
    output : dataloaders : dictionary of dataloaders for train and validation set
             test_loader : dataloader for test set
             dataset_sizes : dictionary of sizes of train, validation and test set
             class_names : list of class names 
    """
    norm1 = (0.485, 0.456, 0.406)
    norm2 = (0.229, 0.224, 0.225)

    #for test set 
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm1, norm2)
    ])
    test_dataset = datasets.ImageFolder(data_dir, transform=test_transforms)
    class_names = test_dataset.classes
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataset_sizes = {'test' : len(test_dataset)}

    print(f'Sizes of test set : {dataset_sizes}')
    return test_loader, dataset_sizes, class_names


def test_model(given_args, test_loader : DataLoader, model : ResNet, dataset_sizes : dict,class_names:list):
    """
    input : args : arguments from the user (parameters  : epochs, optim, scheduler, batch_size, num_workers, aug)
            test_loader : dataloader for test set
            model : trained model
            dataset_sizes : dictionary of sizes of train, validation and test set
            class_names : list of class names

    output: None

    work : write the predicted class ID of the i_th image as the i_th line in the output file
    """
    with torch.no_grad():
        # test iteration
        model = model.to(device)
        model.eval()   

        actuals = []
        predictions = [] # store the predicted class ID of the i_th image as the i_th line in the output file

        for idx, batch in enumerate(test_loader):

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            actuals.extend(labels.squeeze().tolist())
            outputs = model(images)
            predictions.extend(torch.argmax(outputs, dim=1).squeeze().tolist())

        # write the predictions to the output file
        with open(given_args.output_file, 'w') as f:
            for pred in predictions:
                f.write(f'{pred}\n')


    return


def get_parser():
    """
    Generate a parameter parser
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Inference ResNet for Birds Classifier.")

    # path to data files.
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset directory.")
    
    # path to data files.
    parser.add_argument("--model_file", type=str, help="Path to trained model file.")

    # path to data files.
    parser.add_argument("--test_data_file", type=str, help="Path to a csv with each line representing an image.")

    # path to data files.
    parser.add_argument("--output_file", type=str, help="file containing the prediction in the same order as in the input csv.")


    # path to result files.
    parser.add_argument("--result_dir", type=str, default="./results", help="Path to dataset directory.")

    # path to model checkpoints.
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Path to model checkpoints.")

    # batch size training
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to be used during training.")

    # number of workers for dataloader
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers used for dataloading.")

    # max number of epochs
    parser.add_argument("--epochs", type=int, default=50, help="Number of workers used for dataloading.")

    # n
    parser.add_argument("--n", type=int, default=2, help="Number of residual blocks.")

    # r
    parser.add_argument("--r", type=int, default=25, help="Number of classes.")

    # normalization type
    parser.add_argument("--norm_type", type=str, default="inbuilt", help="Type of layer normalization to be used.")

    # normalization type
    parser.add_argument("--normalization", type=str, default="torch_bn", help="Type of layer normalization to be used.")


    # data augmentation
    parser.add_argument("--aug", type=bool, default=True, help="Whether to perform data augmentation during training.")

    # data augmentation
    # parser.add_argument("--comparison_plots_only", type=bool, default=False, help="Whether to perform data augmentation during training.")


    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # load the model
    model = ResNet(n_layers = [args.n,args.n, args.n], n_classes = args.r, norm_type=args.norm_type)
    test_loader, dataset_sizes, class_names = data_loaders(args, args.test_data_file)
    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    # load the data
    

    # test the model
    test_model(args, test_loader, model, dataset_sizes, class_names)