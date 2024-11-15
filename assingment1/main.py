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
from resnet import ResidualBlock, ResNet
from utils import plot_loss_comparison, plot_accu_comparison, plot_time_comparison, plot_train_val_stats, plot_quantiles

"""
Implementing ResNet for classefication of Birds dataset
number of layers  : 6n+2
    classes : r

    layers description : 
    1) first hidden (convolution) layer processing the input of size 256×256.
    2)  n layers with feature map size 256×256
    3)  n layers with feature map size 128×128
    4)  n layers with feature map size 64×64
    5)  fully connected output layer with r units

    number of filters : 16, 32, 64, respectively of size 3x3
    residual connections between each block of 2 layers, except for the first convolutional layer and the output layer.

    Whenever down-sampling, we use the convolutional layer with stride of 2. 
    Appropriate zero padding is done at each layer so that there is no change in size due to boundary effects
    final hidden layer does a mean pool over all the features before feeding into the output layer.
"""
"""
TODO:
1. Expweriment with different types of optimizers
2. make parser for the arguments
3. Change the learning rate and see the effect on the model
4. Experiment with different types of schedulers

"""


def data_loaders(args):
    """
    input : args : arguments from the user
    output : dataloaders : dictionary of dataloaders for train and validation set
             test_loader : dataloader for test set
             dataset_sizes : dictionary of sizes of train, validation and test set
             class_names : list of class names 
    """
    norm1 = (0.485, 0.456, 0.406)
    norm2 = (0.229, 0.224, 0.225)
    if args.aug:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm1, norm2)
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm1, norm2)
        ])
    
    data_dir = 'data/Birds_25'

    # for train and validation set
    sets = ['train','val']
    # print(os.path.join(data_dir, 'train'))
    image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x), transform=train_transforms) for x in sets}    
    dataloaders = {x : DataLoader(image_datasets[x], batch_size=args.batch_size*4, shuffle=True, num_workers=args.num_workers) for x in sets}
    dataset_sizes = {x : len(image_datasets[x]) for x in sets}
    class_names = image_datasets['train'].classes

    #for test set 
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm1, norm2)
    ])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=args.num_workers)
    dataset_sizes.update({'test' : len(test_dataset)})

    print(f'Sizes of train set, validation and test set : {dataset_sizes}')
    return dataloaders, test_loader, dataset_sizes, class_names



def train_model(args, dataloaders : dict, dataset_sizes:dict):
    """
    input : args : arguments from the user (parameters  : epochs, optim, scheduler, batch_size, num_workers, aug)
            dataloader : dictionary of dataloaders for train and validation set
            dataset_sizes : dictionary of sizes of train, validation and test set
            
    output : model : trained model
             best_model_wts : best model weights
             best_acc : best accuracy
    """

    model = ResNet(n_layers=[args.n,args.n,args.n],norm_type=args.norm_type)
    # print(model)
    model = model.to(device)

    #Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizers = {
        'sgd': optim.SGD(model.parameters(), lr=0.1,weight_decay=1e-4, momentum=0.9), 
        'adam': optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-08, amsgrad=False),
        'adagrad': optim.Adagrad(model.parameters(), lr=0.1, weight_decay=1e-4, lr_decay=1e-6,eps = 1e-08)
    }
    current_optimizer  = optimizers[args.optim]

    schedulers = {
        'step_scheduler' : lr_scheduler.StepLR(current_optimizer, step_size=20, gamma=0.1, last_epoch=-1, verbose=False),
        'cosine_scheduler'  : lr_scheduler.CosineAnnealingLR(current_optimizer, args.epochs, verbose=False)
    }
    current_scheduler = schedulers[args.scheduler]
    # print(f'Optimizer : {args.optim}')
    # print(f'Scheduler : {args.scheduler}')
    # print(f'Batch Size : {args.batch_size}')

    # stats
    loss_tracker = {'train':[], 'val':[]}
    accu_tracker = {'train':[], 'val':[]}
    ft_quantile_tracker = {'1':[], '20':[], '80':[], '99':[]}
    time_tracker = {'train':[]}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_epoch = 0


    t0 = time.time()
    print("\n ---------------------- MODEL Training and Validation ----------------------\n")
    for epoch in range(args.epochs):
        print('-'*10)
        print(f'Epoch {epoch+1}/{args.epochs}')

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            feature_list = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predicts = torch.max(outputs,1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        current_optimizer.zero_grad()
                        loss.backward()
                        current_optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predicts == labels.data)
                feature_list.extend(list(model.get_features().view(-1,1).numpy()))
            if phase == 'train':
                current_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            loss_tracker[phase].append(epoch_loss)
            accu_tracker[phase].append(epoch_acc.item())

            if(phase =='val'):
                ft_quantile_tracker['1'].append(np.percentile(feature_list, 1))
                ft_quantile_tracker['20'].append(np.percentile(feature_list, 20))
                ft_quantile_tracker['80'].append(np.percentile(feature_list, 80))
                ft_quantile_tracker['99'].append(np.percentile(feature_list, 99))

        t1 = time.time()
        print(f"{phase} Epoch : {epoch} , Total Time : {t1-t0:.2f}s, Train Loss : {loss_tracker['train'][-1]:.4f}, Val Loss : {loss_tracker['val'][-1]:.4f}, Train Acc : {accu_tracker['train'][-1]:.4f}, Val Acc : {accu_tracker['val'][-1]:.4f}")
        time_tracker['train'].append(t1-t0)
        model_state = {
            'epoch' : epoch,
            'accuracy' : epoch_acc.item(),
            'best_acc' : best_acc.item(),
            'best_acc_epoch' : best_acc_epoch,
        }
        print('Saving the model')
        torch.save(model, os.path.join(args.checkpoint_dir, f"latest_checkpoint_{args.norm_type}.pth"))
        with open(os.path.join(args.checkpoint_dir, f"training_progress_{args.norm_type}.json"), "w") as file:
            json.dump(model_state, file)
        if(phase == 'val' and epoch_acc > best_acc):
            print(f'Best model found at epoch {epoch} with accuracy : {epoch_acc}, previous best accuracy : {best_acc}')
            best_acc = epoch_acc
            best_acc_epoch = epoch
            model_state = {
                'epoch' : epoch,
                'accuracy' : epoch_acc.item(),
                'best_acc' : best_acc.item(),
                'best_acc_epoch' : best_acc_epoch,
            }
            best_model_wts = copy.deepcopy(model.state_dict())
            # print(f'Best model found at epoch {epoch} with accuracy {best_acc:.4f}, previous best accuracy : {accu_tracker["val"][-2]:.4f}')
            print('Saving the best model found')
            #get model from GPU

            torch.save(model, os.path.join(args.checkpoint_dir, f"best_checkpoint_{args.norm_type}.pth"))
            with open(os.path.join(args.checkpoint_dir, "training_progress_{args.norm_type}.json"), "w") as file:
                json.dump(model_state, file)

        with open(os.path.join(args.result_dir, "loss_tracker_{}_{}.json".format(args.norm_type,date_time)), "w") as outfile:
            json.dump(loss_tracker, outfile)

        with open(os.path.join(args.result_dir, "accuracy_tracker_{}_{}.json".format(args.norm_type,date_time)), "w") as outfile:
            json.dump(accu_tracker, outfile)

        with open(os.path.join(args.result_dir, "time_tracker_{}_{}.json".format(args.norm_type,date_time)), "w") as outfile:
            json.dump(time_tracker, outfile)

        with open(os.path.join(args.result_dir, "ft_quantile_tracker_{}_{}.json".format(args.norm_type,date_time)), "w") as outfile:
            json.dump(ft_quantile_tracker, outfile)

        #print all things
        print(f"Loss Tracker : {loss_tracker}")
        print(f"Accuracy Tracker : {accu_tracker}")
        print(f"Time Tracker : {time_tracker}")
        print(f"Feature Quantile Tracker : {ft_quantile_tracker}")
        
    

    time_elapsed = time.time() -t0
    print(f'Training completed in {time_elapsed//60 :.0f}m {time_elapsed%60:.0f}s')
    model.load_state_dict(best_model_wts)
    return model


def test_model(args, test_loader : DataLoader,dataloaders:dict, model : ResNet, dataset_sizes : dict,class_names:list):
    """
    input : args : arguments from the user (parameters  : epochs, optim, scheduler, batch_size, num_workers, aug)
            test_loader : dataloader for test set
            model : trained model
            dataset_sizes : dictionary of sizes of train, validation and test set
            class_names : list of class names

    output: None
    """
    with torch.no_grad():
        # test iteration
        model = model.to(device)
        model.eval()   

        actuals = []
        predictions = []

        for idx, batch in enumerate(test_loader):

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            actuals.extend(labels.squeeze().tolist())
            outputs = model(images)
            predictions.extend(torch.argmax(outputs, dim=1).squeeze().tolist())

        accu = round(accuracy_score(actuals, predictions)*100,2)
        micro_f1 = round(f1_score(actuals, predictions, average='micro'), 4)
        macro_f1 = round(f1_score(actuals, predictions, average='macro'), 4)

        test_accuracy = "Test Accuracy = " + str(accu) + "%"
        test_micro_f1 = "Test Micro-F1 Score = " + str(micro_f1)
        test_macro_f1 = "Test Macro-F1 Score = " + str(macro_f1)
        actuals = []
        predictions = []

        for idx, batch in enumerate(dataloaders['train']):

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            actuals.extend(labels.squeeze().tolist())
            outputs = model(images)
            predictions.extend(torch.argmax(outputs, dim=1).squeeze().tolist())

        accu = round(accuracy_score(actuals, predictions)*100,2)
        micro_f1 = round(f1_score(actuals, predictions, average='micro'), 4)
        macro_f1 = round(f1_score(actuals, predictions, average='macro'), 4)

        train_accuracy = "Train Accuracy = " + str(accu) + "%"
        train_micro_f1 = "Train Micro-F1 Score = " + str(micro_f1)
        train_macro_f1 = "Train Micro-F1 Score = " + str(macro_f1)

        actuals = []
        predictions = []

        for idx, batch in enumerate(dataloaders['val']):

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            actuals.extend(labels.squeeze().tolist())
            outputs = model(images)
            predictions.extend(torch.argmax(outputs, dim=1).squeeze().tolist())

        accu = round(accuracy_score(actuals, predictions)*100,2)
        micro_f1 = round(f1_score(actuals, predictions, average='micro'), 4)
        macro_f1 = round(f1_score(actuals, predictions, average='macro'), 4)

        val_accuracy = "Val Accuracy = " + str(accu) + "%"
        val_micro_f1 = "Val Micro-F1 Score = " + str(micro_f1)
        val_macro_f1 = "Val Macro-F1 Score = " + str(macro_f1)

        test_result = [
            args.norm_type + '_' + date_time , test_accuracy, test_micro_f1, test_macro_f1, train_accuracy,
            train_micro_f1, train_macro_f1, val_accuracy, val_micro_f1, val_macro_f1
        ]
        with open(os.path.join(args.result_dir, f"evaluation_perormance_{args.norm_type}_{date_time}.txt"), "w") as res:
            for r in test_result:
                res.writelines(r)
                res.writelines("\n")

        print(f"\n\n--------- Test Accuracy = {test_accuracy}, \nTrain Accuracy = {train_accuracy}, \nVal Accuracy = {val_accuracy} \n---------")
        print(f"\n\n--------- Test Micro-F1 Score = {test_micro_f1}, \nTrain Micro-F1 Score = {train_micro_f1}, \nVal Micro-F1 Score = {val_micro_f1} \n---------")
        print(f"\n\n--------- Test Macro-F1 Score = {test_macro_f1}, \nTrain Macro-F1 Score = {train_macro_f1}, \nVal Macro-F1 Score = {val_macro_f1} \n---------")
    return
    
def parser():
    """
    To parse the arguments from the user
    """
    parser = argparse.ArgumentParser(description='Birds Classification using ResNet')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model')
    parser.add_argument('--optim', type=str, default='sgd', help='Optimizer to use for training the model')
    parser.add_argument('--scheduler', type=str, default='step_scheduler', help='Scheduler to use for training the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument("--n", type=int, default=2, help="Number of residual blocks.")
    parser.add_argument("--r", type=int, default=25, help="Number of classes.")
    parser.add_argument("--norm_type", type=str, default="torch_bn", help="Type of layer normalization to be used.")
    parser.add_argument('--aug', type=bool, default=True, help='Whether to use data augmentation or not')
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Path to model checkpoints.")
    parser.add_argument("--result_dir", type=str, default="./results", help="Path to dataset directory.")
    return parser.parse_args()



if __name__ == "__main__":
    args = parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on : ", device)
    date_time = datetime.now().strftime("%Y%m%d_%H%M")
    dir_name = args.norm_type
    args.result_dir = os.path.join(args.result_dir, dir_name)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, dir_name)
    # print(args.result_dir)
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    print("\n", args)
    comparison_plots_only = True # CHANGE THIS FLAG BEFORE RUNNING TO SWITCH BETWEEN TRAINING AND GENERATING PLOTS
    if comparison_plots_only:
        args.result_dir = "/".join(args.result_dir.split("/")[:-1])
        plot_loss_comparison(args)
        plot_accu_comparison(args)
        plot_time_comparison(args)
    else:
        dataloaders, test_loader, dataset_sizes, class_names = data_loaders(args)
        model = train_model(args, dataloaders, dataset_sizes)
        test_model(args, test_loader ,dataloaders, model , dataset_sizes,class_names)
        plot_train_val_stats(args, date_time)
        plot_quantiles(args, date_time)