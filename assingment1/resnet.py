import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from normalization import NoNorm, BatchNorm, InstanceNorm, LayerNorm, GroupNorm, BatchInstanceNorm

def layer_normalization(dim, norm_type):
    if norm_type == "inbuilt":
        return nn.BatchNorm2d(dim)

    elif norm_type == "bin":
        return BatchNorm(num_features=dim)
        # return nn.BatchNorm2d(dim)

    elif norm_type == "nn":
        return NoNorm()
        # return nn.Identity()

    elif norm_type == "in":
        return InstanceNorm(num_features=dim)
        # return nn.InstanceNorm2d(dim)

    elif norm_type == "ln":
        return LayerNorm(num_features=dim)
        # return nn.LayerNorm(dim)
    
    elif norm_type == "gn":
        return GroupNorm(num_features=dim)
        # return nn.GroupNorm(num_groups=4, num_channels=dim)

    elif norm_type == "bin":
        return BatchInstanceNorm(num_features=dim)
        # return nn.BatchNorm2d(dim)

    else:
        pass

class ResidualBlock(nn.Module):
    """
    Class: ResidualBlock of Resnet Architecture
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None, norm_type = "inbuilt_bn"):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type

        self.conv1 = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=stride, 
            padding=1, bias=False
        )


        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=1, 
            padding=1, bias=False
        )

        self.relu = nn.ReLU()
        self.norm1 = layer_normalization(self.out_channels, self.norm_type)
        self.norm2 = layer_normalization(self.out_channels, self.norm_type)
    def forward(self,x):
        x_residual = x
        if self.downsample is not None:
            x_residual = self.downsample(x)
        out = self.norm1(self.conv1(x))
        out = self.relu(out)
        out = self.norm2(self.conv2(out))
        out += x_residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """
    Class: ResNet Model
    number of layers  : 6n+2
    classes : r

    layers description : 
    1) first hidden (convolution) layer processing the input of size 256×256.
    2)  n layers with feature map size 256×256
    3)  n layers with feature map size 128×128
    4)  n layers with feature map size 64×64
    5)  fully connected output layer with r units

    Number of filters : 16, 32, 64, respectively of size 3x3
    Residual connections between each block of 2 layers, except for the first convolutional layer and the output layer.
    *** No Normalization is used in the model ***
    There are residual connections between each block of 2 layers
    Whenever down-sampling, we use the convolutional layer with stride of 2. 
    Appropriate zero padding is done at each layer so that there is no change in size due to boundary effects
    Final hidden layer does a mean pool over all the features before feeding into the output layer.
    """

    def __init__(self, n_channels = [16,32,64], n_layers = [2,2,2], n_classes = 25, norm_type = "inbuilt_bn"):
        super(ResNet,self).__init__()

        self.n_channels = n_channels    
        self.n_layers = n_layers
        self.n_classes = n_classes


        #first hidden layer
        self.conv = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = layer_normalization(n_channels[0], norm_type)
        self.relu = nn.ReLU()
        self.in_channels = n_channels[0]
        self.out_channels = 0
        self.features = None
        
        layers = dict()
        for c in range(len(n_channels)):
            layer = list()
            self.out_channels = n_channels[c]
            n = n_layers[c]
            
            for l in range(n):
                downsample = None                
                if self.in_channels != self.out_channels:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, 
                                  stride=2, padding=1, bias=False), 
                        layer_normalization(self.out_channels, norm_type)
                    )
                    
                if c > 0 and l == 0:
                    stride = 2
                else:
                    stride = 1
                layer.append(ResidualBlock(self.in_channels, self.out_channels, stride = stride, downsample = downsample))
                if l == 0:
                    self.in_channels = self.out_channels       
            layers[c+1] = layer

        self.layer1 = nn.Sequential(*layers[1])
        self.layer2 = nn.Sequential(*layers[2])
        self.layer3 = nn.Sequential(*layers[3])
        self.avg_pool = nn.AvgPool2d(kernel_size = 64)
        self.fc = nn.Linear(64, n_classes)


    def forward(self,x):
        # print(f'Size of input {x.shape}')
        # input convolution
        x = self.norm(self.conv(x))
        x = self.relu(x)

        # residual layers
        x = self.layer1(x)
        # print(f'Layer 1 done .Size after layer 1 {x.shape}')
        x = self.layer2(x)
        # print(f'Layer 2 done .Size after layer 2 {x.shape}')
        x = self.layer3(x)
        # print(f'Layer 3 done .Size after layer 3 {x.shape}')

        # average pool
        x = self.avg_pool(x)
        # print(f'Size after average pooling {x.shape}')
        #save the features for visualization
        self.features = x.view(-1).detach().cpu()
        x = x.view(-1,64)

        x = self.fc(x)
        return x

    def get_features(self):
        return self.features


