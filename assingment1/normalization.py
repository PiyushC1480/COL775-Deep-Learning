import torch
import torch.nn as nn


"""
Normalization Classes
(a) Batch Normalization (BN) 
(b) Instance Normalization (IN) 
(c) Batch-Instance Normalization (BIN)
(d) Layer Normalization (LN)
(e) Group Normalization (GN) 
"""

class NoNorm(nn.Module):
    def __init__(self):
        super(NoNorm, self).__init__()
    
    def forward(self,x):
        # no normalization applied
        return x
    
class BatchNorm(nn.Module):
    def __init__(self, num_features, eps =1e-5,momentum = 0.1, affine = True):
        super(BatchNorm, self).__init__()
        self.momentum = momentum
        self.r_mean = 0
        self.r_varience = 0
        self.eps = torch.tensor(eps)
        self.n_features = num_features
        self.affine = affine
        shape = (1, self.n_features, 1, 1)
        if self.affine:
            self.gamma = nn.Parameter(torch.empty(shape))
            self.beta = nn.Parameter(torch.empty(shape))

        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self,x):
        if self.training:
            n = x.numel() / x.size(1)
            dimensions = (0,2,3)
            var = x.var(dim=dimensions, keepdim=True, unbiased=False)
            mean = x.mean(dim=dimensions, keepdim=True)
            with torch.no_grad():
                self.r_mean = self.momentum * mean + (1 - self.momentum) * self.r_mean
                self.r_varience = self.momentum * (n/(n-1)) * var + (1 - self.momentum) * self.r_varience
        else:
            mean = self.r_mean
            var = self.r_varience
        dn = torch.sqrt(var + self.eps)
        x = (x - mean)/ dn
        if self.affine:
            x = x * self.gamma + self.beta
        return x
    
class InstanceNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1, affine = True):
        super(InstanceNorm, self).__init__()
        self.momentum = momentum
        self.r_mean = 0
        self.r_varience = 0
        self.eps = torch.tensor(eps)
        self.n_features = num_features
        self.affine = affine
        shape = (1, self.n_features, 1, 1)
        if self.affine:
            self.gamma = nn.Parameter(torch.empty(shape))
            self.beta = nn.Parameter(torch.empty(shape))

        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self,x):
        _ , n_channels , _ , _ = x.size()
        assert n_channels == self.n_features 
        mean = x.mean(dim=(2,3), keepdim=True)
        var = x.var(dim=(2,3), keepdim=True, unbiased=False)

        #normalizing the input
        x = (x - mean)/ torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.gamma + self.beta
        return x
    
class BatchInstanceNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1,rho=0.5, affine = True):
        super(BatchInstanceNorm, self).__init__()
        self.momentum = momentum
        self.r_mean = 0
        self.r_varience = 0
        self.eps = torch.tensor(eps)
        self.n_features = num_features
        self.affine = affine
        self.rho = rho
        shape = (1, self.n_features, 1, 1)
        if self.affine:
            self.gamma = nn.Parameter(torch.empty(shape))
            self.beta = nn.Parameter(torch.empty(shape))

        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self,x):
        if self.training:
            n = x.numel() / x.size(1)
            dimensions = (0,2,3)
            var_bn = x.var(dim=dimensions, keepdim=True, unbiased=False)
            mean_bn = x.mean(dim=dimensions, keepdim=True)
            with torch.no_grad():
                self.r_mean = self.momentum * mean_bn + (1 - self.momentum) * self.r_mean
                self.r_varience = self.momentum * (n/(n-1)) * var_bn + (1 - self.momentum) * self.r_varience
        else:
            mean_bn = self.r_mean
            var_bn = self.r_varience
        #batch normalier
        x_bn = (x - mean_bn)/ torch.sqrt(var_bn + self.eps)
        mean_in = x.mean(dim=(2,3), keepdim=True)
        var_in = x.var(dim=(2,3), keepdim=True)
        #instance normalizer
        x_in = (x - mean_in)/ torch.sqrt(var_in + self.eps)
        #take convex combination
        x = self.rho * x_bn + (1-self.rho) * x_in
        if self.affine:
            x = x * self.gamma + self.beta
        return x

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1, affine = True):
        super(LayerNorm, self).__init__()
        self.momentum = momentum
        self.eps = torch.tensor(eps)
        self.n_features = num_features
        self.affine = affine
        shape = (1, self.n_features, 1, 1)
        if self.affine:
            self.gamma = nn.Parameter(torch.empty(shape))
            self.beta = nn.Parameter(torch.empty(shape))

        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self,x):
        _ , n_channels , _ , _ = x.size()
        assert n_channels == self.n_features 
        mean = x.mean(dim=(1,2,3), keepdim=True)
        var = x.var(dim=(1,2,3), keepdim=True, unbiased=False)

        #normalizing the input
        x = (x - mean)/ torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.gamma + self.beta
        return x
    
class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups = 4, eps = 1e-5, affine = True):
        super(GroupNorm, self).__init__()
        self.eps = torch.tensor(eps)
        self.n_features = num_features
        self.n_groups = num_groups
        self.affine = affine
        shape = (1, self.n_features, 1, 1)
        if self.affine:
            self.gamma = nn.Parameter(torch.empty(shape))
            self.beta = nn.Parameter(torch.empty(shape))

        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self,x):
        n_feat , n_channels , height , width = x.size()

        assert n_channels % self.n_groups == 0
        assert n_channels == self.n_features 

        x = x.view(n_feat, self.n_groups, n_channels//self.n_groups, height, width)
        mean = x.mean(dim=(1,2,3), keepdim=True)
        var = x.var(dim=(1,2,3), keepdim=True, unbiased=False)

        #normalizing the input
        x = (x - mean)/ torch.sqrt(var + self.eps)
        x = x.view(n_feat, n_channels, height, width)
        if self.affine:
            x = x * self.gamma + self.beta
        return x







