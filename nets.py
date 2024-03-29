# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#----------------------------------------------------------------------------
# Latent classification model

class LCNet(nn.Module):
    def __init__(self, fmaps=[6048, 2048, 512, 40], activ='relu'):
        super().__init__()
        # Linear layers
        self.fcs = nn.ModuleList()
        for i in range(len(fmaps)-1):
            in_channel = fmaps[i]
            out_channel = fmaps[i+1]
            self.fcs.append(nn.Linear(in_channel, out_channel, bias=True))
        # Activation
        if activ == 'relu':
            self.relu = nn.ReLU()
        elif activ == 'leakyrelu':
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = self.relu(layer(x))
        x = self.fcs[-1](x)
        return x

#----------------------------------------------------------------------------
# Get weight tensor for a convolution or fully-connected layer.

def get_weight(weight, gain=1, use_wscale=True, lrmul=1):
    fan_in = np.prod(weight.size()[1:]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init
    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        runtime_coef = he_std * lrmul
    else:
        runtime_coef = lrmul
    return weight * runtime_coef


#----------------------------------------------------------------------------
# Apply activation func.

def apply_bias_act(x, act='linear', alpha=None, gain=None):
    if act == 'linear':
        return x
    elif act == 'lrelu':
        if alpha is None:
            alpha = 0.2
        if gain is None:
            gain = np.sqrt(2)
        x = F.leaky_relu(x, negative_slope=alpha)
        x = x*gain
        return x


#----------------------------------------------------------------------------
# Fully-connected layer.

class Dense_layer(nn.Module):
    def __init__(self, input_size, output_size, gain=1, use_wscale=True, lrmul=1):
        super(Dense_layer, self).__init__()  
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        w = get_weight(self.weight, gain=self.gain, use_wscale=self.use_wscale, lrmul=self.lrmul)
        b = self.bias
        x = F.linear(x, w, bias=b)
        return x

#----------------------------------------------------------------------------
# Mapping network to modify the disentangled latent w+.

class F_mapping(nn.Module):
    def __init__(
        self,
        dlatent_size            = 512,          # Transformed latent (W) dimensionality.
        mapping_layers          = 18,            # Number of mapping layers.
        mapping_fmaps           = 512,          # Number of activations in the mapping layers.
        mapping_lrmul           = 1,         # Learning rate multiplier for the mapping layers.
        mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        dtype                   = torch.float32,    # Data type to use for activations and outputs.
        **_kwargs):                             # Ignore unrecognized keyword args.
        super().__init__()

        self.mapping_layers = mapping_layers
        self.act = mapping_nonlinearity
        self.dtype = dtype

        self.dense = nn.ModuleList()
        # Mapping layers.
        for layer_idx in range(mapping_layers):
            self.dense.append(Dense_layer(mapping_fmaps, mapping_fmaps, lrmul=mapping_lrmul))
        
    def forward(self, latents_in, coeff):
        # Inputs.
        latents_in = latents_in.type(self.dtype)
        
        x = latents_in.split(split_size=512, dim=1)
        out = []
        # Mapping layers.
        for layer_idx in range(self.mapping_layers):
            out.append(apply_bias_act(self.dense[layer_idx](x[layer_idx]), act='linear'))
        x = torch.cat(out, dim=1)

        coeff = coeff.view(x.size(0), -1)
        x = coeff * x + latents_in
        
        # Output.
        assert x.dtype == self.dtype 
        return x


#----------------------------------------------------------------------------
# Modification of the Mapping network to accept multilabel training
class F_mapping_multi(nn.Module):
    def __init__(
        self,
        dlatent_size            = 512,              # Transformed latent (W) dimensionality.
        mapping_layers          = 18,               # Number of mapping layers.
        mapping_fmaps           = 512,              # Number of activations in the mapping layers.
        bottleneck_size         = 1,                # Size of the bottleneck. 18 is as there is no bottleneck
        mapping_lrmul           = 1,                # Learning rate multiplier for the mapping layers.
        mapping_nonlinearity    = 'lrelu',          # Activation function: 'relu', 'lrelu', etc.
        dtype                   = torch.float32,    # Data type to use for activations and outputs.
        n_attributes            = 2,                # Number of attributes it should learn 
        **_kwargs):                                 # Ignore unrecognized keyword args.
        super().__init__()

        self.mapping_layers = mapping_layers
        self.act = mapping_nonlinearity
        self.dtype = dtype
        self.n_attributes = n_attributes

        # Mapping layers.
        self.dense = nn.ModuleList()
        for _ in range(mapping_layers):
            self.dense.append(Dense_layer(mapping_fmaps, mapping_fmaps, lrmul=mapping_lrmul))

        # Adding fully connected layer that will combine the coefficients and the transformer output
        self.fc0 = Dense_layer(mapping_fmaps * mapping_layers, mapping_fmaps * bottleneck_size)
        self.fc = Dense_layer(n_attributes * mapping_fmaps * bottleneck_size, mapping_fmaps*mapping_layers)
        
    def forward(self, latents_in, coeffs, scaling=1):
        # Inputs.
        latents_in = latents_in.type(self.dtype)
        
        x = latents_in.split(split_size=512, dim=1)
        out = []

        # Mapping layers.
        for layer_idx in range(self.mapping_layers):
            out.append(apply_bias_act(self.dense[layer_idx](x[layer_idx]), act='linear'))

        x = torch.cat(out, dim=1)
        x = apply_bias_act(self.fc0(x))

        coeffs = coeffs.transpose(0,1)
        x = x * coeffs * scaling

        x = x.reshape(1, -1)

        x = apply_bias_act(self.fc(x))

        x = x + latents_in
        
        # Output.
        assert x.dtype == self.dtype 
        return x
