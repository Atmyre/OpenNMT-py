import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


import json
import os
import numpy as np


class MLP_D(nn.Module):
    def __init__(self, ninput, noutput, layers, gpu=True, activation=nn.LeakyReLU(0.2)):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            #if i != 0:
                #bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1, dim=2)
                #self.layers.append(bn)
                #self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)
#         self.layers.append(nn.Sigmoid())
#         self.add_module("activation"+str(len(self.layers)), activation)

        self.init_weights()
        
    @classmethod
    def from_opt(cls, opt):
        """Alternate constructor."""
        return cls(
            opt.enc_rnn_size,
            1,
            opt.desc_layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_G(nn.Module):
    def __init__(self, ninput, noutput, layers, gpu=True, activation=nn.ReLU()):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

#             bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
#             self.layers.append(bn)
#             self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()
        
    @classmethod
    def from_opt(cls, opt):
        """Alternate constructor."""
        return cls(
            opt.gen_input,
            opt.enc_rnn_size,
            opt.gen_layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass