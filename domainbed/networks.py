# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
from domainbed.models.resnet import resnet50, resnet18


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SqueezeLastTwo(nn.Module):
    """
    A module which squeezes the last two dimensions,
    ordinary squeeze can be a problem for batch size 1
    """

    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams["mlp_width"])
        self.dropout = nn.Dropout(hparams["mlp_dropout"])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(hparams["mlp_width"], hparams["mlp_width"])
                for _ in range(hparams["mlp_depth"] - 2)
            ]
        )
        self.output = nn.Linear(hparams["mlp_width"], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams, network=None):
        super(ResNet, self).__init__()
        if hparams["resnet18"]:
            if network is None:
                network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
            self.network = network
            self.n_outputs = 512
        else:
            if network is None:
                network = torchvision.models.resnet50(pretrained=hparams["pretrained"])
            self.network = network
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        # del self.network.fc
        self.network.fc = Identity()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["resnet_dropout"])
        self.freeze_bn()

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        if self.hparams["freeze_bn"] is False:
            return

        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """

    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.squeezeLastTwo = SqueezeLastTwo()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = self.squeezeLastTwo(x)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], 128, hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.0)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError(f"Input shape {input_shape} is not supported")
    
    
def encoder(hparams):
    n_outputs = 128 ## small dataset --RotatedMNIST
    if hparams['dataset'] == "OfficeHome":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )
    elif hparams['dataset'] == "PACS":
        scale_weights = 12
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 256 
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )
    elif hparams['dataset'] == "RotatedMNIST":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 128
        hparams['out_dim'] = 32
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )
    elif hparams['dataset'] == "VLCS":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )
    elif hparams['dataset'] == "TerraIncognita":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )
    else:
        pass
    
    return encoder, scale_weights, pcl_weights


def fea_proj(hparams):
    if hparams['dataset'] == "OfficeHome":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['hidden_size']),
            dropout,
            nn.Linear(hparams['hidden_size'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )
    elif hparams['dataset'] == "PACS":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 256#256
        hparams['out_dim'] = 256#256
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )
    elif hparams['dataset'] == "VLCS":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )
    elif hparams['dataset'] == "RotatedMNIST":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 32
        hparams['out_dim'] = 32
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )
    elif hparams['dataset'] == "TerraIncognita":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )
    else:
        pass
    
    return fea_proj, fc_proj
