import torch
from torch import nn
import functools


class Discriminator(nn.Module):
    """
    Adapted from 
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, in_channels, input_size, n_layers=3, ndf=64, norm_layer=nn.BatchNorm2d):
        """
        Parameters:
            in_channels (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the first conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        self.conv_blocks = nn.Sequential(*sequence)
        spatial_dim = ((input_size[0])*(input_size[1]))//(2**(2*n_layers))
        self.fc = nn.Sequential(
            nn.Linear(ndf*nf_mult*spatial_dim, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1))

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)