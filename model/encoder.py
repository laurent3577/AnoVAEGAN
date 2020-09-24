import torch
from torch import nn


class EncoderDecoder(nn.Module):
    """
    Adapted from 
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, in_channels, role,  ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        Parameters:
            in_channels (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the first conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(EncoderDecoder, self).__init__()
        kw = 4
        padw = 1
        self.role = role
        if role == "encoder":
            sequence = [nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw), norm_layer(ndf), nn.ReLU()]
            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):  # gradually increase the number of filters
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                    norm_layer(ndf * nf_mult),
                    nn.ReLU()
                ]
        else:
            sequence = [nn.ConvTranspose2d(ndf, in_channels, kernel_size=kw, stride=2, padding=padw)]
            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):  # gradually increase the number of filters
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                sequence += [
                    nn.ReLU(),
                    norm_layer(ndf * nf_mult_prev),
                    nn.ConvTranspose2d(ndf * nf_mult, ndf * nf_mult_prev, kernel_size=kw, stride=2, padding=padw, bias=False)
                ]
            sequence = sequence[::-1]
        self.blocks = nn.Sequential(*sequence)

        if role == "encoder":
            self.mean_head = self._build_head(ndf*nf_mult, norm_layer)
            self.logvar_head = self._build_head(ndf*nf_mult, norm_layer)

    @staticmethod
    def _build_head(n_dim, norm_layer):
        return nn.Sequential(
            nn.Conv2d(n_dim, n_dim, 1),
            norm_layer(n_dim),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.blocks(x)
        if self.role == "encoder":
            return self.mean_head(x), self.logvar_head(x)
        else:
            return x


