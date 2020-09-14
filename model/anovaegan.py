import torch
import torch.nn as nn 
from .encoder import EncoderDecoder

class AnoVAEGAN(nn.Module):
	""" Combination of VAE and GAN for anomaly detection.
	"""
	def __init__(self, in_channels, n_layers, ndf=64):
		super(AnoVAEGAN, self).__init__()
		self.encoder = EncoderDecoder(in_channels, "encoder", ndf=ndf, n_layers=n_layers)
		self.decoder = EncoderDecoder(in_channels, "decode", ndf=ndf, n_layers=n_layers)

	def forward(self, x):
		mu, logvar = self.encoder(x)
		if self.training:
			std = torch.exp(0.5*logvar)
			eps = torch.randn_like(std)

			z = mu + std * eps
		else:
			z = mu
		logits = self.decoder(z)
		out = {
			'mu': mu,
			'logvar': logvar,
			'logits': logits,
			'rec': torch.sigmoid(logits)
		}
		return out