import torch.nn as nn
import torch

rec_loss_map = {
	'L1':'rec',
	'BCE':'logits'
}

def kld_loss(mu, logvar):
	l = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return l/mu.numel()

def build_loss(loss):
	if loss=="L1":
		return nn.L1Loss(reduction='mean')
	elif loss=="MSE":
		return nn.MSELoss(reduction='mean')
	elif loss=="KLD":
		return kld_loss
	elif loss=="BCE":
		return nn.BCEWithLogitsLoss(reduction='mean')