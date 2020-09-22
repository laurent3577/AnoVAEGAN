import numpy as np 
import torch
from torchvision import datasets
from scipy.ndimage.interpolation import rotate
from .dataset import BaseDataset


class MNISTDataset(BaseDataset):
	def __init__(self, data_dir, split, input_size, transforms=[]):
		super(MNISTDataset, self).__init__(data_dir, split, input_size, transforms)

	def _get_data(self, data_dir, split):
		data = datasets.MNIST(data_dir, download=True, train=self.split=="train").data
		if self.split == "test":
			data = list(map(self._anomalize, data))
		return data

	def _anomalize(self, img):
		img = img.numpy()/255.
		theta = np.random.randint(30,151)
		rotation = rotate(img, theta, reshape=False)
		rotation = rotation * (rotation>0.8)
		np.clip(rotation, 0., 1., rotation)
		xmin, ymin = np.min(np.where(img>0)[0]), np.min(np.where(img>0)[1])
		xmax, ymax = np.max(np.where(img>0)[0]), np.max(np.where(img>0)[1])
		fuse_mask = (img == 0).astype(dtype=int)

		fuse_mask[xmin:xmax,ymin:ymax] = 0
		anom = img + rotation * fuse_mask
		return torch.FloatTensor(anom)