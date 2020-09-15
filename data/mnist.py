import numpy as np 
import torch
from torchvision import datasets
from cv2 import resize
from PIL import Image
from scipy.ndimage.interpolation import rotate
from torch.utils.data import Dataset
from .transforms import build_transforms


class MNISTDataset(Dataset):
	def __init__(self, data_dir, split, input_size, transforms=[]):
		self.split = split
		self.input_size = input_size
		self.data = datasets.MNIST(data_dir, train=self.split=="train", download=True).data
		if self.split=="test":
			self.data = list(map(self._anomalize, self.data))

		self.transforms = build_transforms(transforms, input_size, train=self.split=="train")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		x = self.data[index]
		return self.transforms(x)

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