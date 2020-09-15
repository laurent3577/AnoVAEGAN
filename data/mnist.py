import numpy as np 
import mnist
from cv2 import resize
from PIL import Image
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from .transforms import build_transforms


def visualize(im, size=(100,100)):
	im = resize(im, size)
	plt.imshow(im, cmap='gray')
	plt.axis('off')
	plt.show()

def compare(size, list_comps):
	fig=plt.figure()
	l = len(list_comps)
	i = 0
	for ims in list_comps:
		im0 = resize(ims[0], size)
		im1 = resize(ims[1], size)
		fig.add_subplot(l, 2, i+1)
		plt.imshow(im0, cmap='gray')
		plt.axis('off')
		fig.add_subplot(l, 2, i+2)
		plt.imshow(im1, cmap='gray')
		plt.axis('off')
		i += 2
	plt.show()


class MNISTDataset(Dataset):
	def __init__(self, split, input_size, transforms=[]):
		self.split = split
		self.input_size = input_size
		if self.split == "train":
			self.data = mnist.train_images()/255.
		elif self.split == "test":
			self.data = mnist.test_images()/255.
			self.data = list(map(self._anomalize, self.data))
		else:
			raise ValueError("Only train or test splits supported, {} given".format(split))
		self.transforms = build_transforms(transforms, input_size, train=self.split=="train")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		x = self.data[index]
		return self.transforms(Image.fromarray(x))

	def _anomalize(self, img):
		theta = np.random.randint(30,151)
		rotation = rotate(img, theta, reshape=False)
		rotation = rotation * (rotation>0.8)
		np.clip(rotation, 0., 1., rotation)
		xmin, ymin = np.min(np.where(img>0)[0]), np.min(np.where(img>0)[1])
		xmax, ymax = np.max(np.where(img>0)[0]), np.max(np.where(img>0)[1])
		fuse_mask = (img == 0).astype(dtype=int)

		fuse_mask[xmin:xmax,ymin:ymax] = 0
		anom = img + rotation * fuse_mask
		return anom