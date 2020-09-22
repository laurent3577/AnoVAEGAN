from torch.utils.data import Dataset
from .transforms import build_transforms

class BaseDataset(Dataset):
	def __init__(self,data_dir, split, input_size, transforms=[]):
		self.split = split
		self.input_size = input_size
		self.data = self._get_data(data_dir, self.split)
		self.transforms = build_transforms(transforms, input_size, train=self.split=="train")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		x = self.data[index]
		return self.transforms(x)

	def _get_data(self, data_dir, split):
		raise NotImplementedError