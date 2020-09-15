from torchvision import transforms


transforms_map = {
	"HorizontalFlip": transforms.RandomHorizontalFlip,
	"VerticalFlip": transforms.RandomVerticalFlip,
	"Rotation": transforms.RandomRotation
}

def build_transforms(transforms_list, input_size, train):
	if train:
		img_transforms = [transforms.ToPILImage(), transforms.RandomResizedCrop(input_size)]
	else:
		img_transforms = [transforms.ToPILImage(), transforms.Resize(input_size)]
	for (transf, transf_args) in transforms_list:
		if transf_args is not None:
			img_transforms.append(transforms_map[transf](**transf_args))
		else:
			img_transforms.append(transforms_map[transf]())
	img_transforms.append(transforms.ToTensor())
	return transforms.Compose(img_transforms)