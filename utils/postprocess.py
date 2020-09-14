import numpy as np
import torch
from cv2 import resize
import os
from PIL import Image

def to_RGB(x):
    return torch.cat([x,x,x], dim=1)

def combine_detect_image(img, detection):
    if img.size(1) != 3:
        img = to_RGB(img)
        detection = to_RGB(detection)
    out = img.clone()
    out[detection>0] = 0
    detection[:,1,:,:]= 0
    detection[:,2,:,:] = 0
    out = out + detection
    return img.numpy(), out.numpy()

def save_compare(size, list_comps, output_dir):
    l = len(list_comps)
    grid = np.ones((l*(size+20), 2*size+40 , 3))
    for ind, ims in enumerate(list_comps):
        im0 = resize(ims[0].transpose(1,2,0), (size, size))
        im1 = resize(ims[1].transpose(1,2,0), (size, size))
        grid[(size+20)*ind:size*(ind+1)+ind*20, 10:size+10, :] = im0
        grid[(size+20)*ind:size*(ind+1)+ind*20,size+40:, :] = im1
    grid *= 255
    pil_image = Image.fromarray(grid.astype(np.uint8))
    with open(os.path.join(output_dir, 'test_results.png'), mode='wb') as f:
        pil_image.save(f, 'PNG')
