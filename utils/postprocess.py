import numpy as np
import torch
from cv2 import resize
import os
from PIL import Image

def to_RGB(x):
    return torch.cat([x,x,x], dim=1)

def combine_detect_image(img, detection, rec):
    if img.size(1) != 3:
        img = to_RGB(img)
        detection = to_RGB(detection)
    out = img.clone()
    out[detection>0] = 0
    detection[:,1,:,:]= 0
    detection[:,2,:,:] = 0
    out = out + detection
    return img.cpu().numpy(), out.cpu().numpy(), to_RGB(rec.detach()).cpu().numpy()

def save_compare(size, list_comps, output_dir, suffix):
    l = len(list_comps)
    grid = np.ones((l*(size+20), 3*size+40 , 3))
    for ind, ims in enumerate(list_comps):
        im0 = resize(ims[0].transpose(1,2,0), (size, size))
        im1 = resize(ims[1].transpose(1,2,0), (size, size))
        im2 = resize(ims[2].transpose(1,2,0), (size, size))
        grid[(size+20)*ind:size*(ind+1)+ind*20, 10:size+10, :] = im0
        grid[(size+20)*ind:size*(ind+1)+ind*20,size+40:2*size+40, :] = im1
        grid[(size+20)*ind:size*(ind+1)+ind*20,2*size+40:, :] = im2
    grid *= 255
    pil_image = Image.fromarray(grid.astype(np.uint8))
    with open(os.path.join(output_dir, suffix + '_output.png'), mode='wb') as f:
        pil_image.save(f, 'PNG')

def save_batch_output(img, rec, thresh, save_size, output_dir, suffix):
    detection = (torch.abs(rec-img)>thresh).int()
    img, detect_image, rec = combine_detect_image(img, detection, rec)
    comp = [(i,d, r) for i,d, r in zip(img, detect_image, rec)]
    save_compare(save_size, comp, output_dir, suffix)
