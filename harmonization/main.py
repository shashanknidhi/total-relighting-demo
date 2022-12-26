import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from .src import model
def get_harmonized(composite_img,mask_img):
    comp = Image.open(composite_img).convert('RGB')
    mask = Image.open(mask_img).convert('1')
    if comp.size[0] != mask.size[0] or comp.size[1] != mask.size[1]:
        print('The size of the composite image and the mask are inconsistent')
    # convert to tensor
    comp = tf.to_tensor(comp)[None, ...]
    mask = tf.to_tensor(mask)[None, ...]
    # pre-defined arguments
    cuda = torch.cuda.is_available()
    
    
    harmonizer = model.Harmonizer()
    if cuda:
        harmonizer = harmonizer.cuda()
    harmonizer.load_state_dict(torch.load('pretrained/harmonizer.pth'), strict=True)
    if cuda:
        comp = comp.cuda()
        mask = mask.cuda()

    with torch.no_grad():
        arguments = harmonizer.predict_arguments(comp, mask)
        harmonized = harmonizer.restore_image(comp, mask, arguments)

    output_img = tf.to_pil_image(harmonized.squeeze())
    output_img.save('harmonized.jpg')
    return output_img
