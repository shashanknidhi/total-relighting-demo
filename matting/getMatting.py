#Need to convert this cell to .py file
import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image

from model.model import HumanSegment, HumanMatting
import utils
import inference


def Matting(image_path):

    model = HumanMatting(backbone='resnet50')
    model = nn.DataParallel(model).cuda().eval()
    model.load_state_dict(torch.load("./pretrained/SGHM-ResNet50.pth"))

    with Image.open(image_path) as img:
        img = img.convert("RGB")

    pred_alpha, pred_mask = inference.single_inference(model, img)

    if not os.path.exists('output_matting'):
        os.makedirs('output_matting')
    save_path = 'output_matting/fg_mask.png'
    Image.fromarray(((pred_alpha * 255).astype('uint8')), mode='L').save(save_path)