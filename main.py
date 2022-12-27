# Import Statements
import os, shutil
from PIL import Image
# Functions: Matting, Obj_placement, harmonization
from matting.getMatting import Matting
from object_placement.placement import place
from harmonization.main import get_harmonized

#Input : Foreground Image, Background Image
#Output : Harmonized Image
def main(foreground,background):
    if not os.path.exists('input'):
        os.makedirs('input')
    #Save foreground and background images to input dir as input/foreground.png and background.png
    foreground.save('input/foreground.png')
    background.save('input/background.png')
    if not os.path.exists('matting/pretrained'):
        os.mkdir('matting/pretrained')
        shutil.copy('/content/drive/MyDrive/total-relighting-demo/SGHM-ResNet50.pth','matting/pretrained')
    # Matting
    # Input: input/foreground.png
    # Output: output_matting/fg_mask.png
    #Matting(image_path='input/foreground.png')
    Matting()
    #Object Placement
    # Input: input/foreground.png, input/background.png, output_matting/fg_mask.png
    # Output: output_object_placement/composite_image.png output_object_placement/composite_mask.png
    #place(foreground='input/foreground.png',background='input/background.png',fg_mask='output_matting/fg_mask.png')
    place()
    #Harmonization
    # Input: output_object_placement/composite_image.png output_object_placement/composite_mask.png
    # Output: final/harmonized_image.png
    #harmonized_image = get_harmonized(composite_img='output_object_placement/composite_image.png',mask_img='output_object_placement/composite_mask.png')
    harmonized_image = get_harmonized()
    return harmonized_image

    

