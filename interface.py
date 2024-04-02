import os
import glob
import argparse
import shutil
from PIL import Image
import PIL
import gradio as gr
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", help="path to tensorflow model",
                    default="./models/digestpath/benign/3/tmi_3_100")
parser.add_argument('--d_normalization', default='batchnorm')

args = parser.parse_args()
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def create_image(im_array):
    #create temporary directory
    tmp_dir = "./tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    #determine name and size of image
    im = Image.fromarray(im_array)
    input_path = os.path.join(tmp_dir,"sample_mask.png")
    im.save(input_path)
    width,height = im.size
    imname = os.path.split(input_path)[1]
    # Create patches from input component mask
    mask_patches_path = os.path.join(tmp_dir, "mask_patches")
    os.system('python ./Assistance/SingleImageCropper.py --image_path ' + input_path + ' --output_dir ' + mask_patches_path)
    copy_path = os.path.join(tmp_dir, "mask_patches_copy")
    shutil.copytree(mask_patches_path, copy_path)
    paired_path = os.path.join(tmp_dir, "paired")
    os.system("python tools/process.py --input_dir " + copy_path + " --b_dir " + mask_patches_path + " --operation combine --output_dir " + paired_path)
    #Compute output patches using generator
    results_dir = os.path.join(tmp_dir, "results")
    os.system("python segment2tissue_safron_media.py --mode test --scale_size 296 --output_dir " + results_dir + " --input_dir " + paired_path + " --checkpoint " + args.model_path)
    #Join patches into single file
    output_path = os.path.join(tmp_dir,imname)
    os.system("python ./Assistance/join_images.py --patches_dir " + os.path.join(results_dir,"images") + " --output_file " + output_path + " --im_height " + str(height) + " --im_width " + str(width))
    #Cleanup files
    shutil.rmtree(tmp_dir)
    output_img = Image.open(output_path)
    return output_img


demo = gr.Interface(
    create_image,
    inputs=["image"],
    outputs="image",
    title="SAFRON: Stitching Across the Frontier Network for Generating Colorectal Cancer Histology Images"
)

demo.launch()

