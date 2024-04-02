import glob
import os
from PIL import Image
import numpy as np
import time
import PIL
import matplotlib.pyplot as plt
import argparse

patchsize = 1024
stride = 500
max_background_threshold = 0

def CropImage(image_path,output_dir,mbt_index):

    image_name = os.path.split(image_path)[1]

    image_name = image_name.split('.')[0]
    im = Image.open(image_path)

    new_im = im

    width, height = new_im.size

    x = 0
    y = 0
    right = 0
    bottom = 0

    while (bottom < height):

        while (right < width):
            left = x
            top = y
            right = left + patchsize
            bottom = top + patchsize

            if (right > width):
                offset = right - width
                right -= offset
                left -= offset
            if (bottom > height):
                offset = bottom - height
                bottom -= offset
                top -= offset

            im_crop_name = image_name + "_" + str(left) + "_" + str(top) + ".png"

            im_crop_image = new_im.crop((left, top, right, bottom))
            im_crop_image_np = np.asarray(im_crop_image)

            save_flag = True
            if (np.mean(im_crop_image_np[:, :, 2]) >= 230):  # blue background
                if (mbt_index >= max_background_threshold):
                    save_flag = False
                else:
                    mbt_index = mbt_index + 1

            if (save_flag):
                output_image_path = os.path.join(output_dir, im_crop_name)
                im_crop_image.save(output_image_path)

            x += stride

        x = 0
        right = 0
        y += stride

    return mbt_index


start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to image to crop",
                    default="F:/Datasets/DigestPath/safron/Benign/test/3/results/results_tmi_3_100_older")
parser.add_argument("--output_dir", help="path to output folder",
                    default="F:/Datasets/DigestPath/safron/Benign/test/3/results/results_tmi_3_100_older/1024_500")

args = parser.parse_args()

if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

input_paths = glob.glob(os.path.join(args.input_dir,"*.png"))
mbt_index = 0

for path in input_paths:
    mbt_index = CropImage(path,args.output_dir,mbt_index)

print("--- %s seconds ---" % (time.time() - start_time))