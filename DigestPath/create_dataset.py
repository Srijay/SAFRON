
#The script will take out some images out of main set for experiments
#on malignant images or benign images separately

from utils import *
import os
import shutil
import glob
import random

input_dir = "F:/Datasets/DigestPath"
output_dir = "F:/Datasets/DigestPath/safron/Malignant"
total_data_size = 100

mask_dir = os.path.join(input_dir,"masks")
image_dir = os.path.join(input_dir,"images")

output_mask_dir = os.path.join(output_dir,"masks")
output_image_dir = os.path.join(output_dir,"images")

mkdir(output_mask_dir)
mkdir(output_image_dir)

mask_paths = glob.glob(os.path.join(mask_dir,"*.png"))

imnames = [os.path.split(path)[1].split(".")[0] for path in mask_paths]
imnames = [imname for imname in imnames if "pos" in imname]

imnames = random.sample(imnames,total_data_size)

print(imnames)
print(len(imnames))

for imname in imnames:
    shutil.copy(os.path.join(mask_dir,imname+'.png'), output_mask_dir)
    shutil.copy(os.path.join(image_dir,imname+'.jpg'), output_image_dir)

print("Done")



