
#The script will take out some images out of main set for experiments
#on malignant images or benign images separately

import os
import shutil
import glob
import random

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

input_dir = r"F:\Datasets\DigestPath\scene_generation\all\1000\256"
output_dir = r"F:\Datasets\DigestPath\scene_generation\all\1000\256\split2"
split_factor = 0.8

mask_dir = os.path.join(input_dir,"bimasks")
image_dir = os.path.join(input_dir,"images")

output_train_dir = os.path.join(output_dir,"train")
output_test_dir = os.path.join(output_dir,"valid")

train_mask_dir = os.path.join(output_train_dir,"bimasks")
train_image_dir = os.path.join(output_train_dir,"images")
test_mask_dir = os.path.join(output_test_dir,"bimasks")
test_image_dir = os.path.join(output_test_dir,"images")

mkdir(train_mask_dir)
mkdir(train_image_dir)
mkdir(test_mask_dir)
mkdir(test_image_dir)


mask_paths = glob.glob(os.path.join(mask_dir,"*.png"))

mask_paths = [mask_path for mask_path in mask_paths if "pos" in mask_path]

imnames = [os.path.split(path)[1].split(".")[0] for path in mask_paths]
train_l = int(split_factor*len(imnames))
test_l = len(imnames) - train_l

train_imnames = random.sample(imnames,train_l)

print(train_imnames)
print(len(train_imnames))
print(len(imnames))

for imname in imnames:
    if(imname in train_imnames):
        shutil.copy(os.path.join(mask_dir,imname+'.png'), train_mask_dir)
        shutil.copy(os.path.join(image_dir,imname+'.png'), train_image_dir)
    else:
        shutil.copy(os.path.join(mask_dir, imname + '.png'), test_mask_dir)
        shutil.copy(os.path.join(image_dir, imname + '.png'), test_image_dir)

print("Done")



