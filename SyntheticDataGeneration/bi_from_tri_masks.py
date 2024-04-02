import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

outdir = "F:/Datasets/DigestPath/safron/Benign/train/3/masks"

if not os.path.exists(outdir):
        os.makedirs(outdir)

masks_folder = "F:/Datasets/DigestPath/safron/Benign/train/3/trimasks"
masks_paths = glob.glob(os.path.join(masks_folder, "*.png"))

for mask_path in masks_paths:
    mask = Image.open(mask_path)
    image_name = os.path.split(mask_path)[1]

    mask = np.asarray(mask)

    height,width = mask.shape[0],mask.shape[1]

    if mask.shape[2] == 4:
        mask = mask[:,:,:3]

    stromal_mask = np.random.randint(10, size=(width,height))

    random_mask = np.empty([height,width,3])

    random_mask[:, :, 0][mask[:, :, 1] == 255] = 255  # gland : Green color
    random_mask[:, :, 1][mask[:, :, 1] == 255] = 255  # gland : Green color
    random_mask[:, :, 2][mask[:, :, 1] == 255] = 255  # gland : Green color

    print("Done")
    random_mask = random_mask/255.0

    matplotlib.image.imsave(os.path.join(outdir,image_name), random_mask)