import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

size = 1500
outdir = "F:/Datasets/TheCOT/GeneratedSegmentations/2/1500/trimasks"

if not os.path.exists(outdir):
        os.makedirs(outdir)

masks_folder = "F:/Datasets/TheCOT/GeneratedSegmentations/2/1500/masks"
masks_paths = glob.glob(os.path.join(masks_folder, "*.png"))

for mask_path in masks_paths:
    mask = Image.open(mask_path)
    image_name = os.path.split(mask_path)[1]

    mask = np.asarray(mask)
    prob_count=0

    if len(mask.shape)>2 and mask.shape[2] == 4:
        mask = mask[:,:,:3]

    stromal_mask = np.random.randint(10, size=(size,size))

    random_mask = np.empty([size,size,3])

    random_mask[:, :, 0][stromal_mask[:, :] <= prob_count] = 0  # white background : Blue color
    random_mask[:, :, 1][stromal_mask[:, :] <= prob_count] = 0  # white background : Blue color
    random_mask[:, :, 2][stromal_mask[:, :] <= prob_count] = 255  # white background : Blue color

    random_mask[:, :, 0][stromal_mask[:, :] > prob_count] = 255  # stroma : Red color
    random_mask[:, :, 1][stromal_mask[:, :] > prob_count] = 0  # stroma : Red color
    random_mask[:, :, 2][stromal_mask[:, :] > prob_count] = 0  # stroma : Red color

    random_mask[:, :, 0][mask[:, :] == 255] = 0  # gland : Green color
    random_mask[:, :, 1][mask[:, :] == 255] = 255  # gland : Green color
    random_mask[:, :, 2][mask[:, :] == 255] = 0  # gland : Green color

    print("Done")
    random_mask = random_mask/255.0

    matplotlib.image.imsave(os.path.join(outdir,image_name), random_mask)