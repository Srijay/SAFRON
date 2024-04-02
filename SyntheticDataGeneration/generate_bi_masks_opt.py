import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

glands_folder = "F:/Datasets/CRAG_LabServer/SAFRON/c2/synthetic_data_provider/random_generation/bigger_glands/glands"
outdir = "F:/Datasets/CRAG_LabServer/SAFRON/c2/synthetic_data_provider/random_generation/bigger_glands/masks"

if not os.path.exists(outdir):
        os.makedirs(outdir)

gland_image_paths = glob.glob(os.path.join(glands_folder, "*.png"))

for gland_image_path in gland_image_paths:

    image_name = os.path.split(gland_image_path)[1]
    gland_mask = Image.open(gland_image_path)
    gland_mask = np.asarray(gland_mask)

    if gland_mask.shape[2] == 4:
        gland_mask = gland_mask[:,:,:3]

    size = gland_mask.shape[0]

    random_mask = np.empty([size,size,3])

    random_mask[:, :, 0][gland_mask[:, :, 0] == 128] = 255  # gland : Green color
    random_mask[:, :, 1][gland_mask[:, :, 1] == 128] = 255  # gland : Green color
    random_mask[:, :, 2][gland_mask[:, :, 2] == 128] = 255  # gland : Green color

    print("Done")
    random_mask = random_mask/255.0
    matplotlib.image.imsave(os.path.join(outdir,image_name), random_mask)
    #exit(0)