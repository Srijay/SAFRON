import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

glands_folder = "F:/Datasets/CRAG_LabServer/SAFRON/c2/synthetic_data_provider/random_generation/bigger_glands/glands"
outdir = "F:/Datasets/CRAG_LabServer/SAFRON/c2/synthetic_data_provider/random_generation/bigger_glands/trimasks"

if not os.path.exists(outdir):
        os.makedirs(outdir)

gland_image_paths = glob.glob(os.path.join(glands_folder, "*.png"))
num_images = len(gland_image_paths)

for k in range(0,num_images):
    gland_mask = Image.open(gland_image_paths[k])
    image_name = os.path.split(gland_image_paths[k])[1]

    gland_mask = np.asarray(gland_mask)
    size = gland_mask.shape[0]


    prob_count=0

    if gland_mask.shape[2] == 4:
        gland_mask = gland_mask[:,:,:3]

    stromal_mask = np.random.randint(10, size=(size,size))

    random_mask = np.empty([size,size,3])

    random_mask[:, :, 0][stromal_mask[:, :] <= prob_count] = 0  # white background : Blue color
    random_mask[:, :, 1][stromal_mask[:, :] <= prob_count] = 0  # white background : Blue color
    random_mask[:, :, 2][stromal_mask[:, :] <= prob_count] = 255  # white background : Blue color

    random_mask[:, :, 0][stromal_mask[:, :] > prob_count] = 255  # stroma : Red color
    random_mask[:, :, 1][stromal_mask[:, :] > prob_count] = 0  # stroma : Red color
    random_mask[:, :, 2][stromal_mask[:, :] > prob_count] = 0  # stroma : Red color

    random_mask[:, :, 0][gland_mask[:, :, 0] == 128] = 0  # gland : Green color
    random_mask[:, :, 1][gland_mask[:, :, 1] == 128] = 255  # gland : Green color
    random_mask[:, :, 2][gland_mask[:, :, 2] == 128] = 0  # gland : Green color

    print("Done")
    random_mask = random_mask/255.0

    matplotlib.image.imsave(os.path.join(outdir,image_name), random_mask)