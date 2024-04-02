import skimage.measure
import os
from PIL import Image
import numpy as np
import random

folder = "F:/Datasets/DigestPath/safron/Benign/test/3/1024_500/safron_tmi_3_100"
# folder = "F:/Datasets/DigestPath/safron/Benign/test/3/1024_500/images"
# folder = "F:/Datasets/DigestPath/safron/Benign/test/3/1024_500/safron_tmi_3_100"
# folder = "F:/Datasets/DigestPath/safron/Benign/test/3/1024_500/multiscalegan_output"

max_file_num = 1000

def compute_avg_entropy(folder):
    paths = random.sample(os.listdir(folder),max_file_num)
    entropies = []
    for fname in paths:
        img = Image.open(os.path.join(folder, fname))
        img = np.asarray(img)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        entropy = skimage.measure.shannon_entropy(img)
        entropies.append(entropy)
    return np.mean(entropies)

print(compute_avg_entropy(folder))