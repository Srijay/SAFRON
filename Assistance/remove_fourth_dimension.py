from PIL import Image
import os, sys
import glob
import numpy as np
from PIL import ImageFilter

path = "F:/Datasets/BCI/cyclegan/results/HE_to_IHC"
outdir = "F:/Datasets/BCI/cyclegan/results/HE_to_IHC_filtered"

if not os.path.exists(outdir):
        os.makedirs(outdir)

dirs = os.listdir(path)

image_paths = glob.glob(os.path.join(path,"*.png"))

for path in image_paths:
    imname = os.path.split(path)[1]
    savepath = os.path.join(outdir,imname)
    im = Image.open(path)
    im = im.convert('RGB')
    im = im.filter(ImageFilter.MedianFilter(5))
    im = im.filter(ImageFilter.SHARPEN())
    im.save(savepath)
    print(path)