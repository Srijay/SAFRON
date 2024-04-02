from PIL import Image
import os, sys
import glob

path = r"F:\Datasets\DigestPath\scene_generation\all\1000\masks"
outdir = r"F:\Datasets\DigestPath\scene_generation\all\1000\256\masks"
resize_len = 256

if not os.path.exists(outdir):
        os.makedirs(outdir)

dirs = os.listdir(path)

image_paths = glob.glob(os.path.join(path,"*.png"))

for path in image_paths:
    imname = os.path.split(path)[1]
    savepath = os.path.join(outdir,imname)
    im = Image.open(path)
    imResize = im.resize((resize_len,resize_len), Image.ANTIALIAS)
    imResize.save(savepath)