import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import time
import math

start_time = time.time()

folder = "F:/Datasets/DigestPath/safron/test/single/results_normal/images"
outpath = "F:/Datasets/DigestPath/safron/test/single/"
outfile = os.path.join(outpath,"neg46_normal_linear.png")

if not os.path.exists(outpath):
        os.makedirs(outpath)

paths = glob.glob(os.path.join(folder,"*.png"))

hight = 4610
width = 3546
stride = 236
patch_size = 256
overlap = 21

max_x = math.ceil(width/stride)
max_y = math.ceil(hight/stride)
#print(max_x)
#print(max_y)
#exit(0)
image = np.zeros((hight,width,3))
count_masks = np.zeros((hight,width,3))
k=0

def fade_side(mask,side):
    if(side == "left"):
        for i in range(1,overlap):
            mask[:,i-1,:] = float(i/(overlap))
    if(side == "right"):
        for i in range(1,overlap):
            #print(i," and ",patch_size-i)
            mask[:,patch_size-i,:] = float(i/(overlap))
    if (side == "top"):
        for i in range(1, overlap):
            mask[i-1, :, :] = float(i / (overlap))
    if (side == "bottom"):
        for i in range(1, overlap):
            mask[patch_size - i, :, :] = float(i/(overlap))
    return mask

def create_patch_mask(x,y):
    mask = np.ones((patch_size,patch_size,3))
    if(x != 1):
        mask = fade_side(mask,"left")
    if(x != max_x):
        mask = fade_side(mask,"right")
    if (y != 1):
        mask = fade_side(mask, "top")
    if (y != max_y):
        mask = fade_side(mask, "bottom")
    return mask

k=0
for path in paths:
    if('outputs' in path):
        imname = os.path.split(path)[1].split("-")[0]
        imname = imname.split("_")
        y,x = int(imname[-2]),int(imname[-1])
        print(x," ",y)
        img = Image.open(path)
        img = np.asarray(img)
        mask = create_patch_mask(int(y/stride)+1,int(x/stride)+1)
        masked_img = np.multiply(mask,img)
        #print("X => ",x," Y => ",y)
        image[x:x+patch_size,y:y+patch_size,:] += masked_img
        count_masks[x:x+patch_size,y:y+patch_size,:]+=1.0
        k+=1

count_masks = count_masks.clip(min=1)
count_masks[count_masks==2]=1.0
count_masks[count_masks==4]=2.0
print(np.unique(count_masks))

image = image/count_masks

image = image/255.0

print("--- %s seconds ---" % (time.time() - start_time))

print("Done")
matplotlib.image.imsave(outfile, image)