from math import log10, sqrt
from PIL import Image
import cv2
import numpy as np
import os
from skimage.measure import compare_ssim
import glob
import random

real_folder = "F:/Datasets/DigestPath/safron/Benign/test/3/1024_500/images"
syn_folder = "F:/Datasets/DigestPath/safron/Benign/test/3/1024_500/safron_tmi_3_100"

def remove_alpha_channel(img):
    if img.shape[2] == 4:
        img = img[:,:,:3]
    return img

def PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(img1,img2):
    # 4. Convert the images to grayscale
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(img1, img2,
                                 full=True,
                                 win_size=501,
                                 multichannel=True)
    return score
    #print("SSIM: {}".format(score))

image_paths = glob.glob(os.path.join(syn_folder, "*.png"))
max_image_num = 1000
psnr = ssim = 0
image_paths = random.sample(image_paths, max_image_num)
for path in image_paths:
    imname = os.path.split(path)[1]
    # im1 = cv2.imread(path)
    # im2 = cv2.imread(os.path.join(real_folder,imname))
    im1 = remove_alpha_channel(np.asarray(Image.open(path)))
    im2 = remove_alpha_channel(np.asarray(Image.open(os.path.join(real_folder,imname))))
    #psnr_c = PSNR(im1, im2)
    ssim_c = SSIM(im1, im2)
    #print(imname, " ", ssim_c)
    ssim+=ssim_c

l = len(image_paths)
print("Total images ",l)
#print("Average PSNR value is dB ",psnr/len(image_paths))
print("Average SSIM score is ",ssim/l)