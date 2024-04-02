import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
import PIL
from PIL import Image
import pandas as pd
import colorsys

PIL.Image.MAX_IMAGE_PIXELS = 933120000

masks_folder = "F:/Datasets/BCSS/masks"
outfolder = "F:/Datasets/BCSS/color_masks"
masks_paths = glob.glob(os.path.join(masks_folder,"*.png"))

codes_path = "F:/Datasets/BCSS/gtruth_codes.csv"

code_dict = pd.read_csv(codes_path, index_col=0, header=0, squeeze=True).to_dict()
code_dict = {v: k for k, v in code_dict.items()}

print(code_dict)

code_len = len(code_dict)

if not os.path.exists(outfolder):
        os.makedirs(outfolder)

def generate_colors(n):
  """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
  brightness = 0.7
  hsv = [(i / n, 1, brightness) for i in range(n)]
  colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
  colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
  return colors

def extract_image(mask_path):
    mask_name = os.path.split(mask_path)[1]
    mask_name = mask_name.split(".")[0]+".png"
    mask = Image.open(mask_path)
    mask_np = np.asarray(mask)
    w, h = mask_np.shape
    new_mk = np.empty([w, h, 3])
    for i in range(0,w):
        for j in range(0,h):
            new_mk[i][j] = colors[mask_np[i][j]]
    new_mk = new_mk / 255.0
    matplotlib.image.imsave(os.path.join(outfolder,mask_name), new_mk)
    print(mask_name)

colors = generate_colors(code_len)
print(code_len)
exit(0)

import cv2
size = 1000
ingap = int(size/code_len)+1
gap=ingap
img = np.ones((size+100,size+100,3))
img.fill(255)
for id in code_dict:
    cv2.circle(img,(10,gap), 7, colors[id], -1)
    gap += ingap
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, code_dict[id], ((15,gap-40)), font, 1, (0,0,0), 1, cv2.LINE_AA)

cv2.imwrite("test.png",img)
exit(0)

for path in masks_paths:
    extract_image(path)