import glob
import os
from PIL import Image
import numpy as np
import PIL
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Pool

folder_path = r"F:\Datasets\FayyazImages\ruqayya"
images_input_folder = folder_path

output_dir = r"F:\Datasets\FayyazImages\ruqayya\cropped"
images_output_folder = output_dir

if not os.path.exists(images_output_folder):
        os.makedirs(images_output_folder)

patch_size = 256
stride = 200
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def CropImage(path):
    image_im = Image.open(path)
    image_initial = os.path.split(path)[1].split('.')[0]

    width, height = image_im.size

    x = 0
    y = 0
    right = 0
    bottom = 0

    while (bottom < height):
        while (right < width):
            left = x
            top = y
            right = left + patch_size
            bottom = top + patch_size
            if (right > width):
                offset = right - width
                right -= offset
                left -= offset
            if (bottom > height):
                offset = bottom - height
                bottom -= offset
                top -= offset

            im_crop_name = image_initial + "_" + str(x) + "_" + str(y) + ".png"

            im_crop_image = image_im.crop((left, top, right, bottom))

            im_crop_image_np = np.asarray(im_crop_image)

            output_image_path = os.path.join(images_output_folder, im_crop_name)
            im_crop_image.save(output_image_path)

            x += stride

        x = 0
        right = 0
        y += stride

avgs = []
mbt_index = 0
image_paths = glob.glob(os.path.join(images_input_folder,"*.png"))
image_names = []

if __name__ == '__main__':
    with Pool(10) as p:
        p.map(CropImage, image_paths)

print('done')
# pbar = tqdm(total=len(image_paths))
# for path in image_paths:
#
#     image_im = Image.open(path)
#     image_im = np.array(image_im)
#     image_initial = os.path.split(path)[1].split('.')[0]
#
#     for i in range(0,image_im.shape[0], stride):
#         for j in range(0, image_im.shape[1], stride):
#
#             img_patch = image_im[i:i+patch_size,j:j+patch_size,:]
#             output_image_path = os.path.join(
#                 images_output_folder,
#                 '{}_{}_{}.png'.format(image_initial,i,j)
#             )
#
#             # Image.fromarray(img_patch).save(output_image_path)
#             matplotlib.image.imsave(output_image_path, img_patch)
#     # mbt_index = CropImage(path, mbt_index)
#     pbar.set_description('processing {} '.format(image_initial))
#     pbar.update()

