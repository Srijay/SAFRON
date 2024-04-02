# example of calculating the frechet inception distance in Keras
import numpy
import glob
import random
import os
from PIL import Image
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets.mnist import load_data
from skimage.transform import resize

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    #act1 = numpy.concatenate((act1,act1),axis=0)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

folder_1 = "F:/Datasets/pannuke/fold2/images"
folder_2 = "F:/Datasets/pannuke/fold2/results/pix2pix_fold1trained/images"

scale = 0
size = 256
max_file_num = 1000

paths1 = os.listdir(folder_1)
paths2 = os.listdir(folder_2)
paths2 = [x.replace("pred","gt") for x in paths2]

common_file_names = list(set(paths1) & set(paths2))

if(len(common_file_names)>max_file_num):
    common_file_names = random.sample(common_file_names,max_file_num)

length = len(common_file_names)

print("Number of files to be processed: ",length)

def get_images(folder,file_names,gen=False):
    images = []
    for filename in file_names:
        # if(gen):
        #     fname = filename.replace("targets","outputs")
        # else:
        #     fname = filename
        img = Image.open(os.path.join(folder,fname))
        img = numpy.asarray(img)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        images.append(img)
    return images

images_random = randint(0, 255, length*size*size*3)
images_random = images_random.reshape((length,size,size,3))
images_random = images_random.astype('float32')

images_real = get_images(folder_1,common_file_names)
images_gen = get_images(folder_2,common_file_names,gen=True)

images_real = numpy.array(images_real)
images_gen = numpy.array(images_gen)

# prepare the inception v3 model
model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(size,size,3))


# resize images
#if(scale):
#    images_real = scale_images(images_real, (299,299,3))
#    images_gen = scale_images(images_gen, (299,299,3))

# pre-process images
images_real = preprocess_input(images_real)
images_gen = preprocess_input(images_gen)
images_random = preprocess_input(images_random)

fid = calculate_fid(model, images_real, images_real)
print('FID (same): %.3f' % fid)

fid = calculate_fid(model, images_real, images_random)
print('FID (random): %.3f' % fid)

fid = calculate_fid(model, images_real, images_gen)
print('FID (predicted) : %.3f' % fid)