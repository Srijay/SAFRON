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


def get_images(paths):
    images = []
    for path in paths:
        img = Image.open(path)
        img = numpy.asarray(img)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        images.append(img)
    images = numpy.array(images)
    return images


#calculate frechet inception distance
def calculate_fid_activations(model, paths1, paths2):

    curr_length = len(paths1)

    images_random = randint(0, 255, curr_length * size * size * 3)
    images_random = images_random.reshape((curr_length, size, size, 3))
    images_random = images_random.astype('float32')

    images_real = get_images(paths1)
    images_gen = get_images(paths2)

    # pre-process images
    images_real = preprocess_input(images_real)
    images_gen = preprocess_input(images_gen)
    images_random = preprocess_input(images_random)

    images_real = numpy.array(images_real)
    images_gen = numpy.array(images_gen)

    # calculate activations
    act_real = model.predict(images_real)
    act_gen = model.predict(images_gen)
    act_random = model.predict(images_random)

    return act_real,act_gen,act_random


def calculate_fid_scores(act1,act2):
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


folder_1 = "F:/Datasets/DigestPath/safron/Benign/test/3/4096_2000/images"
folder_2 = "F:/Datasets/DigestPath/safron/Benign/test/3/4096_2000/multiscalegan_output"
scale = 0
size = 4096
max_file_num = 100
batch_size = 3

paths1 = glob.glob(os.path.join(folder_1,"*.png"))
paths2 = glob.glob(os.path.join(folder_2,"*.png"))

paths1 = random.sample(paths1,max_file_num)
paths2 = random.sample(paths2,max_file_num)

#paths2 = [x.replace("-outputs","") for x in paths2]

#file_names = list(set(paths1) & set(paths2))

length = len(paths1)

print("Number of files to be processed: ",length)

paths1_list = [paths1[i:i + batch_size] for i in range(0, length, batch_size)]
paths2_list = [paths2[i:i + batch_size] for i in range(0, length, batch_size)]

# prepare the inception v3 model
model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(size,size,3))

real_act_net,gen_act_net,random_act_net = calculate_fid_activations(model,paths1_list[0],paths2_list[0])

for i in range(1,len(paths1_list)):
    real_act, gen_act, random_act = calculate_fid_activations(model,paths1_list[i],paths2_list[i])
    real_act_net = numpy.concatenate((real_act_net,real_act),axis=0)
    gen_act_net = numpy.concatenate((gen_act_net,gen_act),axis=0)
    random_act_net = numpy.concatenate((random_act_net,random_act),axis=0)
    print(i)

fid = calculate_fid_scores(real_act_net, real_act_net)
print('FID (same): %.3f' % fid)

fid = calculate_fid_scores(real_act_net, random_act_net)
print('FID (random): %.3f' % fid)

fid = calculate_fid_scores(real_act_net, gen_act_net)
print('FID (predicted) : %.3f' % fid)