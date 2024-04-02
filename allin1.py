import os
import glob
import argparse
import shutil
from PIL import Image
import PIL

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", help="path to image directory",
                    default="F:/Datasets/BCSS/safron/test/masks")
parser.add_argument("--output_dir", help="path to output folder",
                    default="F:/Datasets/BCSS/safron/test/results")
parser.add_argument("--model_path", help="path to tensorflow model",
                    default="./models/bcss/safron")
parser.add_argument('--d_normalization', default='batchnorm')

args = parser.parse_args()
PIL.Image.MAX_IMAGE_PIXELS = 933120000

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

def create_image(input_path):
    #create temporary directory
    tmp_dir = "./tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    else:
        shutil.rmtree(tmp_dir)
    #determine name and size of image
    im = Image.open(input_path)
    width,height = im.size
    imname = os.path.split(input_path)[1]
    #out_imname = os.path.split(input_path)[1].split('.')[0]+".jpeg"
    # Create patches from input component mask
    mask_patches_path = os.path.join(tmp_dir, "mask_patches")
    os.system('python ./Assistance/SingleImageCropper.py --image_path ' + input_path + ' --output_dir ' + mask_patches_path)
    copy_path = os.path.join(tmp_dir, "mask_patches_copy")
    shutil.copytree(mask_patches_path, copy_path)
    paired_path = os.path.join(tmp_dir, "paired")
    os.system("python tools/process.py --input_dir " + copy_path + " --b_dir " + mask_patches_path + " --operation combine --output_dir " + paired_path)
    #Compute output patches using generator
    results_dir = os.path.join(tmp_dir, "results")
    os.system("python segment2tissue_safron_media.py --mode test --scale_size 296 --output_dir " + results_dir + " --input_dir " + paired_path + " --checkpoint " + args.model_path)
    #Join patches into single file
    output_path = os.path.join(args.output_dir,imname)
    os.system("python ./Assistance/join_images.py --patches_dir " + os.path.join(results_dir,"images") + " --output_file " + output_path + " --im_height " + str(height) + " --im_width " + str(width))
    #Cleanup files
    shutil.rmtree(tmp_dir)
    print("Done")

image_paths = glob.glob(os.path.join(args.image_dir, "*.png"))
for path in image_paths:
    create_image(path)

