import numpy as np
import matplotlib.pyplot as plt
import cv2, os  # OpenCV for image processing

image_name = 'neg_46_1888_708.png'
# Load the image (replace 'your_image_path.jpg' with the actual path to your image)
image_path0 = os.path.join(r'F:\Datasets\DigestPath\safron\Benign\test\3\ablation_study_neg_46\cropped_real', image_name)
image_path1 = os.path.join(r'F:\Datasets\DigestPath\safron\Benign\test\3\ablation_study_neg_46\cropped_safron', image_name)
image_path2 = os.path.join(r'F:\Datasets\DigestPath\safron\Benign\test\3\ablation_study_neg_46\cropped_safron_patchadv', image_name)

# Choose a row or column for the profile (here, we're using a row)
# You can change 'row_index' to plot a column instead
row_index = 478  # Adjust this to the desired row or column index

def plot_2d_profile(image_path0, image_path1, image_path2):

    image0 = cv2.imread(image_path0, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # Extract the intensity values along the chosen row or column
    profile0 = image0[row_index, :]
    profile1 = image1[row_index, :]
    profile2 = image2[row_index, :]

    # plt.subplot(3, 1, 1)
    # plt.plot(profile0, color='b', label='Profile')
    # plt.xlabel('Pixel Position')
    # plt.ylabel('Intensity')
    # plt.title('Real 2D Profile of Image along Row {}'.format(row_index))
    # plt.legend()
    # plt.grid()

    plt.subplot(2, 1, 1)
    plt.plot(profile1, color='b', label='Profile')
    plt.xlabel('Pixel Position')
    plt.ylabel('Intensity')
    # plt.title('SAFRON 2D Profile of Image along Row {}'.format(row_index))
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(profile2, color='b', label='Profile')
    plt.xlabel('Pixel Position')
    plt.ylabel('Intensity')
    # plt.title('Patch ADV 2D Profile of Image along Row {}'.format(row_index))
    plt.legend()
    plt.grid()

    plt.show()


plot_2d_profile(image_path0, image_path1, image_path2)