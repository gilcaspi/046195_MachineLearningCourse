# imports
import matplotlib.pyplot as plt
from skimage import io
import matplotlib
import os
from skimage.transform import resize, rescale, downscale_local_mean
import numpy as np
from scipy.linalg import eigh as largest_eigh

# initializing
path = os.getcwd()

# define functions
def rgb2gray(path):
    plt.gray()
    gray1 = io.imread(path, as_grey=True)
    matplotlib.image.imsave(path + "_gray.jpg", gray1)


def run_all_files(path, command, skip, change):
    for subdir, dirs, files in os.walk(path):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            if filepath.endswith(skip):
                continue
            elif filepath.endswith(change):
                if os.path.getsize(filepath) > 0:
                    command(filepath)



def downsample_4_image(path):
    image = io.imread(path, as_grey=True)
    image_downsampled = downscale_local_mean(image, (4, 4))
    plt.gray()
    matplotlib.image.imsave(path + "_downsampled.jpg", image_downsampled)


def get_k_largest_eigenfaces(X, N, k, average, gil_sized, johnny_sized):
    vals_good, vecs_good = largest_eigh(((np.dot(np.transpose(X), X)) / num_of_pics), eigvals=(N - k, N - 1))
    good_first_gil = average
    good_first_johnny = average
    for col in range(0, k):
        eigenvec_pic_good = np.ndarray.reshape(vecs_good[:, col], (50, 45))
        # eigenvec_pic_final_good = eigenvec_pic_good * (255 / np.max(eigenvec_pic_good))
        good_first_gil = good_first_gil + (gil_sized * eigenvec_pic_good) * eigenvec_pic_good
        good_first_johnny = good_first_johnny + (johnny_sized * eigenvec_pic_good) * eigenvec_pic_good

    plt.gray()
    matplotlib.image.imsave(path + "\johnny.jpg" + "_%d_first_johnny_rec.jpg" % (k), good_first_johnny)
    matplotlib.image.imsave(path + "\gil.jpg" + "_%d_first_gil_rec.jpg" % (k), good_first_gil)


##Change all Pics to grey color##
run_all_files(path, rgb2gray, "_gray.jpg", ".jpg")
##Downsample pictures from 180*200 -> 45*50##
run_all_files(path+"\pics", downsample_4_image, "_downsampled.jpg", ".jpg_gray.jpg")
##PCA Algorithm##

# Get Average Picture#
# Initializing
average_image = np.zeros((50, 45))
num_of_pics = 0

# run on all files in folder and sub-folders and calculate average picture
for subdir, dirs, files in os.walk(path):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith("downsampled.jpg"):
            if os.path.getsize(filepath) > 0:
                num_of_pics = num_of_pics + 1
                average_image = average_image + io.imread(filepath, as_grey=True)
            continue
        continue
    continue

average_image = average_image / num_of_pics
plt.gray()
matplotlib.image.imsave(path + r"\resized_average.jpg", average_image)
print("Average: number of images = ", num_of_pics)

# Center Pictures#
# Initializing #
num_of_pics = 0

# run on all files in folder and sub-folders and decrease average picture xi <= (xi - average) #
for subdir, dirs, files in os.walk(path):
    for file in files:
        # print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        if filepath.endswith("downsampled.jpg"):
            if os.path.getsize(filepath) > 0:
                num_of_pics = num_of_pics + 1
                current_pic = io.imread(filepath, as_grey = True)
                current_pic = current_pic - average_image
                plt.gray()
                matplotlib.image.imsave(filepath+r"_centered.jpg", current_pic)

            continue
        continue
    continue

print("Center : number of images = ",num_of_pics)

# Calc Pn#
# Initializing
num_of_pics = 0
d = 50 * 45  # pixels
X = np.ones((1, d))
# run on all files in folder and sub-folders and decrease average picture xi <= (xi - average)
for subdir, dirs, files in os.walk(path):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith("downsampled.jpg_centered.jpg"):
            if os.path.getsize(filepath) > 0:
                num_of_pics = num_of_pics + 1
                current_pic = io.imread(filepath, as_grey=True)
                current_pic_vec = np.ndarray.reshape(current_pic, (1, d))
                if X.all() == 1:
                    X = current_pic_vec
                    continue
                else:
                    X = np.vstack((X, current_pic_vec))
                continue
            continue
        continue
    continue
print("Calc Pn : number of images = ", num_of_pics)

# calculate first 10 eigenfaces
N = d
k = 10
vals, vecs = largest_eigh(((np.dot(np.transpose(X), X)) / num_of_pics), eigvals=(N - k, N - 1))
print("calculate first 10 eigenfaces : dim of the first eigenvector is: ", vecs.shape)
plt.gray()
for col in range(0, 10):
    eigenvec_pic = np.ndarray.reshape(vecs[:, col], (50, 45))
    eigenvec_pic_final = eigenvec_pic * (255 / np.max(eigenvec_pic))
    newpath = path +"\%d_eigenvector.jpg" % (col + 1)
    matplotlib.image.imsave(newpath + "_lower_contrast.jpg", eigenvec_pic_final)
    plt.imsave(newpath, eigenvec_pic_final, vmin=0, vmax=255)

# pre-processing own pics
print("Pre Processing own images")
johnny = io.imread(path + "\johnny.jpg", as_grey=True)
gil = io.imread(path + "\gil.jpg", as_grey=True)
johnny_regular = resize(johnny, (50, 45))
johnny_sized = johnny_regular - average_image
gil_regular = resize(gil, (50, 45))
gil_sized = gil_regular - average_image
plt.gray()
matplotlib.image.imsave(path + "\johnny.jpg" + "_regular.jpg", johnny_regular)
matplotlib.image.imsave(path + "\gil.jpg" + "_regular.jpg", gil_regular)
matplotlib.image.imsave(path + "\johnny.jpg" + "_fixed.jpg", johnny_sized)
matplotlib.image.imsave(path + "\gil.jpg" + "_fixed.jpg", gil_sized)

# calc Projected Pics
print("Calculate Projected images")
for i in [1, 10, 30,100,200,300,400,500,600, 700, 800, 900, 1000, 1150, 1200, 1300, 2250]:
    get_k_largest_eigenfaces(X, d, i, average_image, gil_sized, johnny_sized)
