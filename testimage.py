import numpy as np
import cv2
import csv
import torch
import torch.optim as optim
import torch.nn as nn
import random
import keras

from torch.utils import data
from torch.utils.data import DataLoader

from imgaug import augmenters as img_aug
from images_processing import image_flip, blur, change_brightness, pan_images, image_zoom, ima_normal
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

name = 'input/IMA/1569783131.92.jpg'
get_images1 = cv2.imread(name)
get_images = cv2.cvtColor(get_images1, cv2.COLOR_BGR2RGB)
ima1, _ = image_flip(get_images, 1)
ima2 = blur(get_images)
ima3 = pan_images(get_images)
ima4 = image_zoom(get_images)
ima5 = ima_normal(ima4)

fig, axes = plt.subplots(3,2, figsize=(45,30))
axes[0][0].imshow(get_images.astype('uint8'))
axes[0][0].set_title("orig")
axes[0][1].imshow(ima1)
axes[0][1].set_title("flip")
axes[1][0].imshow(ima2.astype('uint8'))
axes[1][0].set_title("blur")
axes[1][1].imshow(ima3.astype('uint8'))
axes[1][1].set_title("pan")
axes[2][0].imshow(ima4.astype('uint8'))
axes[2][0].set_title("zoom")
axes[2][1].imshow(ima5.astype('uint8'))
axes[2][1].set_title("normalize")
plt.show()