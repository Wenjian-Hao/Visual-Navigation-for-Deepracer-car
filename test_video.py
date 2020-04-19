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
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from imgaug import augmenters as img_aug
from images_processing import image_flip, blur, change_brightness, pan_images, image_zoom, ima_normal
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
white_output = 'process_output.mp4'
clip1 = VideoFileClip("input/output.mpg")
white_clip = clip1.fl_image(ima_normal)
white_clip.write_videofile(white_output, audio=False)