## use this file to augment training images
import cv2
import numpy as np
from imgaug import augmenters as ima_aug
import random

############################
# Utility Functions
############################
#-------------------------
## Load images
#-------------------------
def read_ima(ima_name): 
    name = 'input/IMA/' + ima_name + '.jpg'
    get_images = cv2.imread(name)
    get_images = cv2.cvtColor(get_images, cv2.COLOR_BGR2RGB)
    return get_images

#-------------------------
## images augmentation
#-------------------------
def image_flip(get_images, field_angle):
    get_images = cv2.flip(get_images, 1)
    field_angle = field_angle * -1.0
    return get_images, field_angle

def blur(get_images):
    kernel_size = random.randint(1,5)
    get_images = cv2.blur(get_images, (kernel_size, kernel_size))
    return get_images

def change_brightness(get_images):
    brightness = ima_aug.Multiply((0.7, 1.3))
    get_images = brightness.augment_image(get_images)
    return get_images

def pan_images(get_images):
    pan = ima_aug.Affine(translate_percent = {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
    get_images = pan.augment_image(get_images)
    return get_images

def image_zoom(get_images):
    zoom = ima_aug.Affine(scale=(1, 1.3))
    get_images = zoom.augment_image(get_images)
    return get_images

def traingimages_augment(get_images, field_angle, field_throttle):
    # if np.random.rand() < 0.25:
    #     get_images = pan_images(get_images)
    field_throttle = field_throttle
    if np.random.rand() < 0.25:
        get_imagess = image_zoom(get_images)
    if np.random.rand() < 0.25:
        get_images = blur(get_images)
    if np.random.rand() < 0.25:
        get_images = change_brightness(get_images)
        get_images, field_angle = image_flip(get_images, field_angle) 
    return get_images, field_angle, field_throttle

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    try:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    except IndexError:
                    line = []
def hough_lines(img, rho=1, theta=np.pi/180, threshold=10, min_line_len=8, max_line_gap=4):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def ima_normal(get_images):
    get_images = cv2.cvtColor(get_images, cv2.COLOR_RGB2YUV)
    get_images = cv2.GaussianBlur(get_images, (3,3), 0)
    lower_blue = np.array([80,80,80])
    upper_blue = np.array([200, 255, 255])
    linecanny = cv2.inRange(get_images, lower_blue, upper_blue)
    linecanny = cv2.Canny(linecanny, 200, 400)
    lines = hough_lines(linecanny)
    get_images = weighted_img(get_images, lines)
    height, _, _ = get_images.shape
    get_images = get_images[int(height/5):int(height/2), :, :]
    get_images = cv2.resize(get_images, (180, 120))
    # get_images = (get_images / 255)
    return get_images

def expandlabel(x):
    return x * 10