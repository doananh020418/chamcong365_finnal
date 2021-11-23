import cv2
import numpy as np
import math

# read image
# img = cv2.imread(r'C:\Users\doank\PycharmProjects\Face-Verification\deepface\hjhjhj.jpg')
# img = cv2.resize(img,(480,640))

# METHOD 1: RGB

# convert img to gray
def normalize(img):
    norm_img = np.zeros((160,160))
    nor_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    return nor_img



# METHOD 2: HSV (or other color spaces)
def gamma_correction2(img):

# convert img to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(val)
    gamma = math.log(mid*255)/math.log(mean)

    # do gamma correction on value channel
    val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

    # combine new value channel with original hue and sat channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
    norm_img = np.zeros((160,160))
    img_gamma2 = cv2.normalize(img_gamma2, norm_img, 0, 255, cv2.NORM_MINMAX)
    return img_gamma2

