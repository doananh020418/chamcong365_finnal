import cv2
import numpy as np
import math

# read image
# img = cv2.imread(r'C:\Users\doank\PycharmProjects\Face-Verification\deepface\hjhjhj.jpg')
# img = cv2.resize(img,(480,640))

# METHOD 1: RGB

# convert img to gray
def gamma_correction1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(gray)
    gamma = math.log(mid*255)/math.log(mean)

    # do gamma correction
    img_gamma1 = np.power(img, gamma).clip(0,255).astype(np.uint8)
    return img_gamma1


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
    return img_gamma2

