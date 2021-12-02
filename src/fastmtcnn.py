from facenet_pytorch import MTCNN
from PIL import Image
import torch
import cv2
import time
import glob
from tqdm.notebook import tqdm
import tensorflow as tf
import numpy as np
tf_version = int(tf.__version__.split(".")[0])
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


mtcnn = MTCNN(
    margin=5,
    factor=0.8,
    keep_all=True,
    device=device
)
def adjust_gamma(image, gamma=1.2):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def hist_eq(img):
    # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # return cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
    image_equalize = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    channels = cv2.split(image_equalize)
    #
    channels[0] = cv2.equalizeHist(channels[0])
    #
    image_equalize = cv2.merge(channels)
    image_equalize = cv2.cvtColor(image_equalize, cv2.COLOR_YUV2BGR)
    image_equalize = cv2.medianBlur(image_equalize,3)
    image_compare = np.hstack((img, image_equalize))
    return image_equalize

def gamma_correction1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(gray)
    gamma = math.log(mid*255)/math.log(mean)
    print(gamma)
    # do gamma correction
    img_gamma1 = np.power(img, gamma).clip(0,255).astype(np.uint8)
    return img_gamma1

def a(img,minimum_brightness = 1):
   cols, rows = img.shape[:2]
   brightness = np.sum(img) / (255 * cols * rows)
   ratio = brightness / minimum_brightness
   if ratio >= 1:
      print("Image already bright enough")
      return img
   return cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)

def register(vid):
    scale = 0.25
    ptime = 0
    cap = cv2.VideoCapture(vid)
    count = 0
    frame_count = 0
    #while frame_count < 10:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        #frame = gamma_correction1(frame)
        #frame = hist_eq(frame)
        #frame = cv2.bilateralFilter(frame, 5, 75, 75)
        frame_count = 0
        if ret:
            if  count % 1 == 0:
                #frame = cv2.resize(frame, (600, 400))

                img = frame.copy()
                # if img.shape[0] < img.shape[1]:
                #     img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
                # else:
                #     img = cv2.resize(img, (480, 640), interpolation=cv2.INTER_AREA)

                img = cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)),interpolation = cv2.INTER_AREA)
                # Here we are going to use the facenet detector
                boxes, conf = mtcnn.detect(img)

                # If there is no confidence that in the frame is a face, don't draw a rectangle around it
                if conf[0] != None:
                    for (x, y, w, h) in boxes:
                        print(w - x)
                        if (w - x) > 100*scale:
                            text = f"{conf[0] * 100:.2f}%"
                            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                            detected_face = frame[int(y):int(h), int(x):int(w)]

                            frame_count = frame_count + 1

                            cv2.putText(frame, text, (x, y - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 1)
                            cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 1)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f'FPS: {str(int(fps))}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX
                    , 1, (0, 0, 255), 1)

        count = count +1

        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

register(0)