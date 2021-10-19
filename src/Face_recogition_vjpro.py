from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

scale = 0.5
mtcnn = MTCNN(
    margin=14,
    factor=0.6,
    keep_all=False,
    device=device
)


def adjust(img, minimum_brightness=1.2):
    cols, rows = img.shape[:2]
    brightness = np.sum(img) / (255 * cols * rows)
    ratio = brightness / minimum_brightness
    if ratio >= 1:
        return img
    return cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def reg_frame(frame, frame_count, count, path, mtcnn):
    INPUT_IMAGE_SIZE = 160
    frame = cv2.flip(frame, 1)
    img = frame.copy()
    base_img = frame.copy()
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                     interpolation=cv2.INTER_AREA)
    boxes, conf = mtcnn.detect(img)

    if conf[0] != None:
        for (x, y, w, h) in boxes:
            if (w - x) > 100 * scale:
                text = f"{conf[0] * 100:.2f}%"
                x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
                custom_face = base_img[y:h, x:w]
                custom_face = cv2.resize(custom_face, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                         interpolation=cv2.INTER_CUBIC)
                if count % 1 == 0:
                    # frame = gamma_correction2(frame)
                    cv2.imwrite(path + '/%d.png' % (count), custom_face)

                    process_frame1 = adjust_gamma(custom_face, 0.9)
                    cv2.imwrite(path + '/%d_adjusted1.png' % (count), process_frame1)
                    print("frame %d saved" % count)
                    frame_count = frame_count + 1
                cv2.putText(frame, text, (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 1)
                cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 1)
    count = count + 1

    return frame_count, count, frame
