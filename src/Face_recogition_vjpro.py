from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
tf.Graph().as_default()
from faces_augmentation import *
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
import align.detect_face
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = '../Models/facemodel.pkl'
FACENET_MODEL_PATH = '../Models/20180402-114759.pb'
images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")



import os

import cv2
import numpy as np
# import torch
from facenet_pytorch import MTCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

scale = 0.5
mtcnn = MTCNN(
    margin=14,
    factor=0.6,
    keep_all=False
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


def reg_frame(frame, frame_count, count, path):
    INPUT_IMAGE_SIZE = 160
    frame = cv2.flip(frame, 1)
    img = frame.copy()
    # base_img = frame.copy()
    # img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
    #                  interpolation=cv2.INTER_AREA)
    bounding_boxes, _ = align.detect_face.detect_face(img, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

    faces_found = bounding_boxes.shape[0]

    try:
        if faces_found > 1:
            cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (255, 255, 255), thickness=1, lineType=2)
        elif faces_found > 0:
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]
                # print(bb[i][3] - bb[i][1])
                # print(frame.shape[0])
                # print((bb[i][3] - bb[i][1]) / frame.shape[0])
                if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                    custom_face = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                             interpolation=cv2.INTER_CUBIC)
                    if count % 1 == 0:
                        # frame = gamma_correction2(frame)
                        cv2.imwrite(path + '/%d.png' % (count), custom_face)

                        process_frame1 = adjust_gamma(custom_face, 0.9)
                        cv2.imwrite(path + '/%d_adjusted1.png' % (count), process_frame1)
                        print("frame %d saved" % count)
                        frame_count = frame_count + 1
                    cv2.rectangle(frame, (10, 10), (90, 50), (255, 67, 67), -10)
                    cv2.putText(frame, str(frame_count), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                1)
        count = count + 1
    except:
        pass


    return frame_count, count, frame
