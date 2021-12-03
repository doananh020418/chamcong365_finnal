from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import glob
import io
import os
import pickle

import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFile
from tqdm import tqdm

import distance as dst
import facenet

ImageFile.LOAD_TRUNCATED_IMAGES = True
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import align.detect_face
from gamma_correction import *


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
FACENET_MODEL_PATH = '../Models/20180402-114759.pb'

tf.Graph().as_default()

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

# Load the model
print('Loading feature extraction model')
facenet.load_model(FACENET_MODEL_PATH)

# Get input and output tensors
images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")

app = Flask(__name__)
socketio = SocketIO(app)
df = pd.DataFrame()


def embedding_faces(id_user=None, new_paths=None):
    global df
    image_size = 160
    np.random.seed(seed=666)
    dataset = facenet.get_dataset(f'../static2/face_data')
    # Check that there are at least one training image per class
    paths, labels = facenet.get_image_paths_and_labels(dataset)
    if id_user == None:
        emb_arrays = []
        for path in tqdm(paths):
            emb_array = []
            images = facenet.load_data1(path, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array.append(sess.run(embeddings, feed_dict=feed_dict)[0])
            emb_array.append(path.split('/')[-2])
            emb_arrays.append(emb_array)
        df = pd.DataFrame(emb_arrays, columns=['emb_array', 'name'])
        df.to_pickle(f'../Models/df_face_data.pkl')

    if id_user != None:
        emb_arrays = []
        for path in tqdm(new_paths):
            emb_array = []
            images = facenet.load_data1(path, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array.append(sess.run(embeddings, feed_dict=feed_dict)[0])
            emb_array.append(path.split('/')[-2])
            emb_arrays.append(emb_array)
        tmp = pd.DataFrame(emb_arrays, columns=['emb_array', 'name'])
        df = pd.concat([df, tmp])
        df.to_pickle('../Models/df_face_data}.pkl')
    print("len df ", len(df))


def base64ToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    img = cv2.cvtColor(np.array(Image.open(io.BytesIO(imgdata))), cv2.COLOR_BGR2RGB)
    return img


def base64ToImageWeb(base64_string):
    base64_string = base64_string.split(',')[-1]
    imgdata = base64.b64decode(base64_string)
    img = cv2.cvtColor(np.array(Image.open(io.BytesIO(imgdata))), cv2.COLOR_BGR2RGB)
    return img


def imageToBase64(image):
    retval, buffer = cv2.imencode('.png', image)
    jpg_as_text = base64.b64encode(buffer)
    image_data = jpg_as_text.decode("utf-8")
    image_data = str(image_data)
    return image_data


def load_faces_data():
    global df
    with open('../Models/df_face_data.pkl', 'rb') as file:
        df = pickle.load(file)
    print(f"Faces data loaded! ")


@app.route('/train')
def retrain():
    global df
    embedding_faces()
    sc = jsonify({'message': 'Done'})
    sc.status_code = 200
    return sc


def recognition_vjpro(frame, scale=0.2, top5=False):
    cpy = frame.copy()
    img = cv2.resize(cpy, (int(cpy.shape[1] * scale), int(cpy.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    bounding_boxes, _ = align.detect_face.detect_face(img, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
    faces_found = bounding_boxes.shape[0]
    # try:
    label = "Undetected"
    if faces_found > 0:
        det = bounding_boxes[:, 0:4]
        bb = np.zeros((faces_found, 4), dtype=np.int32)
        for i in range(faces_found):
            bb[i][0] = det[i][0] / scale
            bb[i][1] = det[i][1] / scale
            bb[i][2] = det[i][2] / scale
            bb[i][3] = det[i][3] / scale

            if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0:
                # if detect_spoofing(cpy):
                if True:
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                    # cv2.imwrite('verify_web.png', scaled)
                    scaled = normalize(scaled)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)

                    def findDistance(row):
                        distance_metric = 'euclidean_l2'
                        img2_representation = row['emb_array']
                        distance = 1000  # initialize very large value
                        if distance_metric == 'cosine':
                            distance = dst.findCosineDistance(emb_array, img2_representation)
                        elif distance_metric == 'euclidean':
                            distance = dst.findEuclideanDistance(emb_array, img2_representation)
                        elif distance_metric == 'euclidean_l2':
                            distance = dst.findEuclideanDistance(dst.l2_normalize(emb_array),
                                                                 dst.l2_normalize(img2_representation))

                        return distance

                    tmp_df = df.copy()
                    tmp_df['distance'] = tmp_df.apply(findDistance, axis=1)
                    tmp_df = tmp_df.sort_values(by=["distance"])
                    if top5 and len(tmp_df) > 5:
                        candidate = tmp_df.iloc[:5]
                        best_distance = candidate['distance'].to_list()
                        label = candidate['name'].to_list()
                    else:
                        candidate = tmp_df.iloc[0]
                        best_distance = candidate['distance']
                        label = candidate['name']
                    del tmp_df
                    print(f'Best distance {best_distance}, name {label}')
    return label


def save_faces(frame, scale=0.2, path=None):
    save_path = '.'
    curr_faces = len(glob.glob(path + '/*'))
    cpy = frame.copy()
    img = cv2.resize(cpy, (int(cpy.shape[1] * scale), int(cpy.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    bounding_boxes, _ = align.detect_face.detect_face(img, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
    faces_found = bounding_boxes.shape[0]
    # try:

    if faces_found > 0:
        det = bounding_boxes[:, 0:4]
        bb = np.zeros((faces_found, 4), dtype=np.int32)
        for i in range(faces_found):
            bb[i][0] = det[i][0] / scale
            bb[i][1] = det[i][1] / scale
            bb[i][2] = det[i][2] / scale
            bb[i][3] = det[i][3] / scale
            # print(bb[i][3] - bb[i][1])
            # print(frame.shape[0])
            # print((bb[i][3] - bb[i][1]) / frame.shape[0])
            if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0:
                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                custom_face = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                         interpolation=cv2.INTER_CUBIC)
                save_path = path + f'/{curr_faces + 1}.png'
                cv2.imwrite(save_path, custom_face)
                print("frame %d saved" % curr_faces)
                curr_faces = curr_faces + 1
    return save_path



