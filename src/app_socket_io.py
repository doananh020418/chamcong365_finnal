from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import glob
import io
import os
import pickle
from typing import Protocol
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from tqdm import tqdm
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from sklearn.svm import SVC
import pandas as pd
import align.detect_face
import facenet
from faces_augmentation import *
from gamma_correction import *
from Face_Anti_Spoofing.test_realtime_v2 import detect_spoofing
import time

app = Flask(__name__)
import shutil

import warnings

warnings.filterwarnings("ignore")

app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = False
import distance as dst

socketio = SocketIO(app)
print('loading')
frame = 0
from sklearn.decomposition import PCA

ALLOWED_EXTENSIONS = {'jpg'}
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.8]
FACTOR = 0.709
# IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
# CLASSIFIER_PATH = '../Models/facemodel.pkl'
FACENET_MODEL_PATH = '../Models/20180402-114759.pb'

tf.compat.v1.Graph().as_default()

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

model = {}
class_names = {}

employees_list = {}
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df = {}
n_components = 50
company_list = []
reg_hist = {}
count = {}
reg_stt = {}
reg_frame_count = {}
pca = {}
np.random.seed(seed=666)


def train(id_company, id_user=None):
    global df
    image_size = 160
    global pca
    pca[id_company] = PCA(n_components=n_components, whiten=True)

    dataset = facenet.get_dataset(f'../static/{id_company}')

    # Check that there are at least one training image per class
    for cls in dataset:
        assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

    paths, labels = facenet.get_image_paths_and_labels(dataset)
    if id_user == None:

        # print('Number of classes: %d' % len(dataset))
        # print('Number of images:',(paths))

        if not id_company in company_list:
            company_list.append(id_company)
            employees_list[id_company] = []
            df[id_company] = pd.DataFrame(columns=['emb_array', 'name'])

        emb_arrays = []
        for path in tqdm(paths):
            # print(path)
            id = path.split('/')[-2]
            if id not in employees_list[id_company]:
                # employees_list[id_company].append(id)
                reg_hist[id_company][id] = False
                reg_frame_count[id_company][id] = 0
                count[id_company][id] = 0
            else:
                reg_hist[id_company][id] = True
            emb_array = []
            images = facenet.load_data1(path, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array.append(sess.run(embeddings, feed_dict=feed_dict)[0])
            emb_array.append(path.split('/')[-2])
            emb_arrays.append(emb_array)
        df[id_company] = pd.DataFrame(emb_arrays, columns=['emb_array', 'name'])
        # df[id_company].to_csv('hjhj.csv')
        df[id_company].to_pickle(f'../Models/df_{id_company}.pkl', protocol=4)
        if len(df[id_company]) > 150:
            df[id_company] = df[id_company][df[id_company]['name'] != 'base']
        classifier_filename_exp = os.path.expanduser(f'../Models/{id_company}.pkl')
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [2000, 1000, 500, 300, 100, 10, 1, 0.1, 0.01, 0.001],
                      'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
        # print('Training classifier')

        # x_train = df[id_company].emb_array.apply(lambda x: list(map(float, x))).to_list()
        # pca[id_company].fit(x_train)
        # x_train = pca[id_company].transform(x_train)
        # y_train = df[id_company].name
        # y_train = le.fit_transform(y_train)

        # model = SVC(kernel='linear', probability=True)

        # tic = time.time()
        # model.fit(x_train, y_train)
        # print("function last", time.time() - tic)

        class_names = [cls.name.replace('_', ' ') for cls in dataset]
        employees_list[id_company] = class_names
        for class_name in list(set(class_names)):
            reg_hist[id_company][id_user] = True

        # Saving classifier model
        # with open(classifier_filename_exp, 'wb') as outfile:
        #     pickle.dump((model, class_names), outfile)
        # return model, class_names

    if id_user != None:

        if len(df[id_company]) == 0:
            path1 = glob.glob(f'../static/{id_company}/base/*')
        else:
            path1 = []
        path2 = glob.glob(f'../static/{id_company}/{id_user}/*')
        new_paths = path1 + path2

        if id_user in df[id_company]['name'].values:
            df[id_company] = df[id_company][df[id_company]['name'] != id_user]

        emb_arrays = []
        for path in tqdm(new_paths):
            # print(path)
            emb_array = []
            images = facenet.load_data1(path, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array.append(sess.run(embeddings, feed_dict=feed_dict)[0])
            emb_array.append(path.split('/')[-2])
            emb_arrays.append(emb_array)
        tmp = pd.DataFrame(emb_arrays, columns=['emb_array', 'name'])
        df[id_company] = pd.concat([df[id_company], tmp])
        if len(df[id_company]) > 150:
            df[id_company] = df[id_company][df[id_company]['name'] != 'base']
        df[id_company].to_pickle(f'../Models/df_{id_company}.pkl')

        classifier_filename_exp = os.path.expanduser(f'../Models/{id_company}.pkl')
        # param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [2000, 1000, 500, 300, 100, 10, 1, 0.1, 0.01, 0.001],
        #               'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

        # x_train = df[id_company].emb_array.apply(lambda x: list(map(float, x))).to_list()
        # y_train = df[id_company].name
        # y_train = le.fit_transform(y_train)
        # print(len(x_train))
        # pca[id_company].fit(x_train)
        # x_train = pca[id_company].transform(x_train)

        # model = SVC(kernel='linear', probability=True)

        # tic = time.time()
        # model.fit(x_train, y_train)
        # print("function last", time.time() - tic)

        # class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # # Saving classifier model
        # with open(classifier_filename_exp, 'wb') as outfile:
        #     pickle.dump((model, class_names), outfile)
        # print('Saved classifier model to file "%s"' % classifier_filename_exp)

        # return model, class_names


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def load_trained_model(id_company):
    # global model
    # global class_names
    # Load The Custom Classifier
    for id in os.listdir(f'../static/{id_company}'):
        reg_hist[id_company][id] = True
        reg_frame_count[id_company][id] = 0
        count[id_company][id] = 0
        employees_list[id_company].append(id)
        # print(id)
    df_path = f'../Models/df_{id_company}.pkl'
    with open(df_path, 'rb') as file:
        df[id_company] = pickle.load(file)
        # pca[id_company].fit(df[id_company].emb_array.apply(lambda x: list(map(float, x))).to_list())
    print(f"Custom Classifier, Successfully loaded {len(df)} company df")


companies = os.listdir('../static')
for company in companies:
    # if company!='57':
    if True:
        company_list.append(company)
        # pca[company] = PCA(n_components=n_components, whiten=True)
        df[company] = pd.DataFrame(columns=['emb_array', 'name'])
        reg_hist[company] = {}
        employees_list[company] = []
        reg_stt[company] = {}
        reg_frame_count[company] = {}
        count[company] = {}

        # train(company)
        load_trained_model(company)
    else:
        company_list.append(company)
        # pca[company] = PCA(n_components=n_components, whiten=True)
        df[company] = pd.DataFrame(columns=['emb_array', 'name'])
        reg_hist[company] = {}
        employees_list[company] = []
        reg_stt[company] = {}
        reg_frame_count[company] = {}
        count[company] = {}

        train(company)
        # load_trained_model(company)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def base64ToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    img = cv2.cvtColor(np.array(Image.open(io.BytesIO(imgdata))), cv2.COLOR_BGR2RGB)
    return img


def base64ToImageWeb(base64_string):
    base64_string = base64_string.split(',')[-1]
    imgdata = base64.b64decode(base64_string)
    img = cv2.cvtColor(np.array(Image.open(io.BytesIO(imgdata))), cv2.COLOR_BGR2RGB)
    return img


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
    image_equalize = cv2.medianBlur(image_equalize, 3)
    image_compare = np.hstack((img, image_equalize))
    return image_equalize


def imageToBase64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    image_data = jpg_as_text.decode("utf-8")
    image_data = str(image_data)
    return image_data


# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def adjust(img, minimum_brightness=1):
    cols, rows = img.shape[:2]
    brightness = np.sum(img) / (255 * cols * rows)
    ratio = brightness / minimum_brightness
    if ratio >= 1:
        # print("Image already bright enough")
        return img
    return cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)


users = {}
buff = {}
company = []


@socketio.on('face_verify_app', namespace='/stream')
def face_verify(input, user_id, company_id, isAndroid):
    global df

    global model
    frame = base64ToImage(input)
    if isAndroid:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    frame = normalize(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.stack((frame,) * 3, axis=-1)
    cv2.imwrite('app_frame.png', frame)
    bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
    name = 'Undetected'
    faces_found = bounding_boxes.shape[0]
    try:
        if faces_found > 0:
            #     pass
            # elif faces_found > 0:
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(1):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]

                if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                    if True:
                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                            interpolation=cv2.INTER_CUBIC)

                        cv2.imwrite('app.png', scaled)
                        # scaled = normalize(scaled)
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

                        tmp_df = df[company_id][df[company_id]['name'] == user_id]
                        tmp_df['distance'] = tmp_df.apply(findDistance, axis=1)
                        tmp_df = tmp_df.sort_values(by=["distance"])

                        candidate = tmp_df.iloc[0]
                        best_distance = candidate['distance']
                        label = candidate['name']
                        print(f'Best distance {best_distance}, name {label}')
                        if best_distance <= 0.75:
                            name = label
                    else:
                        print("Unknown")

    except:
        pass
    frame = imageToBase64(frame)
    # frame = None
    print("Name-app: ", name)
    socketio.emit('verify_app', {'result': name == user_id, 'image_data': frame}, namespace='/stream', to=request.sid)


def reg_frame(frame, frame_count, count, path):
    face_detect = False
    # THRESHOLD = [0.7, 0.8, 0.8]
    scale = 1
    INPUT_IMAGE_SIZE = 160
    frame = cv2.flip(frame, 1)
    img = frame.copy()
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                     interpolation=cv2.INTER_AREA)
    bounding_boxes, _ = align.detect_face.detect_face(img, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

    faces_found = bounding_boxes.shape[0]

    try:
        if faces_found > 1:
            return frame_count, count, face_detect, True
        elif faces_found > 0:
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                bb[i][0] = det[i][0] / scale
                bb[i][1] = det[i][1] / scale
                bb[i][2] = det[i][2] / scale
                bb[i][3] = det[i][3] / scale

                if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.15:
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]

                    custom_face = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                             interpolation=cv2.INTER_CUBIC)
                    custom_face = normalize(custom_face)

                    cv2.imwrite(path + '/%d_org.png' % (count), custom_face)
                    print("frame %d saved" % count)
                    face_detect = True
                    frame_count = frame_count + 1

        count = count + 1
    except:
        pass

    return frame_count, count, face_detect, False


@socketio.on('face_register', namespace='/reg')
def reg(input, user_id, company_id, isAndroid):  # add new employees
    global model
    global class_names
    face_reg = None
    foldername = str(company_id)
    path = os.path.join(os.path.abspath('../static'), foldername)
    files = []
    if not os.path.exists(path):
        # employees_list[company_id]=[]

        os.mkdir(path)
        # base = os.path.join(os.path.abspath(f'../static/{foldername}'), 'base')
        # shutil.copytree('../2',base)
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))
        os.mkdir(path)
    elif not os.path.exists(os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))):
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))
        # reg_hist[company_id][user_id] = True
        os.mkdir(path)
    else:
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))
        # if reg_hist[company_id][user_id] == True:
        #     reg_hist[company_id][user_id] = False
        #     files = glob.glob(path + "/*")
        # for f in files:
        #     os.remove(f)

    if not company_id in company_list:
        reg_hist[company_id] = {}
        df[company_id] = pd.DataFrame(columns=['emb_array', 'name'])
        company_list.append(company_id)
        employees_list[company_id] = []
        count[company_id] = {}
        users[company_id] = {}
        reg_stt[company_id] = {}
        reg_frame_count[company_id] = {}

    if user_id not in employees_list[company_id]:
        reg_hist[company_id][user_id] = False
        employees_list[company_id].append(user_id)
        reg_stt[company_id][user_id] = False
        reg_frame_count[company_id][user_id] = 0
        count[company_id][user_id] = 0
    frame = base64ToImage(input)
    if isAndroid:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    reg_frame_count[company_id][user_id], count[company_id][user_id], face_detect, multi_faces = reg_frame(frame,
                                                                                                           reg_frame_count[
                                                                                                               company_id][
                                                                                                               user_id],
                                                                                                           count[
                                                                                                               company_id][
                                                                                                               user_id],
                                                                                                           path
                                                                                                           )

    # frame = imageToBase64(frame)
    if reg_frame_count[company_id][user_id] == 10:
        count[company_id][user_id] = 0
        reg_frame_count[company_id][user_id] = 0
        # for f in files:
        #     os.remove(f)
        # return status: True mean register completed
        # return processed frame which show the regions including face
        files = glob.glob(path + '/*')
        try:
            face_reg = imageToBase64(cv2.imread(files[-1]))
        except:
            face_reg = None
        socketio.emit("registered",
                      {'reg_stt': True, 'face_detect': face_detect, 'multi_faces': multi_faces, 'face_ret': face_reg},
                      namespace='/reg', to=request.sid)
        train(company_id, user_id)
        reg_hist[company_id][user_id] = True
        print('Done!')
    else:
        socketio.emit("registered",
                      {'reg_stt': False, 'face_detect': face_detect, 'multi_faces': multi_faces, 'face_ret': face_reg},
                      namespace='/reg', to=request.sid)


@app.route('/verify_web', methods=['GET', 'POST'])
def verify_web():
    tic = time.time()
    global model
    global class_names
    global df
    multi_faces = False
    user_id_verify_web = '.'
    company_id_verify_web = '.'
    name = "Undetected"
    message = "Undetected"
    if True:
        contents = request.json
        for content in contents:
            company_id_verify_web = content['company_id']
            user_id_verify_web = content['user_id']
            image = content['image']
        print('User request: \t', user_id_verify_web)
        if company_id_verify_web not in company_list:
            sc = jsonify({'company_id': company_id_verify_web, 'user_id': user_id_verify_web, 'pred': name,
                          'message': "Unregisted"})
            sc.status_code = 200
            return sc
        if user_id_verify_web not in employees_list[company_id_verify_web]:
            message = 'Unregisted'
            sc = jsonify({'company_id': company_id_verify_web, 'user_id': user_id_verify_web, 'pred': name,
                          'message': message})
            sc.status_code = 200
            return sc
        scale = 0.2

        frame = base64ToImageWeb(image)

        frame = normalize(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
        # frame = clahe.apply(frame)
        tic = time.time()
        frame = np.stack((frame,) * 3, axis=-1)
        frame_resized = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
        cv2.imwrite('frame_verify_app.png', frame)
        cpy = frame.copy()
        toc = time.time()
        print('resize ', toc - tic)
        bounding_boxes, _ = align.detect_face.detect_face(frame_resized, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        toe = time.time()
        # print('detect faces ',toe - toc)
        faces_found = bounding_boxes.shape[0]
        # try:
        if faces_found > 0:
            #     multi_faces = True
            # elif faces_found > 0:
            multi_faces = False
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(1):
                bb[i][0] = det[i][0] / scale
                bb[i][1] = det[i][1] / scale
                bb[i][2] = det[i][2] / scale
                bb[i][3] = det[i][3] / scale

                if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.15:
                    toe = time.time()
                    if True:
                        # if detect_spoofing(cpy):
                        try:
                            dbrr = time.time()
                            # print('detect_spoofing ',dbrr - toe)

                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                interpolation=cv2.INTER_CUBIC)
                            cv2.imwrite('verify_web.png', scaled)
                            # scaled = normalize(scaled)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            cc = time.time()

                            # print('get emb',cc-dbrr)
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

                            tmp_df = df[company_id_verify_web][df[company_id_verify_web]['name'] == user_id_verify_web]
                            tmp_df['distance'] = tmp_df.apply(findDistance, axis=1)
                            tmp_df = tmp_df.sort_values(by=["distance"])
                            # print('cal dist',time.time()-cc)
                            candidate = tmp_df.iloc[0]
                            best_distance = candidate['distance']
                            label = candidate['name']
                            print(f'Best distance: \t {best_distance}')
                            if best_distance <= 0.85:
                                name = label
                            else:
                                name = 'Unknown'
                            # print(time.time()-tic)
                        except:
                            pass
                    else:
                        print("Spoof")
    # print('detect_spoofing ',dbrr - toe)
    if name == user_id_verify_web:
        message = True
    # elif user_id_verify_web not in employees_list[company_id_verify_web]:
    #     message = 'Unregisted'
    else:
        message = False
    print(f'Message:\t {message}')
    sc = jsonify({'company_id': company_id_verify_web, 'user_id': user_id_verify_web, 'pred': name,
                  'message': message})
    sc.status_code = 200

    print('Total time: \t', time.time() - tic)
    print('-' * 100)
    return sc


@app.route('/verify_web_company', methods=['GET', 'POST'])
def verify_web_company():
    # global model
    # global class_names
    tic = time.time()
    global df
    scale = 0.25
    multi_faces = False
    company_id_verify_web = '.'
    best_distance = 1000
    name = "Undetected"
    if True:
        contents = request.json
        for content in contents:
            company_id_verify_web = content['company_id']
            image = content['image']
        if company_id_verify_web not in company_list:
            sc = jsonify({'user_id': 'Unregisted'})
            sc.status_code = 200
            return sc
        # company_id_verify_web = '3183'
        frame = base64ToImageWeb(image)

        # frame = gamma_correction2(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.stack((frame,) * 3, axis=-1)
        # print('func last: ',time.time()-tic)
        cv2.imwrite('company_web.png', frame)
        frame_resized = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
        bounding_boxes, _ = align.detect_face.detect_face(frame_resized, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        faces_found = bounding_boxes.shape[0]
        # try:
        if faces_found > 0:
            #    multi_faces = True
            # elif faces_found > 0:
            multi_faces = False
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(1):
                bb[i][0] = det[i][0] / scale
                bb[i][1] = det[i][1] / scale
                bb[i][2] = det[i][2] / scale
                bb[i][3] = det[i][3] / scale
                # print((bb[i][3] - bb[i][1])/ frame.shape[0])
                if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                    scaled = normalize(scaled)
                    cv2.imwrite('company_face_img.png', scaled)
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

                    tmp_df = df[company_id_verify_web]
                    tmp_df['distance'] = tmp_df.apply(findDistance, axis=1)
                    tmp_df = tmp_df.sort_values(by=["distance"])

                    candidate = tmp_df.iloc[0]
                    best_distance = candidate['distance']
                    label = candidate['name']

                    if best_distance <= 0.75:
                        name = label
                    else:
                        name = "Unknown"
    sc = jsonify({'user_id': name})
    sc.status_code = 200
    print('total ', time.time() - tic)
    print(f'Best distance {best_distance}, name {name}')
    print('-' * 100)
    return sc


@app.route('/register_web', methods=['GET', 'POST'])
def register_web():
    global df
    # global model
    # global class_names
    THRESHOLD = [0.6, 0.7, 0.8]
    user_id = request.args.get('user_id')
    print("Register web, id: ", user_id)
    company_id = request.args.get('company_id')
    foldername = str(company_id)
    path = os.path.join(os.path.abspath('../static'), foldername)
    files = []
    if not os.path.exists(path):
        os.mkdir(path)
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))
        os.mkdir(path)
    elif not os.path.exists(os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))):
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))
        os.mkdir(path)
    else:
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))

    if not company_id in company_list:
        df[company_id] = pd.DataFrame(columns=['emb_array', 'name'])
        company_list.append(company_id)
        employees_list[company_id] = []
    if user_id not in employees_list[company_id]:
        employees_list[company_id].append(user_id)

    contents = request.json
    scale = 0.25
    count = 0
    total = len(contents)
    for content in contents:
        try:
            image = content['image']
            frame = base64ToImageWeb(image)

            img = frame.copy()
            # base_img = frame.copy()
            # img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
            #                  interpolation=cv2.INTER_AREA)
            bounding_boxes, _ = align.detect_face.detect_face(img, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

            faces_found = bounding_boxes.shape[0]

            if faces_found > 1:
                pass
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
                        # cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                        custom_face = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                 interpolation=cv2.INTER_CUBIC)

                        custom_face = normalize(custom_face)

                        cv2.imwrite(path + '/%d_org.png' % (count), custom_face)

                        print("frame %d saved" % count)
                count = count + 1
        except:
            pass

    # model[company_id], class_names[company_id] =
    # if count > 5:
    #     for f in files:
    #         os.remove(f)
    train(company_id, user_id)
    sc = jsonify({'company_id': company_id, 'user_id': user_id,
                  'message': True if count > 5 else False})
    sc.status_code = 200

    return sc


@app.route('/sync_app', methods=['GET', 'POST'])
def sync_app():
    global df
    # global model
    # global class_names
    THRESHOLD = [0.6, 0.7, 0.8]
    user_id = request.args.get('user_id')
    print("Register web, id: ", user_id)
    company_id = request.args.get('company_id')
    foldername = str(company_id)
    path = os.path.join(os.path.abspath('../static'), foldername)
    files = []
    if not os.path.exists(path):
        os.mkdir(path)
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))
        os.mkdir(path)
    elif not os.path.exists(os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))):
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))
        os.mkdir(path)
    else:
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))

    if not company_id in company_list:
        df[company_id] = pd.DataFrame(columns=['emb_array', 'name'])
        company_list.append(company_id)
        employees_list[company_id] = []
    if user_id not in employees_list[company_id]:
        employees_list[company_id].append(user_id)

    contents = request.json
    scale = 0.25
    count = 0
    total = len(contents)
    if len(contents) < 1:
        sc = jsonify({'company_id': company_id, 'user_id': user_id,
                      'message': False})
        sc.status_code = 200

    for content in contents:
        try:
            image = content['image']
            frame = base64ToImageWeb(image)

            frame_resized = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                                       interpolation=cv2.INTER_AREA)
            bounding_boxes, _ = align.detect_face.detect_face(frame_resized, MINSIZE, pnet, rnet, onet, THRESHOLD,
                                                              FACTOR)
            faces_found = bounding_boxes.shape[0]
            # try:
            if faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)
                for i in range(1):
                    bb[i][0] = det[i][0] / scale
                    bb[i][1] = det[i][1] / scale
                    bb[i][2] = det[i][2] / scale
                    bb[i][3] = det[i][3] / scale
                    # print(bb[i][3] - bb[i][1])
                    # print(frame.shape[0])
                    # print((bb[i][3] - bb[i][1]) / frame.shape[0])
                    if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        # cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                        custom_face = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                 interpolation=cv2.INTER_CUBIC)

                        custom_face = normalize(custom_face)

                        cv2.imwrite(path + f'/%d_{user_id}.png' % (count), custom_face)

                        print("frame %d saved" % count)
                count = count + 1
        except:
            pass

    train(company_id, user_id)
    sc = jsonify({'company_id': company_id, 'user_id': user_id,
                  'message': True if count > 0 else False})
    sc.status_code = 200

    return sc


if __name__ == "__main__":
    print('[INFO] Starting server at http://localhost:5001')
    socketio.run(app=app, host='0.0.0.0', port=5001)
