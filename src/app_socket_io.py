from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import glob
import io
import os
import pickle
from tqdm import tqdm
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
import pandas as pd
import align.detect_face
import facenet
from faces_augmentation import *
from gamma_correction import *
import shutil
app = Flask(__name__)
import distance as dst
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = False

socketio = SocketIO(app)
print('loading')
frame = 0

ALLOWED_EXTENSIONS = {'jpg'}
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
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
import time
df = {}

company_list = []
reg_hist = {}
count = {}
reg_stt = {}
reg_frame_count = {}

def train(id_company, id_user=None):
    global df
    image_size = 160
    

    np.random.seed(seed=666)

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
            
            emb_array = []
            images = facenet.load_data1(path, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array.append(sess.run(embeddings, feed_dict=feed_dict)[0])
            emb_array.append(path.split('/')[-2])
            emb_arrays.append(emb_array)
        df[id_company] = pd.DataFrame(emb_arrays, columns=['emb_array', 'name'])
        # df[id_company].to_csv('hjhj.csv')
        df[id_company].to_pickle(f'../Models/df_{id_company}.pkl')
        classifier_filename_exp = os.path.expanduser(f'../Models/{id_company}.pkl')
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [2000, 1000, 500, 300, 100, 10, 1, 0.1, 0.01, 0.001],
                      'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
        print('Training classifier')
        
        
        
        #model = SVC(kernel='linear', C=0.1, gamma=100, probability=True)
        x_train = df[id_company].emb_array.apply(lambda x: list(map(float, x)))
        y_train = df[id_company].name
        y_train = le.fit_transform(y_train)
        #print(x_train)
        
        if len(x_train) <100:
             model = SVC(kernel='linear', C=0.1, gamma=100, probability=True)
        else:
             model = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', C=0.1, gamma=100, probability=True), 
             max_samples = 1.0,n_estimators = 1))
        tic = time.time()
        #model.fit(x_train.to_list(), y_train)
        print("function last", time.time() - tic)
        # model = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        # model.fit(emb_array, labels)
        # print(model.best_estimator_)
        class_names = [cls.name.replace('_', ' ') for cls in dataset]
        employees_list[id_company] = class_names

        # Saving classifier model
        #with open(classifier_filename_exp, 'wb') as outfile:
        #    pickle.dump((model, class_names), outfile)
        return model, class_names

    if id_user != None:

        if len(df[id_company])==0:
            path1 = glob.glob(f'../static/{id_company}/base/*') 
        else:
            path1 = []
        path2 = glob.glob(f'../static/{id_company}/{id_user}/*')
        new_paths = path1+path2

        if id_user in df[id_company]['name'].values:
            df[id_company] = df[id_company][df[id_company]['name']!=id_user]
            
        # else:
        #     employees_list[id_company].append(id_user)

        #new_paths = glob.glob(f'../static/{id_company}/{id_user}/*')
        new_label = []

        #print("new labels:", new_label)
        #print("new path", new_paths)
        # Load the model

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
        df[id_company] = pd.concat([df[id_company],tmp])
        df[id_company].to_pickle(f'../Models/df_{id_company}.pkl')
        classifier_filename_exp = os.path.expanduser(f'../Models/{id_company}.pkl')
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [2000, 1000, 500, 300, 100, 10, 1, 0.1, 0.01, 0.001],
                      'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

        x_train = df[id_company].emb_array.apply(lambda x: list(map(float, x)))
        y_train = df[id_company].name
        y_train = le.fit_transform(y_train)
        print(len(x_train))

        #model = SVC(kernel='linear', C=0.1, gamma=100, probability=True)
        if len(x_train) <100:
             model = SVC(kernel='linear', C=0.1, gamma=100, probability=True)
        else:
             model = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', C=0.1, gamma=100, probability=True), 
             max_samples = 1.0,n_estimators = 1))
        #model.fit(x_train.to_list(), y_train)
        # model = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        # model.fit(emb_array, labels)
        # print(model.best_estimator_)
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        #with open(classifier_filename_exp, 'wb') as outfile:
        #    pickle.dump((model, class_names), outfile)
        #print('Saved classifier model to file "%s"' % classifier_filename_exp)

        return model, class_names



def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def load_trained_model(company_id):
    global model
    global class_names
    # Load The Custom Classifier
    CLASSIFIER_PATH = f'../Models/{company_id}.pkl'
    with open(CLASSIFIER_PATH, 'rb') as file:
        model[company_id], class_names[company_id] = pickle.load(file)
    df_path = f'../Models/df_{company_id}.pkl'
    with open(df_path, 'rb') as file:
        df[company_id] = pickle.load(file)
    print(f"Custom Classifier, Successfully loaded {len(model)} company models")


#companies = os.listdir('../static')

for company in ['58']:
    df[company] = pd.DataFrame(columns = ['emb_array','name'])
    employees_list[company] = []
    company_list.append(company)
    #model[company], class_names[company] = train(company)
    load_trained_model(company)


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


@app.route('/verify_web', methods=['GET', 'POST'])
def verify_web():
    global model
    global class_names
    delta = 1
    user_id_verify_web = '.'
    company_id_verify_web = '.'
    best_class_probabilities = []
    name = 'unknown'
    if True:
        sc = jsonify({'company_id': company_id_verify_web, 'user_id': user_id_verify_web, 'pred': name,
                      'message': True if name == user_id_verify_web else False})
        contents = request.json
        for content in contents:
            company_id_verify_web = content['company_id']
            user_id_verify_web = content['user_id']
            image = content['image']
        frame = base64ToImageWeb(image)
        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        name = 'Unknown'
        faces_found = bounding_boxes.shape[0]
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
                    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
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
                    tmp_df = df[company_id_verify_web]#[df[company_id_verify_web]['name']==user_id_verify_web]
                    tmp_df['distance'] = tmp_df.apply(findDistance, axis=1)
                    tmp_df = tmp_df.sort_values(by=["distance"])

                    candidate = tmp_df.iloc[0]
                    best_distance = candidate['distance']
                    label = candidate['name']
                    print(f'Best distance {best_distance}, name {label}')
                    if best_distance <= 0.8:
                        name = label
                else:
                    name = "Unknown"
            sc = jsonify({'company_id': company_id_verify_web, 'user_id': user_id_verify_web,
                          'message': True if name == user_id_verify_web else False})
            sc.status_code = 200

    return sc


@app.route('/register_web', methods=['GET', 'POST'])
def register_web():
    global model
    global class_names
    user_id = request.args.get('user_id')
    company_id = request.args.get('company_id')
    foldername = str(company_id)
    path = os.path.join(os.path.abspath('../static'), foldername)
    if not os.path.exists(path):
        os.mkdir(path)
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))
        os.mkdir(path)
    elif not os.path.exists(os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))):
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))
        os.mkdir(path)
    else:
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id))
        files = glob.glob(path + '/*')
        for f in files:
            os.remove(f)

    contents = request.json
    scale = 0.25
    count = 0
    total = len(contents)
    for content in contents:
        image = content['image']
        frame = base64ToImageWeb(image)
        frame = Image.open(frame).convert('RGB')
        frame = np.array(frame)
        img = frame.copy()
        # base_img = frame.copy()
        # img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
        #                  interpolation=cv2.INTER_AREA)
        bounding_boxes, _ = align.detect_face.detect_face(img, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

        faces_found = bounding_boxes.shape[0]

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

                    # frame = gamma_correction2(frame)
                    cv2.imwrite(path + '/%d.png' % (count), custom_face)
                    process_frame1 = adjust_gamma(custom_face, 0.9)
                    cv2.imwrite(path + '/%d_adjusted1.png' % (count), process_frame1)
                    print("frame %d saved" % count)
        count = count + 1

    model[company_id], class_names[company_id] = train(company_id,user_id)
    sc = jsonify({'company_id': company_id, 'user_id': user_id,
                  'message': f"Up load register image completed! {count}/{total} images uploaded!"})
    sc.status_code = 200
    return sc


if __name__ == "__main__":
    print('[INFO] Starting server at http://localhost:5001')
    socketio.run(app=app,host='0.0.0.0', port=5002)
