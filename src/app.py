from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tqdm import tqdm
import base64
import glob
import io
import os
import pickle
import time
import pandas as pd
import imutils
import tensorflow as tf
from PIL import Image, ImageFile
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC,SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
import facenet
from faces_augmentation import *
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True
from flask import Flask, Response, request, jsonify
from flask import render_template
from imutils.video import VideoStream
import distance as dst
import align.detect_face
from gamma_correction import *
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


MINSIZE = 20
THRESHOLD = [0.7, 0.8, 0.8]
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

employees_list = {}

le = LabelEncoder()

df = {}

company_list = []
n_components = 50
pca={}
np.random.seed(seed=666)

def train(id_company, id_user=None):
    global df
    global pca
    pca[id_company] = PCA(n_components=n_components, whiten=True)
    image_size = 160
    # if not id_company in company_list:
    #     company_list.append(id_company)
    #     employees_list[id_company] = []
    #     df[id_company] = pd.DataFrame(columns=['emb_array', 'name'])



    dataset = facenet.get_dataset(f'../static/{id_company}')

    # Check that there are at least one training image per class
    for cls in dataset:
        assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

    paths, labels = facenet.get_image_paths_and_labels(dataset)
    if id_user == None:

        # print('Number of classes: %d' % len(dataset))
        # print('Number of images:',(paths))

        print("Get face emb vector \n")
        tic = time.time()
        emb_arrays = []
        for path in tqdm(paths):
            id=(path.split('\\')[-2])
            if id not in employees_list[id_company]:
                employees_list[id_company].append(id)
            emb_array = []
            images = facenet.load_data1(path, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array.append(sess.run(embeddings, feed_dict=feed_dict)[0])
            emb_array.append(path.split('/')[-1].split('\\')[-2])
            emb_arrays.append(emb_array)
        df[id_company] = pd.DataFrame(emb_arrays, columns=['emb_array', 'name'])
        df[id_company].to_pickle(f'../Models/df_{id_company}.pkl')
        if len(df[id_company]) > 100:
            df[id_company] = df[id_company][df[id_company]['name']!='base']
        toc = time.time()
        print(f"Face embbeding for {id_company} last {toc-tic} seconds")
        classifier_filename_exp = os.path.expanduser(f'../Models/{id_company}.pkl')
        # param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [5000, 2000, 1000, 500, 100],
        #               'kernel': ['linear']}
        print('Training classifier')
        x_train = df[id_company].emb_array.apply(lambda x: list(map(float, x))).to_list()
        pca[id_company].fit(x_train)
        x_train = pca[id_company].transform(x_train)
        y_train = df[id_company].name
        y_train = le.fit_transform(y_train)
        #print(x_train)
        n_estimators = 1

        model = SVC(kernel='linear',probability=False)
        #model = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', C=0.1, gamma=100, probability=False), max_samples=1.0,n_estimators=1))

        # model = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        model.fit(x_train, y_train)
        # print(model.best_estimator_)
        class_names = [cls.name.replace('_', ' ') for cls in dataset]
        employees_list[id_company] = class_names

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)

        toe = time.time()
        print(f"Training for {id_company} last {toe - toc} seconds")
        return model, class_names

    if id_user != None:
        tic = time.time()
        if len(df[id_company])==0:
            path1 =glob.glob(f'../static/{id_company}/base/*')
        else:
            path1 = []
        path2 = glob.glob(f'../static/{id_company}/{id_user}/*')
        new_paths = path1+path2
        print("org_df",len(df[id_company]))
        print(type(id_user))
        if id_user in df[id_company]['name'].values:
            df[id_company] = df[id_company][df[id_company]['name']!=str(id_user)]
            print("edit_df", len(df[id_company]))

        # else:
        #     employees_list[id_company].append(id_user)
        emb_arrays = []
        for path in tqdm(new_paths):
            #print(path)
            emb_array = []
            images = facenet.load_data1(path, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array.append(sess.run(embeddings, feed_dict=feed_dict)[0])
            emb_array.append(path.split('/')[-1].split('\\')[-2])
            emb_arrays.append(emb_array)
        tmp = pd.DataFrame(emb_arrays, columns=['emb_array', 'name'])
        df[id_company] = pd.concat([df[id_company],tmp])
        df[id_company].to_pickle(f'../Models/df_{id_company}.pkl')
        toc = time.time()
        print(f"Face embbeding for {id_company},user {id_user} last {toc-tic} seconds")

        classifier_filename_exp = os.path.expanduser(f'../Models/{id_company}.pkl')
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [2000, 1000, 500, 300, 100, 10, 1, 0.1, 0.01, 0.001],
                      'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

        x_train = df[id_company].emb_array.apply(lambda x: list(map(float, x))).to_list()
        y_train = df[id_company].name
        y_train = le.fit_transform(y_train)
        print('Total sample: ',len(x_train))
        pca[id_company].fit(x_train)
        x_train = pca[id_company].transform(x_train)

        model = SVC(kernel='linear', C=0.1, gamma=100, probability=True)
        model.fit(x_train, y_train)
        # model = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        # print(model.best_estimator_)
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

        return model, class_names


def base64ToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    img = cv2.cvtColor(np.array(Image.open(io.BytesIO(imgdata))), cv2.COLOR_BGR2RGB)
    return img


def readb64(base64_string):
    sbuf = io.StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


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


@app.route('/')
def index():
    return "OK!"


model = {}
class_names = {}
threshold = dst.findThreshold('facenet', 'euclidean_l2')

def load_trained_model(id_company):
    global model
    global class_names
    global df
    # Load The Custom Classifier
    pca[id_company] = PCA(n_components=n_components, whiten=True)
    CLASSIFIER_PATH = f'../Models/{id_company}.pkl'
    with open(CLASSIFIER_PATH, 'rb') as file:
        model[id_company], class_names[id_company] = pickle.load(file)
    df_path = f'../Models/df_{id_company}.pkl'
    with open(df_path, 'rb') as file:
        df[id_company] = pickle.load(file)
        pca[id_company].fit(df[id_company].emb_array.apply(lambda x: list(map(float, x))).to_list())
    print(f"Custom Classifier, Successfully loaded {id_company} company models")

#load_trained_model('HHP')
# load_trained_model('HHP_org')
#companies = os.listdir('../static')
companies = ['HHP_x3']
for company in companies:
    df[company] = pd.DataFrame(columns = ['emb_array','name'])
    employees_list[company] = []
    company_list.append(company)
    pca[company] = PCA(n_components=n_components, whiten=True)
    model[company], class_names[company] = train(company)
    #load_trained_model(company)

def gen_frames():
    global company_id_stream
    global model
    global class_names
    # print("total number of model",len(model))
    print(company_id_stream)
    name = "Unknown"
    cap = VideoStream(src=0).start()
    # facenet.load_model(FACENET_MODEL_PATH)
    print('total',len(employees_list[company_id_stream]))
    while (True):
        frame = cap.read()
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)
        # frame = gamma_correction2(frame)

        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

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
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                            interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)

                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)
                        emb_array = pca[company_id_stream].transform(emb_array)
                        # print(" lenght emb_array: ",len(emb_array))
                        predictions = model[company_id_stream].predict_proba(emb_array)
                        # print("predictions",predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[company_id_stream][best_class_indices[0]]
                        print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                        if best_class_probabilities > 0.85:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][1] - 20

                            name = class_names[company_id_stream][best_class_indices[0]]
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)),
                                        (text_x, text_y + 17),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            # person_detected[best_name] += 1
                        else:
                            "Unknown"

        except:
            pass

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/stream')
def streamimg():
    global company_id_stream
    global user_id_stream
    company_id_stream = request.args.get('company_id')
    # print(company_id_stream)
    # user_id_stream = request.args.get('user_id')
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def register():
    global company_id_reg
    global user_id_reg
    global model
    global class_names
    cap = VideoStream(src=0).start()
    if not company_id_reg in company_list:
        df[company_id_reg] = pd.DataFrame(columns=['emb_array','name'])
        company_list.append(company_id_reg)
        employees_list[company_id_reg] = []
    if user_id_reg not in employees_list[company_id_reg]:
        employees_list[company_id_reg].append(user_id_reg)


    foldername = str(company_id_reg)
    path = os.path.join(os.path.abspath('../static'), foldername)
    if not os.path.exists(path):
        os.mkdir(path)
        base = os.path.join(os.path.abspath(f'../static/{foldername}'), 'base')
        shutil.copytree(r'../7749/1', base)
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id_reg))
        os.mkdir(path)

    elif not os.path.exists(os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id_reg))):
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id_reg))
        os.mkdir(path)
    else:
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id_reg))
        files = glob.glob(path + '/*')
        for f in files:
            os.remove(f)
    print("reg_path", path)
    count = 0
    frame_count = 0
    while frame_count < 10:
        frame = cap.read()
        frame = imutils.resize(frame, width=600)

        frame = cv2.flip(frame, 1)
        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

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
                    custom_face = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                             interpolation=cv2.INTER_CUBIC)
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                    if count % 5 == 0:
                        # frame = gamma_correction2(frame)
                        cv2.imwrite(path + '/%d.png' % count, custom_face)
                        process_frame1 = adjust_gamma(custom_face, 0.9)
                        cv2.imwrite(path + '/%d_adjusted1.png' % count, process_frame1)
                        # process_frame2 = rotate_img(custom_face, angle=5)
                        # cv2.imwrite(path + '/%d_adjusted2.png' % (count), process_frame2)
                        process_frame3 = adjust_gamma(custom_face, 1.2)
                        cv2.imwrite(path + '/%d_adjusted3.png' % count, process_frame3)
                        process_frame4 = distort(custom_face)
                        cv2.imwrite(path + '/%d_adjusted4.png' % count, process_frame4)
                        process_frame5 = cv2.flip(custom_face, 1)
                        cv2.imwrite(path + '/%d_adjusted5.png' % count, process_frame5)
                        print("frame %d saved" % count)
                        frame_count = frame_count + 1
        count = count + 1
        cv2.rectangle(frame, (10, 10), (90, 50), (255, 67, 67), -10)
        cv2.putText(frame, str(frame_count), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    1)

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except:
            continue
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    tic = time.time()
    model[company_id_reg], class_names[company_id_reg] = train(company_id_reg, user_id_reg)
    toc = time.time()
    print('function last {} secs'.format(int(toc - tic)))


@app.route('/register')
def reg():
    return Response(register(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reg')
def stream():
    global company_id_reg
    global user_id_reg
    company_id_reg = request.args.get('company_id')
    user_id_reg = request.args.get('user_id')
    return render_template('index1.html')


@app.route('/verify_web_2', methods=['GET', 'POST'])
def verify_web2():
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
                    tmp_df = df[company_id_verify_web][df[company_id_verify_web]['name']==user_id_verify_web]
                    tmp_df['distance'] = tmp_df.apply(findDistance, axis=1)
                    tmp_df = tmp_df.sort_values(by=["distance"])

                    candidate = tmp_df.iloc[0]
                    best_distance = candidate['distance']
                    label = candidate['name']
                    print('Best distance', best_distance)
                    if best_distance <= threshold * delta:
                        name = label
                else:
                    name = "Unknown"
            sc = jsonify({'company_id': company_id_verify_web, 'user_id': user_id_verify_web,'pred':name,
                          'message': True if name == user_id_verify_web else False})
            sc.status_code = 200

    return sc
count_ = 0

@app.route('/verify_web', methods=['GET', 'POST'])
def verify_web():
    global model
    global class_names
    global pca
    global count_
    user_id_verify_web = '.'
    company_id_verify_web = '.'
    best_class_probabilities = []
    if True:
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
                if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                    # cv2.imwrite(f"../test2/{count_}.png",scaled)
                    # count_ = count_ +1

                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    emb_array = pca[company_id_verify_web].transform(emb_array)
                    predictions = model[company_id_verify_web].predict(emb_array)
                    # print(predictions)
                    name = class_names[company_id_verify_web][predictions[0]]
                    print(f"Name: {name}")
                                # if True:
                                #     cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                #     text_x = bb[i][0]
                                #     text_y = bb[i][1] - 20
                                #
                                #     name = best_name
                                #     cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #                 1, (255, 255, 255), thickness=1, lineType=2)
                                #     cv2.putText(frame, str(round(best_class_probabilities[0], 3)),
                                #                 (text_x, text_y + 17),
                                #                 cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #                 1, (255, 255, 255), thickness=1, lineType=2)
                                #     # person_detected[best_name] += 1
                                # else:
                                #     name = "Unknown"
                    sc = jsonify({'company_id': company_id_verify_web, 'user_id': user_id_verify_web,'pred':name,
                                  'message': True if name == user_id_verify_web else False})
                    sc.status_code = 200
                    # except:
                    #     sc = jsonify({'company_id': company_id_verify_web, 'user_id': user_id_verify_web,'pred':name,
                    #                   'message': True if name == user_id_verify_web else False})
                    #     sc.status_code = 404

    return sc


@app.route('/register_web', methods=['GET', 'POST'])
def register_web():
    global model
    global class_names
    user_id_reg_web = request.args.get('user_id')
    company_id_reg_web = request.args.get('company_id')

    foldername = str(company_id_reg_web)
    path = os.path.join(os.path.abspath('../static'), foldername)
    if not os.path.exists(path):
        os.mkdir(path)
        base = os.path.join(os.path.abspath(f'../static/{foldername}'), 'base')
        os.mkdir(base)
        img = cv2.imread("../black_image.jpg")
        cv2.imwrite(f'{base}/black_image.jpg', img)
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id_reg_web))
        os.mkdir(path)

    elif not os.path.exists(os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id_reg_web))):
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id_reg_web))
        os.mkdir(path)
    else:
        path = os.path.join(os.path.abspath(f'../static/{foldername}'), str(user_id_reg_web))
        files = glob.glob(path + '/*')
        for f in files:
            os.remove(f)

    contents = request.json
    scale = 0.5
    count = 0
    reg_frame = 0
    total = len(contents)
    for content in contents:
        image = content['image']
        frame = base64ToImageWeb(image)
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
                    # cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                    custom_face = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                             interpolation=cv2.INTER_CUBIC)
                    print(custom_face)
                    cv2.imwrite(path + '/%d.png' % count, custom_face)
                    process_frame1 = adjust_gamma(custom_face, 0.9)
                    cv2.imwrite(path + '/%d_adjusted1.png' % count, process_frame1)
                    print("frame %d saved" % count)
                    reg_frame = reg_frame + 1
        count = count + 1

    model[company_id_reg_web], class_names[company_id_reg_web] = train(company_id_reg_web)
    sc = jsonify({'company_id': company_id_reg_web, 'user_id': user_id_reg_web,
                  'message': f"Up load register image completed! {reg_frame}/{total} images uploaded!"})
    sc.status_code = 200
    # except:
    #     sc = jsonify({'company_id': company_id_reg_web, 'user_id': user_id_reg_web,
    #                   'message': f"Up load register image unsuccessfull! {count}/{total} images uploaded!"})
    #     sc.status_code = 404
    return sc


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5002)
