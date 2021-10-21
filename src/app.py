from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from sklearn.svm import SVC
import facenet
from faces_augmentation import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
from flask import Flask, Response, request, jsonify
from flask import render_template
from imutils.video import VideoStream

import align.detect_face
from gamma_correction import *
from sklearn.preprocessing import LabelEncoder


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
CLASSIFIER_PATH = '../Models/facemodel.pkl'
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
embedding_list = {}

df = pd.DataFrame()
le = LabelEncoder()

df = {}

def train(id_company,id_user = None):
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
        id = []
        for p in paths:
            id.append(p.split('\\')[-2])


        emb_arrays = []
        for path in paths:
            #print(path)
            emb_array=[]
            images = facenet.load_data1(path, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array.append(sess.run(embeddings, feed_dict=feed_dict)[0])
            emb_array.append(path.split('/')[-1].split('\\')[-2])
            emb_arrays.append(emb_array)
        df[id_company] = pd.DataFrame(emb_arrays,columns = ['emb_array','name'])
        #df[id_company].to_csv('hjhj.csv')

        classifier_filename_exp = os.path.expanduser(f'../Models/{id_company}.pkl')
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [2000,1000,500,300,100, 10, 1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid','linear']}
        print('Training classifier')
        model = SVC(kernel='linear', C=1, gamma=100, probability=True)
        x_train = df[id_company].emb_array.apply(lambda x: list(map(float,x)))
        y_train = df[id_company].name
        y_train = le.fit_transform(y_train)

        print(x_train)
        model.fit(x_train.to_list(), y_train)
        # model = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        # model.fit(emb_array, labels)
        # print(model.best_estimator_)
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        return model, class_names

    if id_user != None:
        new_path = []
        new_label = []
        for i,p in enumerate(paths):
            if p.split('//')[-2] in employees_list[id_company]:
                embedding_list[id_company] = np.delete(embedding_list[id_company],i)

        print("new labels:" ,new_label)
        print("new path", new_path)
        # Load the model

        nrof_images = len(new_path)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 1000))
        print(nrof_batches_per_epoch)
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * 1000
            end_index = min((i + 1) * 1000, nrof_images)
            paths_batch = new_path[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

        classifier_filename_exp = os.path.expanduser(f'../Models/{id_company}.pkl')
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [2000, 1000, 500, 300, 100, 10, 1, 0.1, 0.01, 0.001],
                      'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

        embedding_list[id_company] = embedding_list[id_company] + emb_array

        model = SVC(kernel='linear', C=1, gamma=100, probability=True)
        model.fit(embedding_list[id_company], labels)
        # model = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        # model.fit(emb_array, labels)
        # print(model.best_estimator_)
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)
        employees_list[id_company] = employees_list[id_company] + new_label[0]
        embedding_list[id_company] = embedding_list[id_company] + emb_array
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


def load_trained_model(company_id):
    global model
    global class_names
    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model[company_id], class_names[company_id] = pickle.load(file)
    print(f"Custom Classifier, Successfully loaded {company_id} company models")


@app.route('/train')
def retrain():
    global model
    global class_names
    model, class_names = train()
    sc = jsonify({'message': 'Done'})
    sc.status_code = 200
    return sc

#train('test')
companies = os.listdir('../static')
for company in companies:
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
                        # print(" lenght emb_array: ",len(emb_array))
                        predictions = model[company_id_stream].predict_proba(emb_array)
                        # print("predictions",predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[company_id_stream][best_class_indices[0]]
                        print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                        if best_class_probabilities > 0.6:
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
                            name = "Unknown"

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

    foldername = str(company_id_reg)
    path = os.path.join(os.path.abspath('../static'), foldername)
    if not os.path.exists(path):
        os.mkdir(path)
        base = os.path.join(os.path.abspath(f'../static/{foldername}'), 'base')
        os.mkdir(base)
        img = cv2.imread("../black_image.jpg")
        cv2.imwrite(f'{base}/black_image.jpg', img)
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
    scale = 0.25
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
                        cv2.imwrite(path + '/%d.png' % (count), custom_face)
                        process_frame1 = adjust_gamma(custom_face, 0.9)
                        cv2.imwrite(path + '/%d_adjusted1.png' % (count), process_frame1)
                        process_frame2 = rotate_img(custom_face, angle=5)
                        cv2.imwrite(path + '/%d_adjusted2.png' % (count), process_frame2)
                        process_frame3 = adjust_gamma(custom_face, 1.2)
                        cv2.imwrite(path + '/%d_adjusted3.png' % (count), process_frame3)
                        process_frame4 = distort(custom_face)
                        cv2.imwrite(path + '/%d_adjusted4.png' % (count), process_frame4)
                        process_frame5 = cv2.flip(custom_face, 1)
                        cv2.imwrite(path + '/%d_adjusted5.png' % (count), process_frame5)
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
    model[company_id_reg], class_names[company_id_reg] = train(company_id_reg,user_id_reg)
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


@app.route('/verify_web', methods=['GET', 'POST'])
def verify_web():
    global model
    global class_names
    name = ''
    user_id_verify_web = '.'
    company_id_verify_web = '.'
    best_class_probabilities = []
    if True:

        # image = request.files['image']
        # if 'image' not in request.files:
        #     resp = jsonify({'message': 'No file part in the request'})
        #     resp.status_code = 400
        #     return resp
        contents = request.json
        for content in contents:
            company_id_verify_web = content['company_id']
            user_id_verify_web = content['user_id']
            image = content['image']
        frame = base64ToImageWeb(image)
        # print(frame)
        # frame = gamma_correction2(frame)
        # Convert RGB to BGR
        # print(frame.shape)
        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        name = ''
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

                        predictions = model[company_id_verify_web].predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[company_id_verify_web][best_class_indices[0]]
                        print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                        if best_class_probabilities > 0.4:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][1] - 20

                            name = class_names[company_id_verify_web][best_class_indices[0]]
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)),
                                        (text_x, text_y + 17),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            # person_detected[best_name] += 1
                        else:
                            name = "Unknown"
            sc = jsonify({'company_id': company_id_verify_web, 'user_id': user_id_verify_web,
                          'message': True if name == user_id_verify_web else False,
                          "conf": best_class_probabilities[0]})
            sc.status_code = 200
        except:
            sc = jsonify({'company_id': company_id_verify_web, 'user_id': user_id_verify_web,
                          'message': True if name == user_id_verify_web else False })
            sc.status_code = 404

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
                    #cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                    custom_face = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                             interpolation=cv2.INTER_CUBIC)
                    print(custom_face)
                    cv2.imwrite(path + '/%d.png' % (count), custom_face)
                    process_frame1 = adjust_gamma(custom_face, 0.9)
                    cv2.imwrite(path + '/%d_adjusted1.png' % (count), process_frame1)
                    print("frame %d saved" % count)
                    reg_frame = reg_frame+1
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
