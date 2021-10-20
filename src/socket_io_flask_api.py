from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import glob
import io
import pickle

import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from sklearn.svm import SVC
import os

import align.detect_face
import facenet
#from Face_recogition_vjpro import *
from gamma_correction import *
from faces_augmentation import *
app = Flask(__name__)

app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = False

socketio = SocketIO(app)
print('loading')
# use 0 for web camera
frame = 0
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from facenet_pytorch import MTCNN

# mtcnn = MTCNN(
#     margin=48,
#     factor=0.8,
#     keep_all=False,
#     device=device
# )
ALLOWED_EXTENSIONS = set(['jpg'])
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
# CLASSIFIER_PATH = '../Models/facemodel.pkl'
FACENET_MODEL_PATH = '../Models/20180402-114759.pb'

tf.Graph().as_default()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

# Load the model
print('Loading feature extraction model')
facenet.load_model(FACENET_MODEL_PATH)

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")

model = {}
class_names = {}


def train(id_company):
    image_size = 160

    np.random.seed(seed=666)

    dataset = facenet.get_dataset(f'../static/{id_company}')

    # Check that there are at least one training image per class
    for cls in dataset:
        assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

    paths, labels = facenet.get_image_paths_and_labels(dataset)

    # print('Number of classes: %d' % len(dataset))
    # print('Number of images: %d' % len(paths))
    #
    # # Load the model
    # print('Loading feature extraction model')
    # facenet.load_model('../Models/20180402-114759.pb')

    # Get input and output tensors
    # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    # embedding_size = embeddings.get_shape()[1]

    # Run forward pass to calculate embeddings
    print('Calculating features for images')
    nrof_images = len(paths)
    nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 1000))
    emb_array = np.zeros((nrof_images, embedding_size))
    for i in range(nrof_batches_per_epoch):
        start_index = i * 1000
        end_index = min((i + 1) * 1000, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, image_size)
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

    classifier_filename_exp = os.path.expanduser(f'../Models/{id_company}.pkl')

    if True:
        # Train classifier
        print('Training classifier')
        model = SVC(kernel='linear', C=5, gamma=10, probability=True)
        model.fit(emb_array, labels)

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        # print('Saved classifier model to file "%s"' % classifier_filename_exp)
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
    print(f"Custom Classifier, Successfully loaded {len(model)} company models")


companies = os.listdir('../static')
for company in companies:
    model[company], class_names[company] = train(company)
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


@socketio.on('face_verify', namespace='/stream')
def face_verify(input, user_id, company_id):
    if user_id not in buff[company_id]:
        buff[company_id].append(user_id)
    # input = input.split(",")[1]
    users[company_id][user_id] = request.sid
    global model
    # df = base_df[base_df['name'] == user_id.strip()]
    frame = base64ToImage(input)
    frame = gamma_correction2(frame)
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

                    predictions = model[company_id].predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[
                        np.arange(len(best_class_indices)), best_class_indices]
                    best_name = class_names[company_id][best_class_indices[0]]
                    print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                    if best_class_probabilities > 0.3:
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                        text_x = bb[i][0]
                        text_y = bb[i][1] - 20

                        name = class_names[company_id][best_class_indices[0]]
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
    frame = imageToBase64(frame)
    if name == user_id:
        # return verify result
        socketio.emit('verify', {'result': True, 'image_data': frame}, to=users[company_id][user_id])
        # frame_count[user_id] = 0
    else:
        socketio.emit('verify', {'result': False, 'image_data': frame}, to=users[company_id][user_id])
    # return frame verified
    # socketio.emit("processed", {'image_data': frame}, to=users[user_id])


count = {}
reg_stt = {}
reg_frame_count = {}


def reg_frame(frame, frame_count, count, path, mtcnn):
    scale = 0.25
    INPUT_IMAGE_SIZE = 160
    frame = cv2.flip(frame, 1)
    img = frame.copy()
    base_img = frame.copy()
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                     interpolation=cv2.INTER_AREA)
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

        count = count + 1
    except:
        pass

    return frame_count, count, frame

@socketio.on('face_register', namespace='/reg')
def reg(input, user_id, company_id):  # add new employees
    global model
    global mtcnn
    global class_names
    if user_id not in buff[company_id]:
        buff[company_id].append(user_id)
        reg_stt[company_id][user_id] = False
        reg_frame_count[company_id][user_id] = 0
        count[company_id][user_id] = 0
    input = input.split(",")[1]
    users[company_id][user_id] = request.sid

    foldername = str(company_id)
    path = os.path.join(os.path.abspath('../static'), foldername)
    if not os.path.exists(path):
        os.mkdir(path)
        base = os.path.join(os.path.abspath(f'../static/{foldername}'), 'base')
        os.mkdir(base)
        img = cv2.imread("../black_image.jpg")
        cv2.imwrite(f'{base}/black_image.jpg', img)
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

    frame = base64ToImage(input)
    reg_frame_count[company_id][user_id], count[company_id][user_id], frame = reg_frame(frame,
                                                                                        reg_frame_count[company_id][
                                                                                            user_id],
                                                                                        count[company_id][user_id],
                                                                                        path
                                                                                        )
    frame = imageToBase64(frame)
    if reg_frame_count[company_id][user_id] == 10:
        count[company_id][user_id] = 0
        reg_frame_count[company_id][user_id] = 0
        model[company_id], class_names[company_id] = train(company_id)
        # return status: True mean register completed
        # return processed frame which show the regions including face
        socketio.emit("registered", {'reg_stt': True, 'reg_data': frame}, to=users[company_id][user_id])
    else:
        socketio.emit("registered", {'reg_stt': False, 'reg_data': frame}, to=users[company_id][user_id])


@app.route('/verify_web', methods=['GET', 'POST'])
def verify_web():
    global model
    global class_names
    user_id_verify_web = '.'
    company_id_verify_web = '.'
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
        print(company_id_verify_web)
        print(user_id_verify_web)
        frame = base64ToImage(image)
        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        name = ''
        faces_found = bounding_boxes.shape[0]
        try:
            if faces_found > 1:
                cv2.putText(frame, "Too much faces found! ", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (255, 0, 0), thickness=1, lineType=2)
            elif faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)
                for i in range(faces_found):
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]
                    print(bb[i][3] - bb[i][1])
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

                        if best_class_probabilities > 0.6:
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
        except:
            pass
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
    path = os.path.join(os.path.abspath('static'), foldername)
    if not os.path.exists(path):
        os.mkdir(path)
        path = os.path.join(os.path.abspath(f'static/{foldername}'), str(user_id))
        os.mkdir(path)
    elif not os.path.exists(os.path.join(os.path.abspath(f'static/{foldername}'), str(user_id))):
        path = os.path.join(os.path.abspath(f'static/{foldername}'), str(user_id))
        os.mkdir(path)
    else:
        path = os.path.join(os.path.abspath(f'static/{foldername}'), str(user_id))
        files = glob.glob(path + '/*')
        for f in files:
            os.remove(f)

    contents = request.json
    scale = 0.25
    count = 0
    total = len(contents)
    for content in contents:
        image = content['image']
        frame = base64ToImage(image)
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

    model[company_id], class_names[company_id] = train(company_id)
    sc = jsonify({'company_id': company_id, 'user_id': user_id,
                  'message': f"Up load register image completed! {count}/{total} images uploaded!"})
    sc.status_code = 200
    return sc


if __name__ == "__main__":
    print('[INFO] Starting server at http://localhost:5001')
    socketio.run(app=app, host='0.0.0.0', port=5001)
