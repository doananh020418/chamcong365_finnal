import glob
import os


import align.detect_face

from faces_augmentation import *
import tensorflow as tf
ALLOWED_EXTENSIONS = set(['jpg'])
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

# Get input and output tensors
pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


INPUT_IMAGE_SIZE = 224


def get_faces(raw_path,company_id, user_id_reg):
    foldername = str(company_id)
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
    count = 0
    scale = 0.25
    files = glob.glob(raw_path + '/*.png')
    for f in files:
        img = cv2.imread(f)
        print(img.shape)
        base_img = img.copy()
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
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
                if (bb[i][3] - bb[i][1]) / img.shape[0] > 0.25:
                    cropped = img[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]

                    custom_face = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                             interpolation=cv2.INTER_CUBIC)
                    process_frame1 = adjust_gamma(custom_face, 0.9)
                    cv2.imwrite(path + '/%d_adjusted1.png' % (count), process_frame1)
                    cv2.imwrite(path + '/%d_adjusted2.png' % (count), custom_face)
                    process_frame3 = adjust_gamma(custom_face, 1.2)
                    cv2.imwrite(path + '/%d_adjusted3.png' % (count), process_frame3)
                    process_frame4 = distort(custom_face)
                    cv2.imwrite(path + '/%d_adjusted4.png' % (count), process_frame4)
                    process_frame5 = cv2.flip(custom_face, 1)
                    cv2.imwrite(path + '/%d_adjusted5.png' % (count), process_frame5)
                    count = count + 1

for name in ['phuong','thoa','trang']:

    get_faces(rf'C:\Users\doank\PycharmProjects\Face-Verification-2\raw\{name}','test',f'{name}')

def gen_faces(path):
    files = glob.glob(path + '/*.png')
    count = 0
    for f in files:
        custom_face = cv2.imread(f)
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
        count = count + 1


#gen_faces(r'C:\Users\doank\PycharmProjects\Face-Verification-2\static\7749\thao')
