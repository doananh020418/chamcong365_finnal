import glob
import os
import random

import torch
from facenet_pytorch import MTCNN

from faces_augmentation import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    margin=44,
    factor=0.8,
    keep_all=False,
    device=device
)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

INPUT_IMAGE_SIZE = 160

def get_faces(raw_path, id):
    foldername = str(id)
    #os.mkdir(os.path.abspath('../static/face_data'))
    path = os.path.join(os.path.abspath('../static/face_data'), foldername)
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        # base_df = base_df[base_df['name'] != id_reg]
        files = glob.glob(path + '/*')
        for f in files:
            os.remove(f)
    count = 0
    scale = 0.25
    files = glob.glob(raw_path+f'/{id}/*')
    ran = random.randint(2, 6)

    for f in files[0:ran]:
        try:
            img = cv2.imread(f)
            #print(img.shape)
            base_img = img.copy()
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
            boxes, conf = mtcnn.detect(img)
            if conf[0] != None:
                for (x, y, w, h) in boxes:
                    if (w - x) > 100 * scale:
                        x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
                        custom_face = base_img[y:h, x:w]
                        custom_face = cv2.resize(custom_face, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                 interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(path + f'/%d_{id}.png' % (count), custom_face)
                        #print("frame %d saved" % count)
                        count += 1
        except:
            pass
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

dirs = os.listdir(r'C:\Users\doank\PycharmProjects\chamcong365_finnal\raw\img\img')
for dir in dirs:
    get_faces(r'C:\Users\doank\PycharmProjects\chamcong365_finnal\raw\img\img',dir)