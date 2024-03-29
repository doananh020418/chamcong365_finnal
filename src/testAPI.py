import requests
import cv2
import base64
import glob

def imageToBase64(image):
    retval, buffer = cv2.imencode('.png', image)
    jpg_as_text = base64.b64encode(buffer)
    image_data = jpg_as_text.decode("utf-8")
    image_data = str(image_data)
    return image_data

def find_user(id):
    files = glob.glob(f'../raw/img/img/{id}/*')
    for f in files:
        #print(f)
        img = cv2.imread(f)
        res = requests.post(url='http://103.138.113.112:5000/find_user',json=[
            {
                "image": f"data:image/jpeg;base64,{imageToBase64(img)}"
            }
        ])
        print(res.json()['user_id'])
def add_faces(id):
    files = glob.glob(f'../raw/img/img/{id}/*')
    json_files = []
    for f in files:
        # print(f)
        img = cv2.imread(f)
        json_files.append({
                "image": f"data:image/jpeg;base64,{imageToBase64(img)}"
            })
    res = requests.post(url=f'http://103.138.113.112:5000/add_faces?user_id={id}', json=json_files)
    print(res.json()['message'])

def sync(company_id,id):
    files = glob.glob(f'../raw/img/img/{id}/*')
    json_files = []
    for f in files:
        # print(f)
        img = cv2.imread(f)
        json_files.append({
            "image": f"data:image/jpeg;base64,{imageToBase64(img)}"
        })
    res = requests.post(url=f'http://103.138.113.112:5001/sync_app?company_id={company_id}&user_id={id}', json=json_files)
    print(res.json()['message'])

def verify(company_id,id):
    files = glob.glob(f'../raw/img/img/{id}/*')
    for f in files:
        # print(f)
        img = cv2.imread(f)
        res = requests.post(url='http://103.138.113.112:5001/verify_web', json=[
            {
                "company_id":company_id,
                "user_id":id,
                "image": f"data:image/jpeg;base64,{imageToBase64(img)}"
            }
        ])
        print(res.json()['pred'])
add_faces('123')