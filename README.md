### Server

This code performs a server side, which process incoming frame and return a face verification.

### Setup
```buildoutcfg
pip install -r requirements.txt
```



### Server
Chạy file socket_io_flask_api.py để kết nối với app qua socketio
```buildoutcfg
python socket_io_flask_api.py
```

### Demo và test

```buildoutcfg
python app.py
```
### Cấu trúc thư mục
```shell
│   README.md
│   requirements.txt
│
│
├───Models
│       20180402-114759.pb # model facenet
│       facemodel.pkl # model phân loại
│       model-20180402-114759.ckpt-275.data-00000-of-00001
│       model-20180402-114759.ckpt-275.index
│       model-20180402-114759.meta
│
├───src # folder chứa các file code
│   │   a
│   │   align_dataset_mtcnn.py
│   │   app.py # chạy demo web
│   │   classifier.py # file này để train model phân loại nè
│   │   download_and_extract.py
│   │   facenet.py # file chính, chứa các hàm cần thiết
│   │   Face_recogition_vjpro.py # file này xử lý backgound cho socket io nè
│   │   face_rec_cam.py # vẫn là chạy demo nhưng là trên local
│   │   lfw.py
│   │   socket_io_flask_api.py # socket io để kết nối với app á
│   │   train_softmax.py
│   │   train_tripletloss.py
│   │   validate_on_lfw.py
│   │   __init__.py
│   │
│   ├───align
│   │   │   a
│   │   │   det1.npy
│   │   │   det2.npy
│   │   │   det3.npy
│   │   │   detect_face.py # cái này để phát hiện khuôn mặt á
│   │   │
│   │   └───__pycache__
│   │           detect_face.cpython-37.pyc
│   │
│   ├───models
│   │       a
│   │       dummy.py
│   │       inception_resnet_v1.py
│   │       inception_resnet_v2.py
│   │       squeezenet.py
│   │       __init__.py
│   │
│   ├───templates
│   │       index.html
│   │       index1.html
│   │
│   └───__pycache__
│           classifier.cpython-37.pyc
│           facenet.cpython-37.pyc
│
└───static # trong đây để lưu data khuôn mặt nhaaaaaa

```

### Feature
-Chấm công: kết nối qua 'face_verify',đầu vào là chuỗi frame và id, trả về kết quả xác minh khuôn mặt và frame chứa khuôn mặt chấm công

```shell
@socketio.on('face_verify', namespace='/stream')
```
---
Kết quả trả về

`result` là `true` nếu ảnh đc xác minh là đúng và ngược lại


```shell
socketio.emit('verify', {'result': True,'image_data': frame}, to=users[user_id])
```

`image_data` Ảnh trả về sau khi xử lý (thêm bondingbox bao quanh khuôn mặt)
(Khuyến khích không cần kết nối hàm này để tăng tốc độ stream)

---
-Đăng ký: kết nối qua event handler: 'face_register',đầu vào là chuỗi frame và id, trả về trạng thái đăng ký và frame hiển thị khuôn mặt được phát hiện
```shell
@socketio.on('face_register', namespace='/reg')
```
Trả về:
`reg_stt` lưu trạng thái đăng ký (`true` tức là đã đăng ký xong và ngược lại)

`reg_data` Ảnh trả về sau khi xử lý(thêm bondingbox bao quanh khuôn mặt)
(Khuyến khích không cần kết nối hàm này để tăng tốc độ stream)

```shell
socketio.emit("registered", {'reg_stt': True, 'reg_data': frame}, to=users[user_id])
```
## Note

Ảnh truyền và nhận đều ở dạng base64