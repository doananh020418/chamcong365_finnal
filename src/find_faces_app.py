from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from find_faces import *

app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/train')
def retrain():
    global df
    embedding_faces()
    sc = jsonify({'message': 'Done'})
    sc.status_code = 200
    return sc


@app.route('/find_user', methods=['GET', 'POST'])
def find_user():
    global df
    label = "Undetected"
    image = ''
    if True:
        contents = request.json
        for content in contents:
            image = content['image']
        frame = base64ToImageWeb(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.stack((frame,) * 3, axis=-1)
        label = recognition_vjpro(frame, top5=False)

    sc = jsonify({'user_id': label})
    sc.status_code = 200
    return sc


@app.route('/add_faces', methods=['GET', 'POST'])
def add_faces():
    user_id = request.args.get('user_id')

    foldername = 'face_data'
    path = os.path.join(os.path.abspath('../static2'), foldername)
    if not os.path.exists(path):
        os.mkdir(path)
        path = os.path.join(os.path.abspath(f'../static2/{foldername}'), str(user_id))
        os.mkdir(path)
    elif not os.path.exists(os.path.join(os.path.abspath(f'../static2/{foldername}'), str(user_id))):
        path = os.path.join(os.path.abspath(f'../static2/{foldername}'), str(user_id))
        os.mkdir(path)
    else:
        path = os.path.join(os.path.abspath(f'../static2/{foldername}'), str(user_id))
    contents = request.json
    scale = 0.2
    new_paths = []
    for content in contents:
        image = content['image']
        frame = base64ToImageWeb(image)
        frame = normalize(frame)
        save_path = save_faces(frame, scale=scale, path=path)
        if save_path != '.':
            new_paths.append(save_path)

    embedding_faces(user_id, new_paths)
    sc = jsonify({'message': 'Done'})
    sc.status_code = 200
    return sc


if __name__ == '__main__':
    # embedding_faces()
    load_faces_data()
    print('[INFO] Starting server at http://localhost:5000')
    socketio.run(app=app, debug=False, host='0.0.0.0', port=5000)