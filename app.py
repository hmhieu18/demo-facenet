import os
import gdown
from flask import request, Response
import jsonpickle
from flask_ngrok import run_with_ngrok
import time

from flask import Flask, request, render_template, send_from_directory

from PIL import Image
from google.colab.patches import cv2_imshow
import numpy as np
import cv2
from keras_facenet import FaceNet


def FindPersonInVideo(videoPath, imagePath, threshold=0.2):
    # get face embedding
    embedder = FaceNet()
    inputFaceImage = cv2.imread(imagePath)
    inputFaceEmbedding = embedder.extract(inputFaceImage, threshold=0.95)
    inputFaceEmbedding = inputFaceEmbedding[0]

    cap = cv2.VideoCapture(videoPath)
    dest_size = (160, 160)
    i = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    closeFaces = []
    while(True):
        # Capture ảnh từ video
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % fps == 0:
            pixels = frame

            # Detect khuôn mặt
            results = embedder.extract(pixels, threshold=0.95)

            for face in results:
                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                faceImage = pixels[y1:y2, x1:x2]
                distance = embedder.compute_distance(
                    face['embedding'], inputFaceEmbedding['embedding'])
                if distance < threshold:
                    print(distance)
                    cv2.putText(faceImage, "{:.2f}".format(
                        distance)+f" {int(cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)}", (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 0)
                    cv2_imshow(faceImage)
                    tempDict = {'frameNo': cap.get(cv2.CAP_PROP_POS_FRAMES), 'box': face['box'], 'distance': "{:.2f}".format(
                        distance), 'second': int(cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)}
                    closeFaces.append(tempDict)
    return closeFaces


app = Flask(__name__, template_folder="/content/drive/MyDrive/Thesis/FaceNet-Demo/upload_file_python/src/templates")
run_with_ngrok(app)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("image"))
    filename = ""
    imgDestination = ""
    for upload in request.files.getlist("image"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        imgDestination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", imgDestination)
        upload.save(imgDestination)

    vidDestination = ""
    print(request.files.getlist("video"))
    for upload in request.files.getlist("video"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        vidDestination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", vidDestination)
        upload.save(vidDestination)

    # print("------Threshold" + request.form['threshold'])
    # results = FindPersonInVideo(
    #     vidDestination, imgDestination, float(request.form['threshold']))

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("video.html", imageURL=imgDestination, videoURL=vidDestination, threshold=float(request.form['threshold']))


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory(os.path.join(APP_ROOT, 'images/'), filename)


@app.route('/gallery')
def get_gallery():
    image_names = os.listdir(os.path.join(APP_ROOT, 'images/'))
    print(image_names)
    return render_template("gallery.html", image_names=image_names)


@app.route('/api/FindPerson', methods=['POST', 'GET'])
def FindPerson():
    data = request.get_json()
    videoPath = gdown.download(
        data['videoURL'], output="/content/drive/MyDrive/Thesis/FaceNet-Demo/Files/video.mp4",  quiet=False, fuzzy=True)
    imagePath = gdown.download(
        data['imageURL'], output="/content/drive/MyDrive/Thesis/FaceNet-Demo/Files/image.jpg", quiet=False, fuzzy=True)
    print(videoPath, imagePath)
    if data['threshold'] != "":
        results = FindPersonInVideo(
            videoPath, imagePath, float(data['threshold']))
    else:
        results = FindPersonInVideo(videoPath, imagePath)
    print(results)
    response = jsonpickle.encode(results)
    print(response)
    # cv2_imshow(img)
    return Response(response=response, status=200, mimetype="application/json")


def gen(imagePath, videoPath, threshold=0.2):
    print("-----------", imagePath, videoPath, threshold)
    embedder = FaceNet()
    inputFaceImage = cv2.imread(imagePath)
    inputFaceEmbedding = embedder.extract(inputFaceImage, threshold=0.95)
    inputFaceEmbedding = inputFaceEmbedding[0]

    cap = cv2.VideoCapture(videoPath)
    dest_size = (160, 160)
    i = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    closeFaces = []
    while(True):
        # Capture ảnh từ video
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % fps == 0:
            pixels = frame

            # Detect khuôn mặt
            results = embedder.extract(pixels, threshold=0.95)

            for face in results:
                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                # faceImage = pixels[y1:y2, x1:x2]
                distance = embedder.compute_distance(
                    face['embedding'], inputFaceEmbedding['embedding'])
                print(distance)
                if distance < threshold:
                    cv2.rectangle(pixels,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(pixels, "{:.2f}".format(
                        distance)+f" {int(cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)}", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 0)
                else:
                    cv2.rectangle(pixels,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(pixels, "{:.2f}".format(
                        distance)+f" {int(cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)}", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 0)

            # cv2.imwrite(f"/content/drive/MyDrive/Thesis/FaceNet-Demo/Images/{time.time()}.jpg",pixels)
            ret, jpeg = cv2.imencode('.jpg', pixels)
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(request.args.get('imageURL'), request.args.get('videoURL'), float(request.args.get('threshold'))),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run()
