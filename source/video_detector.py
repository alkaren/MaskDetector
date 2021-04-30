import time
import numpy as np
import cv2
import imutils
import os
from datetime import date
from imutils.video import VideoStream
from tensorflow import keras
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input

from source.utils import preprocess_face_frame, decode_prediction, write_bb, load_cascade_detector
from flask import render_template, Response, flash

model = keras.models.load_model('models/mask_mobilenet.h5')
# model = keras.models.load_model('testmodel/model.h5')
face_detector = load_cascade_detector()

statusNotif = 0

def video_mask_detector():
    video = VideoStream(src=0).start()
    # video = cv2.VideoCapture('test.mp4')
    time.sleep(1.0)
    while True:
        # Capture frame-by-frame
        frame = video.read()

        frame = detect_mask_in_frame(frame)
        # Display the resulting frame
        # show the output frame
        cv2.imshow("Mask detector", frame)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # cleanup
    cv2.destroyAllWindows()
    video.stop()


def detect_mask_in_frame(frame):
    frame = imutils.resize(frame, width=700)
    global statusNotif
    today = date.today()
    pathDefault = "G:/skripsi/CV-Mask-detection-master/app/static/gambar_wajah/"
    checkFolderToday = os.path.isdir(pathDefault + str(today))
    # print("folder hari ini =", checkFolderToday)
    if checkFolderToday == False:
        NewPath = os.path.join(pathDefault, str(today))
        NewPath2 = os.path.join(pathDefault + str(today), "noMask")
        NewPath3 = os.path.join(pathDefault + str(today), "withMask")
        os.mkdir(NewPath)
        os.mkdir(NewPath2)
        os.mkdir(NewPath3)

    # convert an image from one color space to another
    # (to grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray,
                                           scaleFactor=1.1,
                                           minNeighbors=5,
                                           minSize=(40, 40),
                                           flags=cv2.CASCADE_SCALE_IMAGE,
                                           )

    faces_dict = {"faces_list": [],
                  "faces_rect": []
                  }

    SumFaces = len(faces)
    if (SumFaces == 0):
        statusNotif = 0
        # print("tidak ada wajah")
    for rect in faces:
        (x, y, w, h) = rect
        face_frame = frame[y:y + h, x:x + w]
        # preprocess image
        face_frame_prepared = preprocess_face_frame(face_frame)

        faces_dict["faces_list"].append(face_frame_prepared)
        faces_dict["faces_rect"].append(rect)

    if faces_dict["faces_list"]:
        faces_preprocessed = preprocess_input(np.array(faces_dict["faces_list"]))
        preds = model.predict(faces_preprocessed)
        # SumFaces = len(faces_dict["faces_rect"])
        # print(SumFaces)

        for i, pred in enumerate(preds):
            strTime = str(today)
            varNoMask = strTime + "-NoMask-"
            # print(varNoMask)
            varMask = strTime + "-WithMask-"
            path = 'G:/skripsi/CV-Mask-detection-master/app/static/gambar_wajah/' + strTime
            crop_img = face_frame
            mask_or_not, confidence = decode_prediction(pred)
            # write_bb(mask_or_not, confidence, faces_dict["faces_rect"][i], frame)
            FloatConfidence = float(confidence)
            if mask_or_not == "No mask":
                if FloatConfidence > 90.00:
                    statusNotif = 1
                    # print(faces_dict["faces_rect"][i])
                    listToStr = ' '.join(map(str, faces_dict["faces_rect"][i]))
                    cv2.imwrite(os.path.join(path+str("/noMask"), str(varNoMask) + str(listToStr.replace(" ","")) + '.png'), crop_img) 
                    write_bb(mask_or_not, confidence, faces_dict["faces_rect"][i], frame)
                    # print(path)
                    # print(FloatConfidence)
            # print(confidence)
            else:
                if FloatConfidence > 90.00:
                    statusNotif = 0
                    listToStr = ' '.join(map(str, faces_dict["faces_rect"][i]))
                    cv2.imwrite(os.path.join(path+str("/withMask"), str(varMask) +str(listToStr.replace(" ","")) + '.png'), crop_img)
                    write_bb(mask_or_not, confidence, faces_dict["faces_rect"][i], frame)

    return frame

def print_notif():
    global statusNotif
    # print(statusNotif)
    if statusNotif == 1:
        # print(statusNotif)
        return statusNotif
    return statusNotif

if __name__ == '__main__':
    video_mask_detector()
