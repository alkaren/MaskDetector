import time
import numpy as np
import cv2
import imutils
import os
import threading

from datetime import date, datetime
from imutils.video import VideoStream
from tensorflow import keras
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input

from source.utils import preprocess_face_frame, decode_prediction, write_bb, load_cascade_detector, check_count_image
from flask import render_template, Response, flash

model = keras.models.load_model('models/mask_mobilenet.h5')
# model = keras.models.load_model('models/old_model.h5')

# model = keras.models.load_model('testmodel/model.h5')
face_detector = load_cascade_detector()

global asdas
asdas = check_count_image()

statusNotif = 0
statusSave = 0
statusThread = False

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
    global statusSave
    global asdas
    today = date.today()
    currentTime = datetime.now().strftime("%H-%M")
    pathDefault = "G:/skripsi/MaskDetector/app/static/gambar_wajah/"
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
                                           scaleFactor=1.05,
                                           minNeighbors=5,
                                           minSize=(70, 70),
                                           maxSize=(300, 300),
                                           flags=cv2.CASCADE_SCALE_IMAGE,
                                           )
    counterObjectMask = 0
    
    faces_dict = {"faces_list": [],
                  "faces_rect": []
                  }

    SumFaces = len(faces)
    if (SumFaces == 0):
        statusNotif = 0
        statusSave = 0
        asdas = check_count_image()
        print("tidak ada wajah")
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
        
        # print(asdas[1])
        counterObjectNoMask = asdas[1]

        for i, pred in enumerate(preds):

            strTime = str(today)
            varNoMask = currentTime + "-NoMask-"
            # print(varNoMask)
            varMask = currentTime + "-WithMask-"
            path = 'G:/skripsi/MaskDetector/app/static/gambar_wajah/' + strTime
            crop_img = faces_dict["faces_list"][i]
            mask_or_not, confidence = decode_prediction(pred)
            # write_bb(mask_or_not, confidence, faces_dict["faces_rect"][i], frame)
            FloatConfidence = float(confidence)
            #print(FloatConfidence)
            if mask_or_not == "No mask":
                # print(obj)

                counterObjectNoMask = counterObjectNoMask + 1
                print(counterObjectNoMask)
                statusNotif = 1
                
                cv2.imwrite(os.path.join(path+str("/noMask"), str(counterObjectNoMask) + '.png'), crop_img) 
                write_bb(mask_or_not, counterObjectNoMask, faces_dict["faces_rect"][i], frame)
                # write_bb(mask_or_not, confidence, faces_dict["faces_rect"][i], frame)
                # print(faces_dict["faces_rect"][i])
                #listToStr = ' '.join(map(str, faces_dict["faces_rect"][i]))
                # cv2.imwrite(os.path.join(path+str("/noMask"), str(varNoMask) + str(listToStr.replace(" ","")) + '.png'), crop_img)
                # if statusSave == 0:
                #     cv2.imwrite(os.path.join(path+str("/noMask"), str(varNoMask) + str(counterObjectNoMask) + '.png'), crop_img) 
                #     write_bb(mask_or_not, confidence, faces_dict["faces_rect"][i], frame)
                # else:
                #     write_bb(mask_or_not, confidence, faces_dict["faces_rect"][i], frame)
                
                # if statusThread == False:
                #     print(statusSave)
                #     threading.Timer(20.0, stop_save).start()
                #     statusThread == True

            else:
                counterObjectMask = counterObjectMask + 1
                statusNotif = 0
                listToStr = ' '.join(map(str, faces_dict["faces_rect"][i]))
                # cv2.imwrite(os.path.join(path+str("/withMask"), str(varMask) +str(listToStr.replace(" ","")) + '.png'), crop_img)
                cv2.imwrite(os.path.join(path+str("/withMask"), str(counterObjectMask) + '.png'), crop_img)
                write_bb(mask_or_not, confidence, faces_dict["faces_rect"][i], frame)
        
    # time.sleep(10)
    # event.wait

    return frame

def print_notif():
    global statusNotif
    # print(statusNotif)
    if statusNotif == 1:
        # print(statusNotif)
        return statusNotif
    return statusNotif

def stop_save():
    global statusSave
    # print(statusNotif)
    if statusSave == 0:
        print("Pause Save!")
        statusSave = 1
        # print(statusSave)
        continue_save()
        return statusSave
    return statusSave

def continue_save():
    global statusSave
    global statusThread
    print("Continue Save!")
    statusSave = 0
    statusThread = False
    return statusSave, statusThread


class obj():
    def __init__(self, objectid):
        self.objectid = objectid

if __name__ == '__main__':
    video_mask_detector()
