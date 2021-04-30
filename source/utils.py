import os
import cv2
from keras_preprocessing.image import img_to_array
from flask import flash


def preprocess_face_frame(face_frame):
    # convert to RGB
    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    # preprocess input image for mobilenet
    face_frame_resized = cv2.resize(face_frame, (224, 224))
    face_frame_array = img_to_array(face_frame_resized)
    return face_frame_array


def decode_prediction(pred):
    # (incorrectly_mask, mask, no_mask) = pred
    # # print(pred)
    # mask_or_not = ""
    # if mask > no_mask:
    #     mask_or_not = "Mask"
    #     print(mask)
    #     print('mask')
    #     if mask < incorrectly_mask:
    #         mask_or_not = "Incorrectly Worn"
    #         print(incorrectly_mask)
    #         print('Incorrectly Worn')
    # else:
    #     if no_mask < incorrectly_mask:
    #         mask_or_not = "Incorrectly Worn"
    #     mask_or_not = "No Mask"
    #     print(no_mask)
    #     print('no mask')
    # confidence = f"{(max(incorrectly_mask, mask, no_mask) * 100):.2f}"
    # return mask_or_not, confidence

    (mask, no_mask) = pred
    mask_or_not = "Mask" if mask > no_mask else "No mask"
    confidence = f"{(max(mask, no_mask) * 100):.2f}"
    return mask_or_not, confidence


def write_bb(mask_or_not, confidence, box, frame):
    (x, y, w, h) = box
    color = (154, 171, 5) if mask_or_not == "Mask" else (65, 31, 237)
    # label = f"{mask_or_not}: {confidence}%"
    label = f"{mask_or_not}"
    # if mask_or_not == "No mask":
    #     flash("Not So OK", 'error')
    #     print("gapake masker")
    # print(mask_or_not)

    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.45, color, 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def load_cascade_detector():
    cascade_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
    print(cascade_path)
    return face_detector
