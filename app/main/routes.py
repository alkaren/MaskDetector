from base64 import b64encode
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from flask import render_template, Response, flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from werkzeug.exceptions import abort
from wtforms import FileField, SubmitField
from app.main import main_bp
from app.main.camera import Camera

from source.test_new_images import detect_mask_in_image
from source.video_detector import detect_mask_in_frame
from source.video_detector import print_notif

import json
import time
import os
from datetime import date
from datetime import timedelta
from time import gmtime, strftime

counter = 100
statusNotif = "nothing"

@main_bp.route("/")
def home_page():
    # flash("Not So OK", 'error')
    # gg = show_notif()
    # print(gg)
    return render_template("home_page.html")

@main_bp.route("/test")
def chart():
    # Get today's date
  today = date.today()
  print("Today is: ", today)
    
  # Yesterday date
  yesterday = today - timedelta(days = 1)
  print("Yesterday was: ", yesterday)

  pathDefault = "G:/skripsi/CV-Mask-detection-master/app/static/gambar_wajah/"

  checkFolderToday = os.path.isdir(pathDefault + str(today))
  print(checkFolderToday)
  if checkFolderToday == True:
    countWithMaskToday = len(os.listdir(pathDefault + str(today) + "/withMask"))
    countNoMaskToday = len(os.listdir(pathDefault + str(today) + "/noMask"))
  else:
    countWithMaskToday = 0
    countNoMaskToday = 0
  print(countWithMaskToday)
  print(countNoMaskToday)

  checkFolderYesterday = os.path.isdir(pathDefault + str(yesterday))
  print(checkFolderYesterday)
  if checkFolderYesterday == True:
    countWithMaskYesterday = len(os.listdir(pathDefault + str(yesterday) + "/withMask"))
    countNoMaskYesterday = len(os.listdir(pathDefault + str(yesterday) + "/noMask"))
  else:
    countWithMaskYesterday = 0
    countNoMaskYesterday = 0
  print(countWithMaskYesterday)
  print(countNoMaskYesterday)

  my_list = os.listdir(pathDefault)
  countListDir = len(my_list)
  print(my_list)
  print(countListDir)
  countWithMaskAll = 0
  countNoMaskAll = 0
  if countListDir > 0:
    for FolderX in my_list:
      countWithMaskAll = countWithMaskAll + len(os.listdir(pathDefault + str(FolderX) + "/withMask"))
      countNoMaskAll = countNoMaskAll + len(os.listdir(pathDefault + str(FolderX) + "/noMask"))
  else:
    countWithMaskAll = 0
    countNoMaskAll = 0
  
  print(countWithMaskAll)
  print(countNoMaskAll)
  legend = 'Wearing Mask Data'
  labels = ["With Mask", "No Mask"]

  values = [countWithMaskToday, countNoMaskToday]
  values2 = [countWithMaskYesterday, countNoMaskYesterday]
  values3 = [countWithMaskAll, countNoMaskAll]
  return render_template('test.html', values=values, values2=values2, values3=values3, labels=labels, legend=legend)

def gen(camera):

    while True:
        frame = camera.get_frame()
        frame_processed = detect_mask_in_frame(frame)
        frame_processed = cv2.imencode('.jpg', frame_processed)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_processed + b'\r\n')



@main_bp.route('/video_feed')
def video_feed():
    return Response(gen(
        Camera()
    ),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@main_bp.route("/listen")
def listen():

  def respond_to_client():
    while True:
      global counter
      global statusNotif
      statusNotif = print_notif()
    #   print(statusNotif)
      color = "white"
    #   with open("color.txt", "r") as f:
    #     color = f.read()
    #     print("******************")
      if(color == "white"):
        # print(counter)
        # counter += 1
        _data = json.dumps({"color":color, "counter":statusNotif})
        yield f"id: 1\ndata: {_data}\nevent: online\n\n"
      time.sleep(0.5)
  return Response(respond_to_client(), mimetype='text/event-stream')

def allowed_file(filename):
    ext = filename.split(".")[-1]
    is_good = ext in ["jpg", "jpeg", "png"]
    return is_good


@main_bp.route("/image-mask-detector", methods=["GET", "POST"])
def image_mask_detection():
    return render_template("image_detector.html",
                           form=PhotoMaskForm())


@main_bp.route("/image-processing", methods=["POST"])
def image_processing():
    form = PhotoMaskForm()

    if not form.validate_on_submit():
        flash("An error occurred", "danger")
        abort(Response("Error", 400))

    pil_image = Image.open(form.image.data)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    array_image = detect_mask_in_image(image)
    rgb_image = cv2.cvtColor(array_image, cv2.COLOR_BGR2RGB)
    image_detected = Image.fromarray(rgb_image, 'RGB')

    with BytesIO() as img_io:
        image_detected.save(img_io, 'PNG')
        img_io.seek(0)
        base64img = "data:image/png;base64," + b64encode(img_io.getvalue()).decode('ascii')
        return base64img


# form
class PhotoMaskForm(FlaskForm):
    image = FileField('Choose image:',
                      validators=[
                          FileAllowed(['jpg', 'jpeg', 'png'], 'The allowed extensions are: .jpg, .jpeg and .png')])

    submit = SubmitField('Detect mask')
