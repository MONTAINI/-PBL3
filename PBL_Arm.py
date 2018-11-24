#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import time
import cv2
import numpy as np
import sys
sys.path.append('/home/pi/caffe/python')
import caffe
from picamera.array import PiRGBArray
from picamera import PiCamera
import threading
import requests
import Arm_Angle

caffe.set_mode_cpu()

label_file = '/home/pi/label.txt'
labels = np.loadtxt(label_file, str, delimiter='\t')
classifier = caffe.Classifier('/home/pi/deploy_v1.1.prototxt',
    '/home/pi/SqueezeNet_iter_10000.caffemodel',
    image_dims=[227,227], mean=np.load('/home/pi/mean.npy'),
    raw_scale=255.0,
    channel_swap=[2,1,0])
#上記は必要なモジュールをimportしcaffeの設定をしている

S_Width = 640 
S_Height = 480
delta_thresh = 20
min_area = 100
max_area = S_Width*S_Height/3

#上記がカメラで取得する画像の設定 testの結果画像サイズを大きくすると動体検知が遅くなる
camera = PiCamera()
camera.resolution = (S_Width,S_Height)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(S_Width,S_Height))

avg = None  #平均画像を保存する変数
cv_float = 0.0 #caffeで判別するための画像を保存する変数
#上二行はroopで使用する変数初期化

for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = cv2.flip(f.array, -1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if avg is None:
        avg = gray.copy().astype("float")
        rawCapture.truncate(0) #画像が溜まっているキューをクリアして最新の画像を取得する関数
        continue

    cv2.accumulateWeighted(gray, avg, 0.5)
    f_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    _, thresh = cv2.threshold(f_delta, delta_thresh, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = frame.shape[:2]
    min_x = width
    min_y = height
    max_x = 0
    max_y = 0
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x+w)
        max_y = max(max_y, y+h)

    too_big_flg = True
    if max_x - min_x < max_y - min_y:
        xy_length = abs(max_y - min_y)
    else:
        xy_length = abs(max_x - min_x)

    center_x = (min_x + max_x)/2.0
    min_x = int(round(max(0, center_x - xy_length/2.0)))
    max_x = int(round(min(width, center_x + xy_length/2.0)))
    center_y = (min_y + max_y)/2.0
    min_y = int(round(max(0, center_y - xy_length/2.0)))
    max_y = int(round(min(height, center_y + xy_length/2.0)))

    if max_area < (max_x-min_x)*(max_y-min_y):
        too_big_flg = True
    else:
        too_big_flg = False

    dst = frame[min_y:max_y, min_x:max_x]
    cv_img = dst.copy()
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_float = cv_img.astype(np.float32)
    cv_float /= 255.0

    cat_detected_time = datetime.datetime.now() #ディープラーニング前の下処理
    if too_big_flg == False:
        tx = (min_x+max_x)/2
        ty = (min_y+max_y)/2
        print ("座標は (%d, %d), size = %d x %d" % ( tx, ty, max_x-min_x, max_y-min_y))
	if int(tx) <= 300:
		u = (max_x-min_x) / 10
		i = (max_y-min_y) / 10
        	arm=Arm_Angle.Arm()
        	arm.angle_LR(int(u))
        	arm.angle_UD(int(i))
	else:
 		u = (max_x-min_x) / 10
                i = (max_y-min_y) / 10
                arm=Arm_Angle.Arm()
                arm.angle_LR(int(-u))
                arm.angle_UD(int(-i))
        predictions = classifier.predict([cv_float], False)
        prob = predictions[0].max()
        out_label = labels[predictions[0].argmax()]
        if out_label == 'cat'  and 0.99 < prob:
            cv2.putText(frame, "Probability:"+str(prob), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, str(datetime.datetime.today()), (S_Width-240, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
            print ('*** Targetを発見した ! ' + str(prob) + " : " + out_label + " ***")
	    print ("座標は (%d, %d), size = %d x %d" % ( tx, ty, max_x-min_x, max_y-min_y))

            tmpstr = cat_detected_time.strftime('value1=detect%Y%m%d_%H%M%S.%f.jpg')
	    fn = cat_detected_time.strftime('/home/pi/Pictures/detect%Y%m%d_%H%M%S.%f.jpg')
            threading.Thread(target=cv2.imwrite, args=(fn, frame,)).start()

    rawCapture.truncate(0)
