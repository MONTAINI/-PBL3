import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera


delta_thresh = 20
min_area = 100
max_area = 640*480/3

camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640,480))

cnt = 0
avg = None
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = cv2.flip(f.array, -1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if avg is None:
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
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

    if too_big_flg == False:
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
        fn = "/home/pi/Pictures/moved_" + str(cnt) + ".jpg"
        cv2.imwrite(fn, frame)
        print (fn)
        cnt = cnt + 1
        if 10 <= cnt:
            break

    rawCapture.truncate(0)
    
