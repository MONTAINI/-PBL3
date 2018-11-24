import numpy as np
import caffe
import cv2


caffe.set_mode_cpu()

label_file = '/home/pi/label.txt'
labels = np.loadtxt(label_file, str, delimiter='\t')

classifier = caffe.Classifier('/home/pi/deploy_v1.1.prototxt',
    '/home/pi/SqueezeNet_iter_10000.caffemodel',
    image_dims=[227,227], mean=np.load('/home/pi/mean.npy'),
    raw_scale=255.0,
    channel_swap=[2,1,0])

file_path = '/home/pi/nyan_a_way_raspi/lerda.jpg'
cv_img = cv2.imread(file_path)
cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
cv_float = cv2_img.astype(np.float32)
cv_float /= 255.0
predictions = classifier.predict([cv_float], False)
out_label = labels[predictions[0].argmax()]
print 'output label:', out_label
print 'probability:', predictions[0].max()
