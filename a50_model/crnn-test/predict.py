import cv2
import itertools, os, time
import numpy as np
from model import *
from keras import backend as K
K.set_learning_phase(0)


model = create_model(training=False)
try:
    model.load_weights('./logs/weights201904191045.h5')
    print("...Previous weight data...")
except:
    raise Exception("No weight file!")

def predict_image():
    img = cv2.imread('D:\\dataset\\mjsynth\\mnt\\ramdisk\\max\\90kDICT32px/./2697/6/466_MONIKER_49537.jpg')

    img_pred = img.astype(np.float32)
    img_pred = cv2.resize(img_pred, (img_w, img_h))
    print(img_pred.shape)
    img_pred = (img_pred / 255.0)
    img_pred = img_pred.transpose(1,0,2)
    print(img_pred.shape)
    img_pred = np.expand_dims(img_pred, axis=0)
    print(img_pred.shape)

    net_out_value = model.predict(img_pred)
    out = net_out_value[0]
    print(out)
    out = np.argmax(out, axis=1)
    print(out)
    out = [CHAR_VECTOR[i - 1] for i in out]
    print(out)

predict_image()

