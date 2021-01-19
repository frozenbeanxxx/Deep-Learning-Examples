import sys
import os
import cv2
import time 
import random  
import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
import tensorflow as tf
from config import *
from model import create_model

import tensorflow as tf 
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

def predict_one_with_h5(file):
    x = load_img(file, target_size=(image_height, image_weight))
    x = img_to_array(x)
    x = x[:,:,::-1]
    x = x /255.0
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    #print(array[0])
    result = array[0]
    if result[0] > result[1]:
        #print("Predicted answer: neg")
        answer = 'neg'
    else:
        #print("Predicted answer: pos")
        answer = 'pos'
    return answer, result

def predict_one_with_tflite(file):
    img = load_img(file, target_size=(image_height, image_weight))
    x = img_to_array(img)
    x = x[:,:,::-1]
    x = x /255.0
    x = np.expand_dims(x, axis=0)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    result = output_data[0]
    if result[0] > result[1]:
        #print("Predicted answer: neg")
        answer = 'neg'
    else:
        #print("Predicted answer: pos")
        answer = 'pos'
    return answer, result

def predict(predict_one):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i, ret in enumerate(os.walk(test_data_dir + "/pos")):
        for i, filename in enumerate(ret[2]):
            if filename.startswith("."):
                continue
            result, probability = predict_one(ret[0] + '/' + filename)
            if result == "pos":
                tp += 1
            else:
                print(filename)
                print(probability)
                fn += 1

    for i, ret in enumerate(os.walk(test_data_dir + "/neg")):
        for i, filename in enumerate(ret[2]):
            if filename.startswith("."):
                continue
            result, probability = predict_one(ret[0] + '/' + filename)
            if result == "neg":   
                tn += 1
            else:
                print(filename)
                print(probability)
                fp += 1

    print("True Positive: ", tp)
    print("True Negative: ", tn)
    print("False Positive: ", fp)
    print("False Negative: ", fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("accuracy: ", (tp + tn)/ (tp + tn + fp + fn))
    f_measure = (2 * recall * precision) / (recall + precision)
    print("F-measure: ", f_measure)

def cut_black_edge(image):
    shape = image.shape
    edge = 65
    #qedge = 52
    p1x = edge
    p2x = shape[1] - edge
    image2 = image[:, p1x:p2x]
    return image2

def cal_image_roi(image):
    # shape[1] is image width and is image.cols too
    w_rate = 0.104629
    #y_rate = 0.601851
    y_rate = 0.7
    shape = image.shape
    w = int(shape[1] * w_rate)
    p1x = shape[1] - w
    p1y = int(shape[1] * y_rate)
    h = w * 3
    p2x = p1x + w 
    p2y = p1y + h 

    return frame[p1y:p2y, p1x:p2x]

def predict_one_image(file, mode='--h5'):
    x = cv2.imread(file)
    p1x,p1y,w,h,p2x,p2y = cal_image_roi(x)
    x = x[p1y:p1y+h, p1x:p1x+w]
    x = x.astype(np.float32)
    x = x /255.0
    x = cv2.resize(x, dsize=(image_weight, image_height))
    x = np.expand_dims(x, axis=0)

    if mode == "--tflite":
        interpreter = tf.lite.Interpreter(model_path=os.path.join(model_dir, tflite_name))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        print(str(input_details))
        output_details = interpreter.get_output_details()
        print(str(output_details))
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        result = output_data[0]
    else:
        model = load_model(os.path.join(model_dir, model_name))
        model.load_weights(os.path.join(model_dir, weight_name))
        output_data = model.predict(x)
        result = output_data[0]
    if result[0] > result[1]:
        #print("Predicted answer: neg")
        answer = 'neg'
    else:
        #print("Predicted answer: pos")
        answer = 'pos'
    print("Predicted answer: ",answer, result)

def random_pos(image_h, p1x, p1y, w, h):
    a = random.randint(-w, w)
    #print(a)
    #p1y += a
    return p1x, p1y, w, h

def save_sample(image, _p1x, _p1y, _w, _h, pos, dst_dir):
    for i in range(2):
        p1x, p1y, w, h = random_pos(image.shape[0], _p1x, _p1y, _w, _h)
        pos_img = image[p1y:p1y+h, p1x:p1x+w]
        pos_str = str(pos)
        run_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        newfile_name = run_time + "_" + pos_str + "_" + str(i) + ".jpg"
        cv2.imwrite(os.path.join(dst_dir, newfile_name), pos_img)

def predict_video(video_path, mode='--model'):
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow("video", 0)
    cv2.namedWindow("video2", 2)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    loop_flag = 0
    loop_flag = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    pos = loop_flag
    cv2.createTrackbar('time', 'video', 0, frames, lambda emp: emp )

    if mode == "--tflite":
        interpreter = tf.lite.Interpreter(model_path=os.path.join(model_dir, tflite_name))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        print(str(input_details))
        output_details = interpreter.get_output_details()
        print(str(output_details))
    elif mode == "--weight":
        model = create_model(image_height, image_weight, is_training=False)
        model.load_weights(os.path.join(model_dir, weight_name))
    else:
        model = load_model(os.path.join(model_dir, model_name))
        model.load_weights(os.path.join(model_dir, weight_name))

    # gen empty feature array
    print(model.layers[-4].name)
    print(model.layers[-4].output)
    features = np.zeros(shape=(frames, 1), dtype=np.float32)
    labels = np.zeros(shape=(frames, 1), dtype=np.float32)
    feature_func = K.function([model.input], [model.layers[-4].output])

    total_time = 0
    time_count = 1
    while(True): 
        if loop_flag == pos:
            loop_flag = loop_flag + 1
            ret, frame = cap.read()
            cv2.setTrackbarPos('time', 'video', loop_flag)
        else:
            pos = cv2.getTrackbarPos('time', 'video')
            loop_flag = pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            continue

        # Capture frame-by-frame
        if(frame is None):
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #frame = cut_black_edge(frame)
        #frame = cal_image_roi(frame)
        cv2.imshow("video2", frame)
        x = cv2.resize(frame, dsize=(image_weight, image_height))
        x = x.astype(np.float32)
        x = x /255.0
        x = np.expand_dims(x, axis=0)
        start_time = cv2.getTickCount()

        if mode == "--tflite":
            interpreter.set_tensor(input_details[0]['index'], x)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
        else:
            output_data = model.predict(x)
            feature = feature_func(x)

        once_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
        total_time += once_time
        print('time : ', once_time, total_time / time_count)
        time_count += 1
        result = output_data[0]
        if result[0] > result[1]:
            print("Predicted answer: neg", 'probility: ', result)
            answer = 'neg'
            neg_dir = '/media/wx/diskE/prj_data/video/cod_box_2/neg'
            #save_sample(frame, p1x,p1y,w,h, pos, neg_dir)
        else:
            print("Predicted answer: pos", 'probility: ', result, 'pppppppppppppppppppppppp')
            answer = 'pos'
            pos_dir = '/media/wx/diskE/prj_data/video/cod_box_2/pos'
            #save_sample(frame, p1x,p1y,w,h, pos, pos_dir)
        
        # Display the resulting frame
        #cv2.imshow("video2", frame)
        c = cv2.waitKey(30)
        if c & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    argvs = sys.argv
    argc = len(argvs)
    print('argvs:', argvs)
    if argc > 2 and (argvs[1] == "--dir"):
        if argc > 2 and (argvs[2] == "--tflite"):
            interpreter = tf.lite.Interpreter(model_path=os.path.join(model_dir, tflite_name))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            print(str(input_details))
            output_details = interpreter.get_output_details()
            print(str(output_details))
            predict(predict_one=predict_one_with_tflite)
        else:
            model = load_model(os.path.join(model_dir, model_name))
            model.load_weights(os.path.join(model_dir, weight_name))
            predict(predict_one=predict_one_with_h5)
    elif argc > 2 and (argvs[1] == "--image"):
        predict_one_image(file=argvs[3], mode=argvs[2])
    elif argc > 2 and (argvs[1] == "--video"):
        predict_video(video_path=argvs[3], mode=argvs[2])
    else:
        pass
