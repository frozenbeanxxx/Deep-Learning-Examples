import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
#from tensorflow import keras

def parse_args():
    parser = argparse.ArgumentParser('pb to tflite')
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    args = parser.parse_args()
    return args

def convert_xxx_to_tflite():
    args = parse_args()
    src_path = args.src
    dst_path = args.dst
    converter = tf.lite.TFLiteConverter.from_saved_model(src_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(dst_path, 'wb') as f:
        f.write(tflite_model)

def load_saved_model():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    args = parse_args()
    src_path = args.src
    dst_path = args.dst
    dummy_input = np.random.rand(192, 192, 3) * 255
    dummy_input = dummy_input.astype(np.int32)
    dummy_input = np.expand_dims(dummy_input, 0)
    #dummy_input = tf.convert_to_tensor(dummy_input)
    loaded = tf.saved_model.load(src_path)
    print(list(loaded.signatures.keys()))
    model = loaded.signatures["serving_default"]


    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(x=(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)))
    y = full_model(dummy_input)
    #frozen_func = convert_variables_to_constants_v2(full_model)
    #frozen_func.graph.as_graph_def()
    #tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
    #                  logdir="./frozen_models",
    #                  name="simple_frozen_graph.pb",
    #                  as_text=False)
    print('over')

def load_tflite_inference():
    cv2.namedWindow('win', 0)
    #model_file = '/home/weixing/temp/MoveNet/movenet_singlepose_lightning_3.tflite'
    model_file = '/home/weixing/src/06_model/PINTO_model_zoo/115_MoveNet/saved_model/movenet_singlepose_lightning_3.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    #image_path = '/home/weixing/temp/image/202104221317.jpg'
    #image_path = '/media/weixing/diskD/dataset_res/yoga/images_20210514_filter/Akarna_Dhanurasana/19.jpg'
    image_path = '/media/weixing/diskD/dataset_res/yoga/images_20210514_filter/Cockerel_Pose/4_45.jpg'
    #image_path = '/media/weixing/diskD/dataset_res/yoga/images_20210514_filter/Side_Plank_Pose_or_Vasisthasana_/74.jpg'

    image = cv2.imread(image_path)
    imgH, imgW, _ = image.shape
    input_data = cv2.resize(image, (width, height))
    input_data = input_data[:,:,::-1]
    #input_data = input_data.astype(np.int32)
    input_data = input_data.astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = cv2.getTickCount()
    interpreter.invoke()
    print('time use:', (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000)

    if 0:
        looping_step = 100
        for i in range(looping_step):
            start_time = cv2.getTickCount()
            interpreter.invoke()
            print('time use:', (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000)

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    for i in range(results.shape[0]):
        if results[i][2] > 0.3:
            x = int(results[i][1] * imgW)
            y = int(results[i][0] * imgH)
            cv2.circle(image, (x,y), 3, (255,0,0), 2)
    cv2.imshow('win', image)
    cv2.waitKey(0)
    print('over')


def load_tflite_inference_on_camera():
    cv2.namedWindow('win', 0)

    cap = cv2.VideoCapture(0)

    #model_file = '/home/weixing/temp/MoveNet/movenet_singlepose_lightning_3.tflite'
    model_file = '/home/weixing/src/06_model/PINTO_model_zoo/115_MoveNet/saved_model/movenet_singlepose_lightning_3.tflite'

    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        frameH, frameW, _ = frame.shape
        startX = (frameW - frameH)//2
        image = frame[:, startX:(startX + frameH),:]
        imgH, imgW, _ = image.shape
        input_data = cv2.resize(image, (width, height))
        #input_data = input_data[:, :, ::-1]
        #input_data = input_data.astype(np.int32)
        input_data = input_data.astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = cv2.getTickCount()
        interpreter.invoke()
        print('time use:', (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000)

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        for i in range(results.shape[0]):
            if results[i][2] > 0.3:
                x = int(results[i][1] * imgW)
                y = int(results[i][0] * imgH)
                cv2.circle(image, (x, y), 3, (255, 0, 0), 2)

        cv2.imshow('win', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
    #convert_xxx_to_tflite()
    #load_saved_model()
    load_tflite_inference()
    #load_tflite_inference_on_camera()
