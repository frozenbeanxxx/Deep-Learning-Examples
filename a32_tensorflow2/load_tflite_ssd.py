import argparse
import time

import numpy as np
from PIL import Image
import cv2
import tensorflow as tf # TF2

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
    root = '/home/weixing/temp/APU_test_models'
    model_name = '/mb2-ssd-lite_0227_2_tf1140.tflite'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default=root+'/test_image_1.jpg',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default=root+model_name,
        help='.tflite model to be executed')
    parser.add_argument(
        '--input_mean',
        default=0., type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=1., type=float,
        help='input standard deviation')
    parser.add_argument(
        '--num_threads', default=None, type=int, help='number of threads')
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(
        model_path=args.model_file)#, num_threads=args.num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][2]
    width = input_details[0]['shape'][3]
    #img = Image.open(args.image).resize((width, height))
    image = cv2.imread(args.image)
    img = cv2.resize(image, (width, height))
    img = img.transpose((2, 0, 1))

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    warmup_step = 5
    for i in range(warmup_step):
        interpreter.invoke()

    looping_step = 100
    start_time = time.time()
    for i in range(looping_step):
        interpreter.invoke()
    stop_time = time.time()

    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
'''
    results = np.sum(results, axis=-1)

    mask = cv2.resize(results, (image.shape[1], image.shape[0]))
    mask *= 192
    mask = mask.astype(np.int8)
    cv2.imshow('win1', image)
    cv2.imshow('win2', mask)
    cv2.waitKey(0)

    
'''