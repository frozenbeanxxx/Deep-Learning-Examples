'''
test inference image by using onnxruntime(ORT) interface
created by AndrewWei, 20210726
'''

import os
import cv2
import numpy as np
import onnxruntime as ort


def inference_one_image():
    image_path = '../../a41_testdata/images/apple_202107261647.jpg'
    image = cv2.imread(image_path)

    #model_path = '/home/weixing/temp/opt_2021Q2/kps14_mbv2_fpn_w075_192x192_server_5.onnx'
    #model_path = '/home/weixing/temp/food/model/seresnext50_32x4d.onnx'
    model_path = '/home/weixing/prj/FoodCls/output/seresnext50_32x4d.onnx'
    #EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    #EP_list = ['CPUExecutionProvider'] # 当使用onnxruntime-gpu时，不加此语句，默认使用GPU，如果直接用CUDA_VISIBLE_DEVICES=-1屏蔽GPU，会报错
    #sess = ort.InferenceSession(model_path, providers=EP_list)
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    print('sess.get_inputs()[0]', sess.get_inputs()[0])
    input_shape = sess.get_inputs()[0].shape
    output_name = sess.get_outputs()[0].name

    img = cv2.resize(image, (input_shape[3], input_shape[2]))
    img = img.astype(np.float32)
    img = img / 255.0
    img = (img - 0.5) / 0.25
    img = np.transpose(img, axes=(2, 0, 1))
    img = np.expand_dims(img, axis=0)
    #img = np.repeat(img, 32, axis=0)
    run_num = 10
    start_time = cv2.getTickCount()
    for i in range(run_num):
        result = sess.run([output_name], {input_name: img})
    end_time = cv2.getTickCount()
    print('time:', (end_time - start_time) / cv2.getTickFrequency() * 1000 / run_num)
    #print(result)
    print('over')

if __name__ == "__main__":
    inference_one_image()

