#!/usr/bin/python
# coding:utf8
import tensorflow as tf
import numpy as np
import os
import cv2

def predict(jpg_path, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
            # print (tensors)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.graph.get_operations()
            input_x = sess.graph.get_tensor_by_name("conv2d_1_input:0")  # 具体名称看上一段代码的input.name
            out_softmax = sess.graph.get_tensor_by_name("dense_2/Softmax:0")  # 具体名称看上一段代码的output.name
            image = cv2.imread(jpg_path)
            img = cv2.resize(image,(48,48),interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #cv2.imshow("win", img)
            #cv2.waitKey(0)
            img_out_softmax = sess.run(out_softmax,feed_dict={input_x: np.array(img).reshape((-1, 48, 48, 3)) / 255.0})
            print (img_out_softmax[0])
            #cv2.imshow("win", image)
            #cv2.waitKey(0)
            if (img_out_softmax[0][0] > img_out_softmax[0][1]):
                return 0
            else:
                return 1


pb_path = 'models/new_tensor_model.pb'
#rootdir = 'D:\\dataset\\hv_resource_map\\20190318_test\\pos'
rootdir = '/media/wx/0B8705400B870540/dataset/hv_resource_map/20190318_test/pos'

if __name__ == '__main__':
    label = 1
    count = 0
    right_count = 0
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            print (path)
            index = predict(path, pb_path)
            count+=1
            if(label == index):
                right_count += 1
    
    print(right_count)
    print(count)
    print(right_count/count)
