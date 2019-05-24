#!/usr/bin/python
# coding:utf8
##-------keras模型保存为tensorflow的二进制模型-----------

import os
import os.path as osp
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    # 将会话状态冻结为已删除的计算图,创建一个新的计算图,其中变量节点由在会话中获取其当前值的常量替换.
    # session要冻结的TensorFlow会话,keep_var_names不应冻结的变量名列表,或者无冻结图中的所有变量
    # output_names相关图输出的名称,clear_devices从图中删除设备以获得更好的可移植性
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        # 从图中删除设备以获得更好的可移植性
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        # 用相同值的常量替换图中的所有变量
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

output_fld = 'D:\\prj\\ClashRoyale\\pyscript-ingame\\models'
if not os.path.isdir(output_fld):
    raise print('errorrrrrrrrrrrrrrrr')
weight_file_path = osp.join(output_fld, 'model.h5')
K.set_learning_phase(0)
net_model = load_model(weight_file_path)

print('input is :', net_model.input.name)
print ('output is:', net_model.output.name)

# 获得当前图
sess = K.get_session()
# 冻结图
frozen_graph = freeze_session(sess, output_names=[net_model.output.op.name])

from tensorflow.python.framework import graph_io
graph_io.write_graph(frozen_graph, output_fld, 'new_tensor_model.pb', as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, 'new_tensor_model.pb'))
print (K.get_uid())


# tflite_convert --output_file=model.tflite --keras_model_file=model.h5
# toco --output_file=model.tflite --keras_model_file=model.h5
# toco --output_file=model.tflite --graph_def_file=new_tensor_model.pb
# tflite_convert --output_file=/media/wx/0B8705400B870540/prj/ResourceMap/models/model.tflite --keras_model_file=/media/wx/0B8705400B870540/prj/ResourceMap/models/model.h5
'''
tflite_convert \
  --output_file=model.h5 \
  --graph_def_file=new_tensor_model.pb \
  --input_arrays="conv2d_1_input:0" \
  --output_arrays="dense_2/Softmax:0"

toco --output_file=model.tflite \
    --graph_def_file=new_tensor_model.pb \
    --input_arrays=conv2d_1_input \
    --output_arrays=dense_2/Softmax

toco --output_file=model.tflite \
    --keras_model_file=model.h5 \
    --input_arrays=conv2d_input \
    --output_arrays=dense_1/Softmax

toco --output_file=resource_list_bgr_model.tflite --keras_model_file=model.h5 --input_arrays=conv2d_input --output_arrays=dense_1/Softmax
toco --output_file=clash_royale_ingame_bgr_model.tflite --keras_model_file=model.h5 --input_arrays=conv2d_input --output_arrays=dense_1/Softmax
toco --output_file=knives_out_ingame_bgr_model.tflite --keras_model_file=model.h5 --input_arrays=conv2d_input --output_arrays=dense_1/Softmax
'''


