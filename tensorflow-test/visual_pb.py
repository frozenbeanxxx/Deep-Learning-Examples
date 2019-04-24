import tensorflow as tf
from tensorflow.python.platform import gfile

from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary

def t1():
    #model = gfile.FastGFile("D:/prj_bp/ocr_server/bin/ocr_recognition/crnn_frozen_model.pb", "rb")
    model = gfile.FastGFile("D:\prj\opencv-east/frozen_east_text_detection.pb", "rb")
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
    tf.import_graph_def(graph_def, name='graph')
    summaryWriter = tf.summary.FileWriter('./', graph)

def import_to_tensorboard(model_dir, log_dir):
  """View an imported protobuf model (`.pb` file) as a graph in Tensorboard.

  Args:
    model_dir: The location of the protobuf (`pb`) model to visualize
    log_dir: The location for the Tensorboard log to begin visualization from.

  Usage:
    Call this function with your model location and desired log directory.
    Launch Tensorboard by pointing it to the log directory.
    View your imported `.pb` model as a graph.
  """
  with session.Session(graph=ops.Graph()) as sess:
    with gfile.FastGFile(model_dir, "rb") as f:
      graph_def = graph_pb2.GraphDef()
      graph_def.ParseFromString(f.read())
      #print(graph_def)
      
      '''
      # fix nodes
      for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
      '''
      importer.import_graph_def(graph_def, name='')

    pb_visual_writer = summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(log_dir))

#import_to_tensorboard('D:\\Src\\CNN-Image-Classifier\\src\\models/new_tensor_model.pb', log_dir='./log') 
import_to_tensorboard('D:\\prj_bp\\ocr_server\\bin\\ocr_recognition/crnn_frozen_model.pb', log_dir='./log') 
#import_to_tensorboard('D:\\prj_bp\\ocr_server\\bin\\ocr_detect/east_frozen_model.pb', log_dir='./log') 
