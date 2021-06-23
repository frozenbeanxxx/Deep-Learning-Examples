import numpy as np
import cv2

from tensorflow.python.platform import gfile
import tensorflow as tf

def inference_keypoint():
    # test old cpm model, inference one picture, get keypoint
    model_file = '/home/weixing/Documents/models/online_used_20201202/mv2_cpm_small_3_concat_stages_6_small_heat_192_128.pb'
    image_path = '/home/weixing/prj/deep-high-resolution-net.pytorch/tmp/person_picture_3.jpg'


    img = cv2.imread(image_path)
    input_size = (128, 192)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)

    with gfile.FastGFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
        tensor = sess.graph.get_tensor_by_name("image:0")
        output_tensor = sess.graph.get_tensor_by_name(
            "Convolutional_Pose_Machine/stage_5_mv1/stage_5_mv1_2_pointwise/Relu:0")
        heatmap = sess.run(output_tensor, {"image:0": img})
    heatmap = heatmap.squeeze()
    heatmap = np.transpose(heatmap, (2,0,1))
    print('over')



if __name__ == "__main__":
    inference_keypoint()