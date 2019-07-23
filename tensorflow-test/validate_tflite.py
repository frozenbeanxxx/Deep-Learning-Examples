import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import numpy as np
import time
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input

#test_image_dir = '../data/trainingData/pos'
model_path = "D:\\prj\\ClashRoyalebp\\pyscript-ingame\\models/clash_royale_ingame_bgr_model.tflite"
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print(str(input_details))
output_details = interpreter.get_output_details()
print(str(output_details))

#full_path = 'D:/dataset/hv_royale/InGameTest20190417/neg/2019-07-19-15-56-33_208_0.jpg'
full_path = 'D:/dataset/hv_royale/InGameTest20190417/pos/2019-07-19-16-02-28_9_1.jpg'
img = image.load_img(full_path, target_size=(48, 16))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x, mode='tf')

model_interpreter_start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], x)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
model_interpreter_time = time.time() - model_interpreter_start_time

# 出来的结果去掉没用的维度
result = np.squeeze(output_data)
print('result:{}'.format(result))