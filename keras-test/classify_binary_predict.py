import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from classify_binary_train import img_width, img_height

BGR_FLAG = True
SHOW_FLAG = True

model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
#model_path = 'D:/prj/ResourceMap/models/model.h5'
#model_weights_path = 'D:/prj/ResourceMap/models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  global SHOW_FLAG
  x = load_img(file, target_size=(img_height, img_width))
  x = img_to_array(x)
  if BGR_FLAG is True:
    x = x[:,:,::-1]
  #if SHOW_FLAG is True:
  #  SHOW_FLAG = False
  #  print(x)
  x = x /255.0
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  if result[0] > result[1]:
    #print("Predicted answer: neg")
    answer = 'neg'
  else:
    #print("Predicted answer: pos")
    answer = 'pos'

  return answer

tp = 0
tn = 0
fp = 0
fn = 0

validation_data_dir = 'D:\\dataset\\hv_royale\\InGame20190416'

for i, ret in enumerate(os.walk(validation_data_dir + "/pos")):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: pos")
    result = predict(ret[0] + '/' + filename)
    if result == "pos":
      tp += 1
    else:
      print(filename)
      fn += 1

for i, ret in enumerate(os.walk(validation_data_dir + "/neg")):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: neg")
    result = predict(ret[0] + '/' + filename)
    if result == "neg":   
      tn += 1
    else:
      print(filename)
      fp += 1

"""
Check metrics
"""
print("True Positive: ", tp)
print("True Negative: ", tn)
print("False Positive: ", fp)  # important
print("False Negative: ", fn)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
print("Precision: ", precision)
print("Recall: ", recall)

f_measure = (2 * recall * precision) / (recall + precision)
print("F-measure: ", f_measure)
