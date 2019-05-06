from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

from keras.preprocessing import image
import keras.backend as K
import numpy as np
import cv2
import sys

img_path = './data/dog.jpg'

# load the model
model = VGG16(weights='D:\\weights\\keras/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
model.summary()
# load an image from file
image = load_img(img_path, target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))


class_idx = np.argmax(yhat[0])
print(model.output)
print(class_idx)
class_output = model.output[:, class_idx]
print(class_output)
last_conv_layer = model.get_layer("block5_conv1")
print(last_conv_layer.name)
print(last_conv_layer.output)
print(last_conv_layer.output[0])
print(model.input)

grads = K.gradients(class_output, last_conv_layer.output)[0]
print(grads)
pooled_grads = K.mean(grads, axis=(0, 1, 2))
print(pooled_grads)
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([image])
#print(pooled_grads_value)
#print(conv_layer_output_value)
for i in range(512):
    #conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    conv_layer_output_value[:, :, i]# = pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#cv2.imshow("heatmap", heatmap)
#cv2.waitKey(0)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
cv2.imshow("Original", img)
cv2.imshow("GradCam", superimposed_img)
cv2.waitKey(0)
