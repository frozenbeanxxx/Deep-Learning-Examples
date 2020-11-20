#coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data 
import scipy.misc 
import os 

mnist_path = "D:/dataset/mnist"
mnist = input_data.read_data_sets(mnist_path, one_hot=True)

save_dir = mnist_path + "/raw/"
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

for i in range(20):
    image = mnist.train.images[i,:]
    image = image.reshape(28, 28)
    filename = save_dir + "mnist_train_%d.jpg" % i 
    scipy.misc.toimage(image, cmin=0.0, cmax=1.0).save(filename)

image = mnist.train.images[0,:]
print(type(image), image.shape)
image.reshape(1, 784)
print(type(image), image.shape)
