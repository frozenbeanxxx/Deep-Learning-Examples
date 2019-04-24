# 使用迁移学习的思想，以VGG16作为模板搭建模型，训练识别手写字体
# 引入VGG16模块
import os
# 使用第一张与第三张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"

from keras.applications.vgg16 import VGG16

# 其次加载其他模块
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam
from voc_ import DataGenerator, index_files, classes

# 加载OpenCV（在命令行中窗口中输入pip install opencv-python），这里为了后期对图像的处理，
# 大家使用pip install C:\Users\28542\Downloads\opencv_python-3.4.1+contrib-cp35-cp35m-win_amd64.whl
# 比如尺寸变化和Channel变化。这些变化是为了使图像满足VGG16所需要的输入格式
import cv2
import h5py as h5py
import numpy as np

import platform

sysstr = platform.system()
if(sysstr =="Windows"):
    user_root_path = 'D:'
    print ("Call Windows tasks")
elif(sysstr == "Linux"):
    user_root_path = '/home/weixing'
    print ("Call Linux tasks")
else:
    print ("Other System tasks")

weights_path = '/weights/keras/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
# 建立一个模型，其类型是Keras的Model类对象，我们构建的模型会将VGG16顶层（全连接层）去掉，只保留其余的网络
# 结构。这里用include_top = False表明我们迁移除顶层以外的其余网络结构到自己的模型中
# VGG模型对于输入图像数据要求高宽至少为48个像素点，由于硬件配置限制，我们选用48个像素点而不是原来
# VGG16所采用的224个像素点。即使这样仍然需要24GB以上的内存，或者使用数据生成器
image_width = 96
image_height = 96
model_vgg = VGG16(include_top=False, weights=user_root_path+weights_path, input_shape=(image_height, image_width, 3))#输入进来的数据是48*48 3通道
#选择imagnet,会选择当年大赛的初始参数
#include_top=False 去掉最后3层的全连接层看源码可知
#for layer in model_vgg.layers:
#    layer.trainable = False#别去调整之前的卷积层的参数
model = Flatten(name='flatten')(model_vgg.output)#去掉全连接层，前面都是卷积层
model = Dense(4096, activation='relu', name='fc1')(model)
model = Dense(4096, activation='relu', name='fc2')(model)
model = Dropout(0.5)(model)
model = Dense(20, activation='softmax')(model)#model就是最后的y
model_vgg_voc = Model(inputs=model_vgg.input, outputs=model, name='vgg16')
#把model_vgg.input  X传进来
#把model Y传进来 就可以训练模型了

# 打印模型结构，包括所需要的参数
#model_vgg_voc.summary()

# 新的模型不需要训练原有卷积结构里面的1471万个参数，但是注意参数还是来自于最后输出层前的两个
# 全连接层，一共有1.2亿个参数需要训练
sgd = SGD(lr=0.05, decay=1e-5)#lr 学习率 decay 梯度的逐渐减小 每迭代一次梯度就下降 0.05*（1-（10的-5））这样来变
optimizer = Adam(lr=0.0004, decay=1e-5)#lr 学习率 decay 梯度的逐渐减小 每迭代一次梯度就下降 0.05*（1-（10的-5））这样来变
#随着越来越下降 学习率越来越小 步子越小
model_vgg_voc.compile(loss='categorical_crossentropy',
                                 optimizer=sgd, metrics=['accuracy'])

pos = int(len(index_files) * 0.9)
train_images = index_files[:pos]
val_images = index_files[pos:]

partition = {'train': train_images, 'val':val_images} # IDs
labels = classes # Labels

params = {'dim': (image_height,image_width),
          'batch_size': 32,
          'n_classes': 20,
          'n_channels': 3,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['val'], labels, **params)

model_vgg_voc.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    epochs=8000)