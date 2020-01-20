import numpy as np 
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.models import load_model
import keras.backend as K
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def t1():
    val = np.array([[[[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
                    [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
                    [[-1,-1,-1],[-1,-1,-1],[-2,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
                    [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
                    [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
                    [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
                    [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]]])
    print(val.shape) # (1, 6, 6, 3)
    x = K.variable(value = val)
    y = MaxPooling2D((3, 3), strides = 2, padding = 'same')(x)
    #y = MaxPooling2D((3, 3), strides = 2, padding = 'valid')(x)
    print(y.shape) # (1, 3, 3, 3)
    print(y) # (1, 3, 3, 3)
    print(K.eval(y))

def t2():
    model = load_model(r'E:\prj_data\hv_ocr\cod\model\model.h5')
    model.summary()

def t3():
    a = np.arange(24).reshape(2,3,4) # a和b的维度有些讲究，具体查看Dot类的build方法
    b = np.arange(48).reshape(2,3,8)
    output1 = K.batch_dot(K.constant(a), K.constant(b),  axes=1)
    print(output1)
    output2 = K.batch_dot(K.constant(a), K.constant(b),  axes=2)
    print(output2)

if __name__ == "__main__":
    t3()

