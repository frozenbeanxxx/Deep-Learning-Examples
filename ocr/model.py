import os 
import fire
import cv2
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import ReLU
from tensorflow.python.keras.layers import ZeroPadding2D, Conv2D, AveragePooling2D, DepthwiseConv2D, Dropout, Lambda
from tensorflow.python.keras.layers.normalization import BatchNormalization

def HardSigmoid(input):
    x = Lambda(lambda x: x+3.0)(input)
    y = ReLU(max_value=6)(x)
    y = Lambda(lambda x: x/6.0)(y)
    return y

def mul(x):
    return x[0]*x[1]

def add(x):
    return x[0]+x[1]

def HardSwish(input):
    a = Lambda(lambda x: x+3.0)(input)
    b = ReLU(max_value=6)(a)
    c = Lambda(lambda x: x/6.0)(b)
    y = Lambda(mul)([input,c])
    return y

def conv_norm_act(
    input, 
    filters, 
    kernel_size=3, 
    stride=1, 
    padding=0, 
    norm_layer=None, 
    act_layer="relu", 
    use_bias=True, 
    l2_reg=1e-5):

    if padding > 0:
        x = ZeroPadding2D(padding)(input)
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, kernel_regularizer=regularizers.l2(l2_reg), use_bias=use_bias)(x)
    else:
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, kernel_regularizer=regularizers.l2(l2_reg), use_bias=use_bias)(input)
    if norm_layer=='bn': 
        x = BatchNormalization(momentum=0.99)(x)
    
    if act_layer == "relu":
        x = ReLU()(x)
    
    if act_layer == "relu6":
        x = ReLU(max_value=6)(x)
      
    if act_layer == "hswish":
        x = HardSwish(x)
    
    if act_layer == "hsigmoid":
        x = HardSigmoid(x)
      
    if act_layer == "softmax":
        x = Softmax()(x)
    
    return x

def SE(input, reduction=4, l2_reg=0.01):

    input_shape = input.shape
    input_channels = int(input_shape[3])
    print('reduction', reduction)
    print('input_channels', input_channels)
    pool_size = tuple(map(int, input_shape[1:3]))
    y = AveragePooling2D(pool_size=pool_size)(input)
    y = conv_norm_act(y, input_channels // reduction, kernel_size=1, norm_layer=None, act_layer="relu", use_bias=False, l2_reg=l2_reg)
    y = conv_norm_act(y, input_channels, kernel_size=1, norm_layer=None, act_layer="hsigmoid", use_bias=False, l2_reg=l2_reg)
    return Lambda(mul)([input,y])

def Bneck(input, out_channels, exp_channels, kernel_size, stride, use_se, act_layer, l2_reg=1e-5):
    input_shape = input.shape
    in_channels = int(input_shape[3])
    x = conv_norm_act(input, exp_channels, kernel_size=1, norm_layer="bn", act_layer=act_layer, use_bias=False, l2_reg=l2_reg)
    dw_padding = (kernel_size - 1) // 2
    x = ZeroPadding2D(padding=dw_padding)(x)
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, depthwise_regularizer=regularizers.l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization(momentum=0.99)(x)
    
    if use_se:
        input_shape = x.shape
        x = SE(x, l2_reg=l2_reg)
   
    x = BatchNormalization(momentum=0.99)(x)
    
    if act_layer == "relu":
        x = ReLU()(x)
    
    if act_layer == "relu6":
        x = ReLU(max_value=6)(x)
      
    if act_layer == "hswish":
        x = HardSwish(x)
    
    if act_layer == "hsigmoid":
        x = HardSigmoid(x)
      
    if act_layer == "softmax":
        x = Softmax()(x)
    
    x = conv_norm_act(x, out_channels, kernel_size=1, norm_layer="bn", act_layer=None, use_bias=False, l2_reg=l2_reg)
    
    if stride == 1 and in_channels == out_channels:
        return Lambda(add)([input,x])
    else:
        return x

def MobilenetV3(inputs, l2_reg=1e-5):
    x = conv_norm_act(inputs, 16, kernel_size=3, stride=1, padding=1, norm_layer='bn', act_layer='hswish', use_bias=False, l2_reg=l2_reg)
    x = Bneck(x, out_channels=16, exp_channels=16, kernel_size=3, stride=1, use_se=False, act_layer="relu", l2_reg=l2_reg)
    x = Bneck(x, out_channels=24, exp_channels=64, kernel_size=3, stride=2, use_se=False, act_layer="relu", l2_reg=l2_reg)
    x = Bneck(x, out_channels=24, exp_channels=72, kernel_size=3, stride=1, use_se=False, act_layer="relu", l2_reg=l2_reg)
    x = Bneck(x, out_channels=40, exp_channels=72, kernel_size=5, stride=1, use_se=True, act_layer="relu", l2_reg=l2_reg)
    x = Bneck(x, out_channels=40, exp_channels=120, kernel_size=3, stride=1, use_se=True, act_layer="hswish", l2_reg=l2_reg)
    x = Bneck(x, out_channels=72, exp_channels=120, kernel_size=3, stride=2, use_se=False, act_layer="hswish", l2_reg=l2_reg)
    x = Bneck(x, out_channels=72, exp_channels=120, kernel_size=3, stride=1, use_se=False, act_layer="hswish", l2_reg=l2_reg)
    x = Bneck(x, out_channels=112, exp_channels=180, kernel_size=3, stride=1, use_se=False, act_layer="hswish", l2_reg=l2_reg)
    x = Bneck(x, out_channels=160, exp_channels=320, kernel_size=3, stride=1, use_se=False, act_layer="hswish", l2_reg=l2_reg)
    
    #x = layers.MaxPool2D(pool_size=(2,1))(x)
    x = layers.AvgPool2D(pool_size=(8,1))(x)

    return x

def _make_divisible(v, divisor, min_value=None):
    """https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def MobileNetV3_2(input, num_classes=1001, width_multiplier=1.0, divisible_by=8, l2_reg=1e-5, training=True):
    # MobileNetV3-Large
    x = conv_norm_act(input, 16, kernel_size=3, stride=1, padding=1, norm_layer="bn", act_layer="hswish", use_bias=False, l2_reg=l2_reg)
    
    bneck_settings = [
            # k   exp   out   SE      NL         s
            [ 3,  16,   16,   True,  "relu",    1 ],
            [ 3,  72,   24,   False,  "relu",    2 ],
            [ 3,  88,   24,   False,  "relu",    1 ],
            [ 5,  96,   40,   True,   "hswish",    1 ],
            [ 5,  240,  40,   True,   "hswish",    1 ],
            #[ 5,  240,  40,   True,   "hswish",    1 ],
            [ 3,  120,  48,   True,  "hswish",  2 ],
            [ 3,  144,  48,   True,  "hswish",  1 ],
            [ 5,  288,  96,   True,  "hswish",  1 ],
            [ 3,  576,  96,   True,  "hswish",  1 ],
            [ 3,  576,  96,   True,   "hswish",  1 ]
        ]
    for idx, (k, exp, out, SE, NL, s) in enumerate(bneck_settings):
            out_channels = _make_divisible(out * width_multiplier, divisible_by)
            exp_channels = _make_divisible(exp * width_multiplier, divisible_by)
            x = Bneck(x, out_channels=out_channels, exp_channels=exp_channels, kernel_size=k, stride=s, use_se=SE, act_layer=NL)
    
    penultimate_channels = _make_divisible(576 * width_multiplier, divisible_by)
    last_channels = _make_divisible(576 * width_multiplier, divisible_by)
    x = conv_norm_act(x, penultimate_channels, kernel_size=1, stride=1, norm_layer="bn", act_layer="hswish", use_bias=False, l2_reg=l2_reg)
    
    x = layers.AvgPool2D(pool_size=(8,1))(x)
    x = conv_norm_act(x, last_channels, kernel_size=1, norm_layer=None, act_layer="hswish", l2_reg=l2_reg)
    return x

class CRNNCTCNetwork(object):
    def __init__(self, imgH, imgW, max_label_length, class_number, training=True, lr=0.0001, decay=0):
        self.imgH=imgH  
        self.imgW=imgW  
        self.max_label_length=max_label_length  
        self.class_number=class_number  
        self.training=training  
        self.lr=lr  
        self.decay=decay  
        self.filters = [16, 32, 48, 64, 128]
        self.length = self.imgW // 4# - 3

    def feature_extraction2(self, inputs, training=True):
        l2_reg = 1e-4
        x = layers.Conv2D(self.filters[0], (3,3), padding='same', kernel_regularizer=regularizers.l2(l2_reg))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(self.filters[1], (3,3), padding='same', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(self.filters[1], (3,3), padding='same', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(self.filters[2], (3,3), padding='same', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x2 = layers.Activation('relu')(x)

        x = layers.Conv2D(self.filters[2] // 2, (1,1), padding='same', kernel_regularizer=regularizers.l2(l2_reg))(x2)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(self.filters[2] // 2, (3,3), padding='same', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(self.filters[2], (1,1), padding='same', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Add()([x, x2])

        x = layers.Conv2D(self.filters[2], (3,3), padding='same', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(self.filters[2], (3,3), padding='same', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2,1))(x)

        x = layers.Conv2D(self.filters[3], (3,3), padding='same', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(self.filters[3], (3,3), padding='same', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2,1))(x)

        k = 5
        x = layers.ZeroPadding2D(padding=(0,k // 2))(x)
        x = layers.Conv2D(self.filters[4], (2,k), padding='valid', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Reshape(target_shape=(self.length, self.filters[4]))(x)
        if training:
            x = layers.Dropout(0.25)(x)
        x = layers.Dense(self.class_number)(x)

        return x

    def sequence_label(self, inputs):
        # normalize is important
        query = layers.Dense(inputs.shape[-1], activation=None, name="query")(inputs)
        key = layers.Dense(inputs.shape[-1], activation=None, name="key")(inputs)
        value = layers.Dense(inputs.shape[-1], activation=None, name="value")(inputs)
        logits = layers.Dot(axes=2)([query, key])
        logits = layers.Activation('softmax', name='att_logits')(logits)
        logits = layers.Dot(axes=1)([logits, value])
        
        return logits

    def feature_extraction3(self, inputs, training=True):
        x = MobilenetV3(inputs)
        
        x = layers.Reshape(target_shape=(self.length, x.shape[-1]))(x)

        #x = layers.recurrent.LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(x)
        # RNN layer
        #inner = layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(x)  # (None, 32, 64)
        #lstm_1 = layers.recurrent.LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 32, 256)
        #lstm_1b = layers.recurrent.LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)  # (None, 32, 256)
        #reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_1b)  # (None, 32, 256)
        #lstm1_merged = layers.merge.add([lstm_1, reversed_lstm_1b])  # (None, 32, 256)
        #lstm1_merged = BatchNormalization()(lstm1_merged)  # (None, 32, 256)
        #lstm_2 = layers.recurrent.LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)  # (None, 32, 256)
        #lstm_2b = layers.recurrent.LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)  # (None, 32, 256)
        #reversed_lstm_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_2b)  # (None, 32, 256)
        #lstm2_merged = layers.merge.concatenate([lstm_2, reversed_lstm_2b])  # (None, 32, 512)
        #x = BatchNormalization()(lstm2_merged)

        #x = self.sequence_label(x)

        if training:
            x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.class_number)(x)
        return x

    def feature_extraction(self, inputs, training=True):
        x = MobileNetV3_2(inputs, width_multiplier=0.4, training=training)
        x = layers.Reshape(target_shape=(self.length, x.shape[-1]))(x)
        if training:
            x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.class_number)(x)
        return x

    def feature_extraction_forTVM(self, inputs, training=True):
        x = MobileNetV3_2(inputs, width_multiplier=0.4, training=training)
        if training:
            x = layers.Dropout(0.5)(x)
        x = layers.Conv2D(self.class_number, (1,1))(x)
        x = layers.Reshape(target_shape=(self.length, x.shape[-1]))(x)
        return x

    def build_network(self, training=True):
        input_shape = (self.imgH, self.imgW, 3)
        inputs = layers.Input(name='inputs', shape=input_shape, dtype='float32')
        x = self.feature_extraction3(inputs, training)
        
        y_pred = layers.Activation('softmax', name='y_pred')(x)
        labels = layers.Input(name='labels', shape=[self.max_label_length], dtype='float32')
        input_length = layers.Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
        label_length = layers.Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)

        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
        loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        if training:
            self.model =  models.Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
        else:
            y_pred = layers.Lambda(K.argmax, name='y_pred2')(y_pred)
            self.model =  models.Model(inputs=[inputs], outputs=y_pred)
            #self.model =  models.Model(inputs=[inputs], outputs=x)

    def build_loss(self):
        self.model.compile(optimizer=optimizers.Adam(lr=self.lr, decay=self.decay), loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['acc']) 
        #self.model.compile(optimizer=optimizers.RMSprop(self.lr), loss={'ctc': lambda y_true, y_pred: y_pred}) 

def test_create_network():
    net = CRNNCTCNetwork(32, 64, 4, 11)
    net.build_network()
    net.model.summary()
