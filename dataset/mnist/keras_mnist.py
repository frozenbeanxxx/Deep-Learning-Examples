# encoding: utf-8
import numpy as np 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import callbacks
import os


def mnist_load_data():
    file_dir = "D:\\dataset\\mnist"
    train_image_file = "train-images.idx3-ubyte"
    train_label_file = "train-labels.idx1-ubyte"
    test_image_file = "t10k-images.idx3-ubyte"
    test_label_file = "t10k-labels.idx1-ubyte"

    # decode train-images.idx3-ubyte
    with open(file_dir+'/'+train_image_file, 'rb') as f:
        train_image = f.read()
    magic = np.frombuffer(train_image[:4], dtype=np.dtype(np.uint32).newbyteorder('>'))[0]
    num_images = np.frombuffer(train_image[4:8], dtype=np.dtype(np.uint32).newbyteorder('>'))[0]
    num_rows = np.frombuffer(train_image[8:12], dtype=np.dtype(np.uint32).newbyteorder('>'))[0]
    num_cols = np.frombuffer(train_image[12:16], dtype=np.dtype(np.uint32).newbyteorder('>'))[0]
    train_image = np.frombuffer(train_image[16:], dtype=np.uint8)
    train_image = train_image.reshape(num_images, num_rows, num_cols, 1)

    # decode train-labels.idx1-ubyte
    with open(file_dir+'/'+train_label_file, 'rb') as f:
        train_label = f.read()
    magic = np.frombuffer(train_label[:4], dtype=np.dtype(np.uint32).newbyteorder('>'))
    num_images = np.frombuffer(train_label[4:8], dtype=np.dtype(np.uint32).newbyteorder('>'))
    train_label = np.frombuffer(train_label[8:], dtype=np.uint8)

    # decode t10k-images.idx3-ubyte
    with open(file_dir+'/'+test_image_file, 'rb') as f:
        test_image = f.read()
    magic = np.frombuffer(test_image[:4], dtype=np.dtype(np.uint32).newbyteorder('>'))[0]
    num_images = np.frombuffer(test_image[4:8], dtype=np.dtype(np.uint32).newbyteorder('>'))[0]
    num_rows = np.frombuffer(test_image[8:12], dtype=np.dtype(np.uint32).newbyteorder('>'))[0]
    num_cols = np.frombuffer(test_image[12:16], dtype=np.dtype(np.uint32).newbyteorder('>'))[0]
    test_image = np.frombuffer(test_image[16:], dtype=np.uint8)
    test_image = test_image.reshape(num_images, num_rows, num_cols, 1)

    # decode train-labels.idx1-ubyte
    with open(file_dir+'/'+test_label_file, 'rb') as f:
        test_label = f.read()
    magic = np.frombuffer(test_label[:4], dtype=np.dtype(np.uint32).newbyteorder('>'))
    num_images = np.frombuffer(test_label[4:8], dtype=np.dtype(np.uint32).newbyteorder('>'))
    test_label = np.frombuffer(test_label[8:], dtype=np.uint8)

    return (train_image,train_label),(test_image,test_label)

np.random.seed(1671)

NB_EPOCH = 100
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
#OPTIMIZER = SGD()
OPTIMIZER = Adam(lr=0.0004)
N_HIDDEN = 128
VALIDATION_SPLIT = 0.05

(X_train, y_train), (X_test, y_test) = mnist_load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

RESHAPED = 784
#X_train = X_train.reshape(60000, RESHAPED)
#X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

def model1():
    model = Sequential()
    model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
    model.add(Activation('softmax'))
    return model

def model2():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, padding='same', strides=(2,2), input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    return model

model = model2()

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

cb = callbacks.TensorBoard(log_dir='./log', write_images=True, write_grads=True)
cbks = [cb]
model.fit(X_train, Y_train,
        batch_size=BATCH_SIZE, epochs=NB_EPOCH,
        verbose=VERBOSE, validation_split=VALIDATION_SPLIT,
        callbacks=cbks)
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print('\nTest score:', score[0])
print('Test accuracy:', score[1])

model_dir = './models'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model.save(model_dir+'/model.h5')