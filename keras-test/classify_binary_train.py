import sys
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D, Reshape
from tensorflow.keras import callbacks


DEV = False
argvs = sys.argv
argc = len(argvs)
print('argvs:', argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
    DEV = True

if DEV:
    epochs = 1
else:
    epochs = 20

train_data_dir = 'D:\\dataset\\hv_royale\\InGame20190416'
validation_data_dir = 'D:\\dataset\\hv_royale\\InGame20190416'

img_width, img_height = 16, 48
batch_size = 64
lr = 0.0004
decay = 1e-8

def main():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding="same", strides=(2,2), input_shape=(img_height, img_width, 3))) # (24, 8, 16)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2))) # (12, 4, 16)

    model.add(Conv2D(32, (3, 3), padding="same")) # (12, 4, 32)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2))) # (6, 2, 32)

    model.add(Conv2D(32, (3, 3), padding="same")) # (6, 2, 32)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2))) # (3, 1, 32)

    model.add(Reshape(target_shape=((8, 12)), name='reshape'))
    model.add(Dense(12))
    model.add(Flatten()) # (96,)
    model.add(Dense(64)) # (64,)
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(lr=lr),
                metrics=['accuracy'])

    def convert_color_channels(image):
        return image[:,:,::-1]

    train_datagen = ImageDataGenerator(
        preprocessing_function=convert_color_channels,
        rescale=1. / 255,
        width_shift_range=6,
        height_shift_range=2,
        horizontal_flip=False,
        vertical_flip=False)

    test_datagen = ImageDataGenerator(
        preprocessing_function=convert_color_channels,
        rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    """
    Tensorboard log
    """
    log_dir = './log/'
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,write_graph=True)
    cbks = [tb_cb]

    model.fit_generator(
        train_generator,
        epochs=epochs,
        #validation_data=validation_generator,
        callbacks=cbks)

    target_dir = './models/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    model.save('./models/model.h5')
    model.save_weights('./models/weights.h5')

if __name__ == "__main__":
    main()

