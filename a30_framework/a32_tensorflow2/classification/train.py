import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from config import *
from model import create_model

import tensorflow as tf 
gpus= tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def train_model():
    os.makedirs(model_dir, exist_ok=True)
    model = create_model(image_height, image_weight, is_training=True)
    model.summary()
    try:
        model.load_weights(os.path.join(model_dir, weight_name))
        print("...Previous weight data...")
    except:
        print("...New weight data...")

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(lr=learning_rate, decay=decay),
                metrics=['accuracy'])
    
    def convert_color_channels(image):
        return image[:,:,::-1]

    train_datagen = ImageDataGenerator(
        preprocessing_function=convert_color_channels,
        rescale=1. / 255,
        width_shift_range=0.1,
        height_shift_range=0.2,
        horizontal_flip=False,
        vertical_flip=False)

    val_datagen = ImageDataGenerator(
        preprocessing_function=convert_color_channels,
        rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_weight),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(image_height, image_weight),
        batch_size=batch_size,
        class_mode='categorical')

    weight_path = os.path.join(model_dir, weight_name)
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
    cb_cp= callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    cbks = [tb_cb, cb_cp]

    model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=cbks)

    model.save(os.path.join(model_dir, model_name))

if __name__ == "__main__":
    train_model()