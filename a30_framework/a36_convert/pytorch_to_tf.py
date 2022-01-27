import tensorflow as tf
from tensorflow import keras

import numpy as np

def aa():
    tf_bn = keras.Sequential([keras.layers.UpSampling2D()])
    tf_bn = keras.Sequential([keras.layers.Conv2DTranspose()])
    tf.image.resize_bilinear()
    tf.lite.TFLiteConverter.from_keras_model()
    tf.lite.TFLiteConverter.from_keras_model_file()

if __name__ == '__main__':
    image = np.random.rand(1, 5, 5, 3)  # [N, H, W, C]
    tf_image = image.astype(np.float32)
    tf_bn = keras.Sequential([keras.layers.BatchNormalization(epsilon=1e-5, input_shape=(5, 5, 3))])
    bn_weight = np.array([
        np.array([1, 1, 1], dtype=np.float),
        np.array([2, 2, 2], dtype=np.float),
        np.array([3, 3, 3], dtype=np.float),
        np.array([4, 4, 4], dtype=np.float),
    ]).astype(np.float32)

    tf_bn.layers[0].set_weights(bn_weight)

    print(tf_bn.layers[0].get_weights())
    print(bn_weight)

    print('over')