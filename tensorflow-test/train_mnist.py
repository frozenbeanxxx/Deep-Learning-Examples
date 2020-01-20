import os
import fire
import cv2 
import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras import layers

def test1():
    print(tf.__version__)

def train():
    epochs = 1
    batch_size = 1024
    imgH = 28
    imgW = 28
    imgC = 1
    checkpoint_path = 'models_train_mnist'
    os.makedirs(checkpoint_path, exist_ok=True)
    mnist_file_dir = 'E:/dataset/mnist'
    mnist = input_data.read_data_sets(mnist_file_dir, one_hot=True)
    train_data = mnist.train.images
    num_train_images = mnist.train.images.shape[0]
    num_batch = num_train_images // batch_size
    print(train_data.shape)
    train_data_images = np.reshape(train_data, (train_data.shape[0], imgH, imgW, imgC))
    train_data_labels = mnist.train.labels
    print(train_data.shape)

    images_placeholder = tf.placeholder(tf.float32, shape=[None, imgH, imgW, imgC])
    labels_placeholder = tf.placeholder(tf.int64, shape=[None, 10])

    x = layers.Conv2D(16, (3,3), padding='same')(images_placeholder)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    logits = layers.Dense(10, activation=None)(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder))
    op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    #init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #try:
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        #except ValueError:
        #    print('load weights from initialize')
        #sess.run(init)
        for step in range(epochs):
            loss_mean = 0
            for batch in range(num_batch):
                index = np.random.randint(0, num_train_images, size=(batch_size))
                images = train_data_images[index]
                labels = train_data_labels[index]
                _, loss_once = sess.run([op, loss], feed_dict={images_placeholder:images, labels_placeholder:labels})
                loss_mean += loss_once
            print(f"step: {step},\t loss: {loss_mean/num_batch}")

        #saver.save(sess, checkpoint_path+"/model.ckpt", global_step=epochs)
    


if __name__ == "__main__":
    train()