import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"
from keras import backend as K
from keras.optimizers import Adadelta, Adam
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model
from Image_Generator import TextImageGenerator
from model import *
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.Session(config=config)
KTF.set_session(session)
K.set_learning_phase(0)

model = create_model(training=True)
try:
    model.load_weights('./logs/weights.h5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

parallel_model = multi_gpu_model(model, gpus=2)

annotation_dir = '/home/weixing/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px'
#annotation_dir = 'D:\\dataset\\mjsynth\\mnt\\ramdisk\\max\\90kDICT32px'
annotation_train = 'annotation_train.txt'
annotation_val = 'annotation_val.txt'
batch_size = 256
data_train = TextImageGenerator(annotation_dir, annotation_train, img_w, img_h, img_c, batch_size, max_text_len)
data_val = TextImageGenerator(annotation_dir, annotation_val, img_w, img_h, img_c, batch_size, max_text_len)

opt = Adadelta(lr=0.5)
parallel_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

log_dir = 'logs/'
logging = TensorBoard(log_dir=log_dir)

parallel_model.fit_generator(generator=data_train.next_batch(),
                    steps_per_epoch=int(data_train.n / batch_size),
                    epochs=20,
                    callbacks=[logging],
                    validation_data=data_val.next_batch(),
                    validation_steps=int(data_val.n / batch_size))

model.save_weights(log_dir + 'weights.h5')