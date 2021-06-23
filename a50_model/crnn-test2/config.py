from keras import backend as K

# params
MAX_LEN_TEXT = 32
IMAGE_SIZE = (128, 32)
IMG_W, IMG_H = IMAGE_SIZE
NO_CHANNELS = 1
print(K.image_data_format())
if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (NO_CHANNELS, IMG_W, IMG_H)
else:
    INPUT_SHAPE = (IMG_W, IMG_H, NO_CHANNELS)

BATCH_SIZE = 256
CONV_FILTERS = 16
KERNEL_SIZE = (3, 3)
POOL_SIZE = 2
DOWNSAMPLE_FACTOR = POOL_SIZE ** 2
TIME_DENSE_SIZE = 32
RNN_SIZE = 512
NO_LABELS = 80
NO_EPOCHS = 2
INITIAL_EPOCH = 12
DATASET_SPLIT = 4

# path
#WORDS_FOLDER = '/home/weixing/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px'
#WORDS_DATA = '/home/weixing/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px/annotation_val.txt'
#WORDS_TRAIN = '/home/weixing/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px/annotation_train.txt'
#WORDS_TEST = '/home/weixing/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px/annotation_val.txt'
CONFIG_MODEL = './logs/model.json'
MODEL = './logs/model.h5'
WEIGHT_MODEL = './logs/weights.h5'
MODEL_CHECKPOINT = './logs'
LOGGING = './logs'

WORDS_FOLDER = 'D:/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px'
WORDS_DATA = 'D:/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px/annotation_val.txt'
WORDS_TRAIN = 'D:/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px/annotation_train.txt'
WORDS_TEST = 'D:/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px/annotation_val.txt'


