import os
import itertools

import cv2
from tqdm import tqdm
import numpy as np
import keras
from keras import backend as K

import config as cf
from utils import preprocess
#from src.log import get_logger

#logger = get_logger(__name__)

chars = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'  # noqa


def labels_to_text(letters, labels):
    return ''.join(list(map(lambda x: letters[x] if x < len(letters) else "", labels)))  # noqa


def text_to_labels(letters, text):
    return list(map(lambda x: letters.index(x), text))


def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret


def decode_predict_ctc(out, chars, top_paths=1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        lables = K.get_value(
            K.ctc_decode(
                out, input_length=np.ones(out.shape[0]) * out.shape[1],
                greedy=False, beam_width=beam_width, top_paths=top_paths
            )[0][i]
        )[0]
        text = labels_to_text(chars, lables)
        results.append(text)
    return results


def predit_a_image(model_p, pimg, top_paths=1):
    # c = np.expand_dims(a.T, axis=0)
    net_out_value = model_p.predict(pimg)
    top_pred_texts = decode_predict_ctc(net_out_value, top_paths)
    return top_pred_texts


def is_valid_str(letters, s):
    for ch in s:
        if ch not in letters:
            return False
    return True


def head(stream, n=10):
    """
    Return the first `n` elements of the stream, as plain list.
    """
    return list(itertools.islice(stream, n))


class TextSequenceGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, fn_path, batch_size=16,
                 img_size=(128, 32), max_text_len=32,
                 downsample_factor=4,
                 shuffle=True, data_aug=True):
        self.fn_path = fn_path
        self.max_text_len = max_text_len
        self.samples, self.chars = self.__get_metadata()
        #logger.info("Len samples: %d", len(self.samples))
        #logger.info("Len chars: %d", len(self.chars))

        self.blank_label = len(self.chars)
        self.ids = range(len(self.samples))

        self.img_size = img_size
        self.img_w, self.img_h = self.img_size
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor

        self.shuffle = shuffle
        self.data_aug = data_aug

        self.on_epoch_end()

    def __get_metadata(self):
        samples = []
        #chars = set()
        with open(self.fn_path) as f:
            lines = f.readlines()
            print(len(lines))
        n = len(lines) // cf.DATASET_SPLIT
        lines = lines[:n]
        for line in tqdm(lines, desc='readlines'):
            line = line.rstrip().strip()
            line = line.split(" ")[0]
            img_path = cf.WORDS_FOLDER + "/" + line
            gt_text = img_path.split("/")[-1].split("_")[-2]
            #chars = chars.union(set(list(gt_text)))

            samples.append([img_path, gt_text])

        #chars = sorted(list(chars))
        print(chars)
        return samples, chars

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        ids = [self.ids[k] for k in indexes]

        X, y = self.__data_generation(ids)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids):
        """Generates data containing batch_size samples"""
        size = len(ids)

        if K.image_data_format() == 'channels_first':
            X = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X = np.ones([size, self.img_w, self.img_h, 1])
        Y = np.zeros([size, self.max_text_len])
        input_length = np.ones((size, 1), dtype=np.float32) * \
            (self.img_w // self.downsample_factor - 2)
        label_length = np.zeros((size, 1), dtype=np.float32)

        # Generate data
        for i, id_ in enumerate(ids):
            img = preprocess(
                cv2.imread(self.samples[id_][0], cv2.IMREAD_GRAYSCALE),
                self.img_size, self.data_aug
            )
            if K.image_data_format() == 'channels_first':
                img = np.expand_dims(img, 0)
            else:
                img = np.expand_dims(img, -1)

            X[i] = img
            len_text = len(self.samples[id_][1])
            Y[i, :len_text] = \
                text_to_labels(self.chars, self.samples[id_][1])
            label_length[i] = len_text

        inputs = {
            'the_input': X,  # (bs, 128, 32, 1)
            'the_labels': Y,  # (bs, max_text_len) ~ (bs, 32)
            'input_length': input_length,  # (bs, 1)
            'label_length': label_length,  # (bs, 1)
        }
        outputs = {'ctc': np.zeros([size])}  # (bs, 1)

        return inputs, outputs
