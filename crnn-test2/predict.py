import pickle

import numpy as np
from keras.models import Model, model_from_json
from models import CRNN_model
import config as cf
from data_generator import (
    TextSequenceGenerator,
    decode_predict_ctc,
    labels_to_text,
    chars
)

#from src.log import get_logger

#logger = get_logger(__name__)
chars_ = [char for char in chars]


def load_trained_model():
    model, y_func = CRNN_model()
    model.load_weights(cf.WEIGHT_MODEL)

    return model

def load_trained_model2():
    with open(cf.CONFIG_MODEL) as f:
        json_string = f.read()

    model = model_from_json(json_string)
    model.load_weights(cf.WEIGHT_MODEL)

    return model


def load_test_samples():
    test_set = TextSequenceGenerator(
        cf.WORDS_TEST,
        img_size=cf.IMAGE_SIZE, max_text_len=cf.MAX_LEN_TEXT,
        downsample_factor=cf.DOWNSAMPLE_FACTOR,
        shuffle=False,
        data_aug=False
    )
    return test_set

def labels_to_text2(letters, labels):
    return ''.join(list(map(lambda x: letters[x], labels))) 


def predict(model_p, test_set, index_batch, index_img):
    

    samples = test_set[index_batch]
    img = samples[0]['the_input'][index_img]
    img = np.expand_dims(img, axis=0)
    #logger.info(img.shape)

    net_out_value = model_p.predict(img)
    #print('gt_texts: ', np.argmax(net_out_value[0], axis=1), net_out_value.shape)
    #logger.info(net_out_value.shape)
    pred_texts = decode_predict_ctc(net_out_value, chars_)
    #logger.info(pred_texts)
    gt_texts = test_set[index_batch][0]['the_labels'][index_img]
    #print('gt_texts: ', gt_texts.astype(int))
    gt_texts = labels_to_text(chars_, gt_texts.astype(int))
    #logger.info(gt_texts)
    print('gt_texts: ', gt_texts, ', pred_texts: ', pred_texts)


if __name__ == '__main__':
    import random  # noqa
    rd_index_batch = 0#random.randint(0, 10)
    test_set = load_test_samples()
    print(chars_)
    model = load_trained_model()
    input_data = model.get_layer('the_input').output
    y_pred = model.get_layer('softmax').output
    model_p = Model(inputs=input_data, outputs=y_pred)

    for i in range(cf.BATCH_SIZE):
        predict(model_p, test_set, rd_index_batch, i)
