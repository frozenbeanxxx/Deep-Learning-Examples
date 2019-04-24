import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import SGD

import config as cf
from data_generator import TextSequenceGenerator
from models import CRNN_model
#from src.log import get_logger

#logger = get_logger(__name__)


def train():

    train_set = TextSequenceGenerator(
        cf.WORDS_TRAIN,
        img_size=cf.IMAGE_SIZE, max_text_len=cf.MAX_LEN_TEXT,
        downsample_factor=cf.DOWNSAMPLE_FACTOR
    )
    test_set = TextSequenceGenerator(
        cf.WORDS_TEST,
        img_size=cf.IMAGE_SIZE, max_text_len=cf.MAX_LEN_TEXT,
        downsample_factor=cf.DOWNSAMPLE_FACTOR,
        shuffle=False, data_aug=False
    )

    no_train_set = len(train_set.ids)
    no_val_set = len(test_set.ids)
    #logger.info("No train set: %d", no_train_set)
    #logger.info("No val set: %d", no_val_set)

    model, y_func = CRNN_model()
    
    try:
        model.load_weights('./logs/weights.h5')
        print("...Previous weight data...")
    except:
        print("...New weight data...")
        pass

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    log = TensorBoard(cf.LOGGING)
    ckp = ModelCheckpoint(
        cf.MODEL_CHECKPOINT, monitor='val_loss',
        verbose=1, save_best_only=True, save_weights_only=True, period=1)
    #earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')

    model.fit_generator(generator=train_set,
                        steps_per_epoch=no_train_set // cf.BATCH_SIZE,
                        epochs=cf.NO_EPOCHS,
                        validation_data=test_set,
                        validation_steps=no_val_set // cf.BATCH_SIZE,
                        callbacks=[log])
                        #initial_epoch=cf.INITIAL_EPOCH)

    return model, y_func


if __name__ == '__main__':
    model, test_func = train()

    model_json = model.to_json()
    with open(cf.CONFIG_MODEL, 'w') as f:
        f.write(model_json)
    
    model.save_weights(cf.WEIGHT_MODEL)
