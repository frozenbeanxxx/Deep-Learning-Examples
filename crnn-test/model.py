from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
K.set_learning_phase(0)

CHAR_VECTOR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'_&.!?,\""
num_classes = len(CHAR_VECTOR) + 2
#letters = [letter for letter in CHAR_VECTOR]
letters_dict = {letter:i+1 for i, letter in enumerate(CHAR_VECTOR)}
img_w, img_h, img_c = 100, 32, 3
max_text_len = 25
print(num_classes)
#print(letters)
#print(letters_dict)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def create_model(training):
    # create crnn model
    input_shape = (img_w, img_h, img_c)
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    # CNN convolution layer
    x = Conv2D(64, (3,3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), name='max1')(x)

    x = Conv2D(128, (3,3), padding='same', name='conv2', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), name='max2')(x)

    x = Conv2D(256, (3,3), padding='same', name='conv3', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3,3), padding='same', name='conv4', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(1,2), name='max3')(x)

    x = Conv2D(512, (3,3), padding='same', name='conv5', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3,3), padding='same', name='conv6', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(1,2), name='max4')(x)

    x = Conv2D(512, (2,2), padding='same', name='conv7', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Reshape(target_shape=((max_text_len, 1024)), name='inner')(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(x)
    
    # RNN
    lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(x)
    lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1b')(x)
    reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)

    lstm_1_merged = add([lstm_1, reversed_lstm_1b])
    lstm_1_merged = BatchNormalization()(lstm_1_merged)

    lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm_1_merged)
    lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2b')(lstm_1_merged)
    reversed_lstm_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_2b)
    
    lstm_2_merged = add([lstm_2, reversed_lstm_2b])
    lstm_2_merged = BatchNormalization()(lstm_2_merged)

    x = Dense(num_classes, kernel_initializer='he_normal',name='dense2')(lstm_2_merged) #(None, 32, 42)
    y_pred = Activation('softmax', name='softmax')(x) 

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32') # (None ,8)
    input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)

    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    else:
        return Model(inputs=[inputs], outputs=y_pred)

if __name__ == "__main__":
    model = create_model(training=True)
    #model.summary()