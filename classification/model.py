from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential, Model, Input
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras import regularizers 

def create_model(h, w, is_training=False):
    inputs = Input(shape=(h, w, 3))
    x = Conv2D(16, (3,3), padding="same", strides=(2,2), input_shape=(h, w, 3), 
                        kernel_regularizer=regularizers.l2(0.01),
                        bias_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01))(inputs)  # (24, 8, 16)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) # (12, 4, 16)

    x = Conv2D(24, (3, 3), padding="same", )(x) # (12, 4, 32)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) # (6, 2, 32)

    x = Conv2D(32, (3, 3), padding="same")(x) # (6, 2, 32)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) # (3, 1, 32)

    x = Conv2D(32, (3, 3), padding="same")(x) # (6, 2, 32)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Flatten()(x) # (96,)
    if is_training:
        x = Dropout(0.5)(x)
    x = Dense(64)(x) # (64,)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if is_training:
        x = Dropout(0.25)(x)
    logits = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=logits)
    return model 

if __name__ == "__main__":
    model = create_model(16, 48)
    model.summary()