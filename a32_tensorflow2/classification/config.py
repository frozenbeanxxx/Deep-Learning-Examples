image_weight = 48
image_height = 48
batch_size = 64
learning_rate = 0.0004
decay=1e-8
epochs = 200


train_data_dir = '/home/weixing/data/other/CatAndDog/PetImages'
val_data_dir = '/home/weixing/data/other/CatAndDog/PetImages'
test_data_dir = '/home/weixing/data/other/CatAndDog/PetImages'
log_dir = './log/'
model_dir = './models'
model_name = 'model.h5'
weight_name = 'weights.h5'
tflite_name = 'model.tflite'