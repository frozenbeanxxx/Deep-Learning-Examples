import os 
import cv2 
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.utils.data_utils import Sequence
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
from keras.engine.training_utils import iter_sequence_infinite
import xml.etree.ElementTree as XET

import platform

sysstr = platform.system()
if(sysstr =="Windows"):
    user_root_path = 'D:'
    print ("Call Windows tasks")
elif(sysstr == "Linux"):
    user_root_path = '/home/weixing'
    print ("Call Linux tasks")
else:
    print ("Other System tasks")

classes = {'aeroplane':0,'bicycle':1,'bird':2,'boat':3,'bottle':4,
            'bus':5,'car':6,'cat':7,'chair':8,'cow':9,
            'diningtable':10,'dog':11,'horse':12,'motorbike':13,'person':14,
            'pottedplant':15,'sheep':16,'sofa':17,'train':18,'tvmonitor':19}

def read_images(path, rect=None, dim=(224,224)):
    image = cv2.imread(path)
    if rect :
        image = image[rect[1]:rect[3], rect[0]:rect[2]]
        #print(image.shape)
    image = cv2.resize(image, dim)
    return image

def read_index(index, dim=(224,224)):
    f2 = open(os.path.join(voc_root_dir, years_dir_name, index_dir_name, index), 'r')
    line = f2.read()
    #print(line)
    strings = line.split(",")
    f2.close()
    image_path = os.path.join(images_path, strings[0])
    #print(image_path)
    rect = (int(strings[2]),int(strings[3]),int(strings[4]),int(strings[5]))
    image = read_images(image_path, rect=rect, dim=dim)
    return image, strings[1]

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels=None, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8)
        strX = []
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            strX.append(ID) #np.load('data/' + ID + '.npy')
            image, class_name = read_index(ID, self.dim)
            #print(image)
            # Store class
            X[i,] = image
            X[i,] = X[i,] / 255
            #print(X[i,])
            y[i] = self.labels[class_name]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


voc_root_dir = user_root_path + '/dataset/voc/VOCdevkit'
years_dir_name = 'VOC2007'
annotations_dir_name = 'Annotations'
images_dir_name = 'JPEGImages'
index_dir_name = 'index'
annotations_path = os.path.join(voc_root_dir, years_dir_name, annotations_dir_name)
images_path = os.path.join(voc_root_dir, years_dir_name, images_dir_name)
annotations = os.listdir(annotations_path)
index_path = os.path.join(voc_root_dir, years_dir_name, index_dir_name)
index_files = os.listdir(index_path)
random.shuffle(annotations)
train_images = list(map(lambda x: x.replace('xml', 'jpg'), annotations[:100]))
train_labels = annotations#[:100]
#train_labels = {image:(label+1)%10 for label, image in enumerate(train_images) }

def process_voc_labels():
    ii = 0
    for label in train_labels:
        if ii % 50 == 0:
            print('process : ', ii)
        ii += 1
        label_path = os.path.join(voc_root_dir, years_dir_name, annotations_dir_name, label)
        tree = XET.parse(label_path)
        root = tree.getroot()
        filename = root.find('filename').text
        count = 0
        for obj in root.iter('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')   
            xmin = bndbox.find('xmin').text
            xmax = bndbox.find('xmax').text
            ymin = bndbox.find('ymin').text
            ymax = bndbox.find('ymax').text
            add_str = ","
            new_line = filename + add_str + name + \
                        add_str + xmin + add_str + ymin + \
                        add_str + xmax + add_str + ymax
            #print(new_line)
            f2 = open(os.path.join(voc_root_dir, years_dir_name, index_dir_name, label[:-4]+'_'+str(count)+'.txt'), 'w')
            f2.write(new_line)
            f2.close()
            count += 1

#process_voc_labels()

def test_read_image():
    
    f2 = open(os.path.join(voc_root_dir, years_dir_name, index_dir_name, index_files[0]), 'r')
    line = f2.read()
    print(line)
    strings = line.split(",")
    f2.close()
    
    #image_path = os.path.join(images_path, train_images[0])
    image_path = os.path.join(images_path, strings[0])
    print(image_path)
    rect = (int(strings[2]),int(strings[3]),int(strings[4]),int(strings[5]))
    image = read_images(image_path, rect=rect)
    
#test_read_image()


# Parameters
params = {'dim': (224,224),
          'batch_size': 1,
          'n_classes': 20,
          'n_channels': 3,
          'shuffle': True}

# Datasets
partition = {'train': index_files} # IDs
labels = classes # Labels

# Generators
training_generator_test = DataGenerator(partition['train'], labels, **params)
#validation_generator = DataGenerator(partition['validation'], labels, **params)

#is_sequence = isinstance(training_generator, Sequence)
#print("is_sequence : ", is_sequence)
#print("training_generator length : ", len(training_generator))
#print("training_generator length : ", list(training_generator))

def show_image_by_generator():
    output_generator = iter_sequence_infinite(training_generator_test)
    for i in range(10):
        X, Y = next(output_generator)
        print("generator : ", i)
        
        print(Y)
        print(X.shape)
        X = X[0]
        #print(X)
        print(X.shape)
        cv2.imshow('win', X)
        cv2.waitKey(0)


