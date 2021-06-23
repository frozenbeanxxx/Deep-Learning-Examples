import cv2
import os
from tqdm import tqdm
import random
import numpy as np
from model import *
from keras.preprocessing.sequence import pad_sequences

class TextImageGenerator:
    def _create_dataset_from_file(self, annotation_dir, annotation):
        with open(os.path.join(annotation_dir, annotation), "r") as f:
            readlines = f.readlines()

        readlines = readlines[:10000]
        img_paths = []
        for img_name in tqdm(readlines, desc="read dir:"):
            img_name = img_name.rstrip().strip()
            img_name = img_name.split(" ")[0]
            img_path = annotation_dir + "/" + img_name
            if os.path.exists(img_path):
                img_paths.append(img_path)
            else:
                print(img_path, ' is not exists')
        labels = [img_path.split("/")[-1].split("_")[-2] for img_path in tqdm(img_paths, desc="generator label:")]
        return img_paths, labels

    def __init__(self, annotation_dir, annotation, img_w, img_h, img_c, 
                 batch_size, max_text_len=25):
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.annotation_dir = annotation_dir  
        self.annotation = annotation 
        self.cur_index = 0 
        self.image_paths, self.labels = self._create_dataset_from_file(self.annotation_dir, self.annotation)
        self.n = len(self.image_paths)

    def next_sample(self):  
        self.cur_index += 1
        if self.cur_index >= len(self.image_paths):
            self.cur_index = 0
            random.shuffle(self.image_paths)
        image_path = self.image_paths[self.cur_index]
        label_str = image_path.split("/")[-1].split("_")[-2]
        if self.img_c == 3:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (self.img_w, self.img_h))
        else:
            img = cv2.imread(image_path, 0)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = np.expand_dims(img, axis=-1)
            
        img = img.astype(np.float32)
        img = img /255.0
        
        return img, label_str

    def text_to_labels(self, text):  
        a = [[letters_dict[letter] for letter in text]]
        a = pad_sequences(a, maxlen=self.max_text_len, padding='post')
        return a[0]

    def next_batch(self):
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, self.img_c])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.max_text_len - 2)
            label_length = np.zeros((self.batch_size, 1))
            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.transpose(1,0,2)
                #img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = self.text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,  
                'input_length': input_length, 
                'label_length': label_length 
            }
            outputs = {'ctc': np.zeros([self.batch_size])} 
            yield (inputs, outputs)
''''''

if __name__ == "__main__":
    #annotation_dir = 'D:\\dataset\\mjsynth\\mnt\\ramdisk\\max\\90kDICT32px'
    annotation_dir = '/home/weixing/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px'
    #annotation = 'annotation_val.txt'
    annotation = 'annotation_train.txt'
    batch_size = 2
    data_train = TextImageGenerator(annotation_dir, annotation, img_w, img_h, img_c, batch_size)
    #print(data_train.image_paths, data_train.labels)
    print(len(data_train.image_paths))
    for i in range(0):
        img, label = data_train.next_sample()
        print(label)
        cv2.imshow('win', img)
        cv2.waitKey(0)

    iter1 = data_train.next_batch()
    for i in range(3):
        inputs, outputs = next(iter1)
        print(inputs['the_input'].shape)
        print(inputs['the_labels'])
        print(inputs['input_length'])
        print(inputs['label_length'])
        print(outputs)
            