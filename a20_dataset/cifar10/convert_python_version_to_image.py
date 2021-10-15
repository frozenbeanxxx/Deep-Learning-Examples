import os
import numpy as np
import cv2
import pickle


def convert_to_image():
    #srt_dir = '/media/weixing/diskD/dataset/pytorch/cifar10/cifar-10-batches-py'
    #dst_dir = '/media/weixing/diskD/dataset/pytorch/cifar10/images'
    srt_dir = '/home/weixing/temp/99_misc/cifar10/cifar-10-batches-py'
    dst_dir = '/home/weixing/temp/99_misc/cifar10/images'

    phase = 'train'
    bin_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

    for file in bin_files:
        file_path = os.path.join(srt_dir, file)
        with open(file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            labels = dict[b'labels']
            data = dict[b'data']
            imagenames = dict[b'filenames']
            for i, label in enumerate(labels):
                label_str = str(label)
                imagename = str(imagenames[i], encoding="utf-8")
                image = data[i]
                image = image.reshape((3,32,32))
                image = image.transpose((1,2,0))
                image_dir = os.path.join(dst_dir, phase, label_str)
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, imagename)
                cv2.imwrite(image_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                a = 1
    print('over train')


    phase = 'val'
    bin_files = ['test_batch']
    for file in bin_files:
        file_path = os.path.join(srt_dir, file)
        with open(file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            labels = dict[b'labels']
            data = dict[b'data']
            imagenames = dict[b'filenames']
            for i, label in enumerate(labels):
                label_str = str(label)
                imagename = str(imagenames[i], encoding="utf-8")
                image = data[i]
                image = image.reshape((3, 32, 32))
                image = image.transpose((1, 2, 0))
                image_dir = os.path.join(dst_dir, phase, label_str)
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, imagename)
                cv2.imwrite(image_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                a = 1

    print('over val')


if __name__ == '__main__':
    convert_to_image()