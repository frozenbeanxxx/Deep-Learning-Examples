import os
import random
from pycocotools.coco import COCO
import numpy as np
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = '/home/weixing/dataset/coco'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

print(os.listdir(dataDir))
image_dir = os.path.join(dataDir, dataType)
filenames = os.listdir(image_dir)
print(f'in {image_dir} image number : ', len(filenames))

def show_image_test():
    for filename in filenames:
        image_path = os.path.join(image_dir, random.choice(filenames))
        image = cv2.imread(image_path)
        print(image_path, image.shape)
        cv2.imshow('win', image)
        c = cv2.waitKey(0)
        if c&0xFF == ord('q'):
            break

#show_image_test()

coco = COCO(annFile)

print(type(COCO))

cats = coco.getCatIds()
cats = coco.loadCats(cats)
names = [cat['name'] for cat in cats]
pass