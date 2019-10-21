from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='E:/dataset/coco'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

