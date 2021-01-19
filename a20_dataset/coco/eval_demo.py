import argparse

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def parse_args() -> argparse.Namespace():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', '-a', type=str)
    parser.add_argument('--res_file', '-r', type=str)
    args = parser.parse_args()
    return args

def eval_coco_boxes(args):
    annFile = args.ann_file
    resFile = args.res_file
    cocoGt=COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = sorted(cocoGt.getImgIds())
    imgIds = imgIds[0:100]
    #print(imgIds)
    #imgIds = imgIds[np.random.randint(100)]
    annType = 'bbox'
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    eval_coco_boxes(args)