import os
import argparse
import numpy as np
import cv2
from pycocotools.coco import COCO


def load_annotation_coco_style():
    '''加载coco格式的mpii标注'''
    #ann_file = '/media/weixing/diskD/dataset/mpii/mpii_human_pose_v1_u12_2/mpii.json'
    ann_file = '/media/weixing/diskD/dataset/mpii/annotations/person_keypoints_mpii2coco_val.json'
    coco = COCO(ann_file)
    print('over')

if __name__ == '__main__':
    load_annotation_coco_style()