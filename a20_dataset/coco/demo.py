import os
import argparse
import numpy as np
import cv2
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', '-a', type=str)
    parser.add_argument('--image_dir', '-d', type=str)
    args = parser.parse_args()
    return args

def load_coco_with_boxes(args):
    annFile = args.ann_file
    image_dir = args.image_dir
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    catIds = coco.getCatIds(catNms=['person']);
    imgIds = coco.getImgIds(catIds=catIds);
    # imgIds = coco.getImgIds(imgIds = [324158])
    imgIds2 = coco.getImgIds(imgIds)
    imgIds2 = imgIds2[:10]
    #img = coco.loadImgs(imgIds2[np.random.randint(0, len(imgIds2))])[0]

    {
        "image_id": 74,
        "category_id": 1,
        "bbox": [
            281.65,
            103.41,
            11.69,
            24.25
        ],
        "score": 0.087
    }

    image_infos = coco.loadImgs(imgIds2)
    print('image num:', len(image_infos))
    for img_info in image_infos:
        img_path = os.path.join(image_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        if 1:
            cv2.imshow('win', img)
            c = cv2.waitKey(0)
            if c & 0xFF == ord('q'):
                break


def show_keypoints():
    dataDir = '/media/weixing/diskD/dataset'
    #dataDir = '/media/weixing/diskD/dataset/mpii'

    #annFile = f'{dataDir}/keypoints_merge_annotations/kps14_coco_val2017.json'
    #annFile = f'{dataDir}/keypoints_merge_annotations/kps14_aic_val.json'
    annFile = f'{dataDir}/keypoints_merge_annotations/kps14_mpii_train.json'
    #annFile = f'{dataDir}/annotations/train.json'
    coco_kps = COCO(annFile)
    catIds = coco_kps.getCatIds(catNms=['person']);
    imgIds = coco_kps.getImgIds(catIds=catIds);
    imgIds = coco_kps.getImgIds(imgIds = [310000036])#210000159])
    imgIds = coco_kps.getImgIds(imgIds)
    img = coco_kps.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    I = io.imread('%s/%s' % (dataDir, img['file_name']))
    #I = io.imread('%s/%s/%s' % (dataDir, 'images', img['file_name']))
    plt.imshow(I)
    plt.axis('off')
    ax = plt.gca()

    annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco_kps.loadAnns(annIds)
    coco_kps.showAnns(anns)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    #load_coco_with_boxes(args)
    show_keypoints()