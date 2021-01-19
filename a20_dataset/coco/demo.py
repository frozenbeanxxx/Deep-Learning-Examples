import os
import argparse
import numpy as np
import cv2
from pycocotools.coco import COCO

def parse_args() -> argparse.Namespace():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', '-a', type=str)
    parser.add_argument('--image_dir', '-d', type=str)
    args = parser.parse_args()
    return args

def load_coco_with_boxes(args):
    annFile = args.ann_file
    image_dir = args.image_dir
    coco=COCO(annFile)
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
    for img_info in image_infos:
        img_path = os.path.join(image_dir, img_info['file_name'])
        img = cv2.imread(img_path)

        if 0:
            cv2.imshow('win', img)
            c = cv2.waitKey(0)
            if c & 0xFF == ord('q'):
                break



if __name__ == '__main__':
    args = parse_args()
    print(args)
    load_coco_with_boxes(args)