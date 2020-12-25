import os
import argparse
import json
import PIL
from PIL import Image

def convert_aic2coco_keypoint(args : argparse.Namespace):
    aic_annos = json.load(open(args.aic_keypoint_anno, 'r'))
    if False:
        total = 10000
        aic_annos = aic_annos[:total]
    else:
        total = 'all'

    image_dir = args.aic_imagedir
    coco = {}
    coco['images'] = list()
    coco['annotations'] = list()
    transformer = {'head': 0, 'neck': 1,
                   'ls': 2, 'le': 4, 'lw': 6,
                   'rs': 3, 're': 5, 'rw': 7,
                   'lh': 8, 'lk': 10, 'la': 12,
                   'rh': 9, 'rk': 11, 'ra': 13}
    id = 0
    for index, anno in enumerate(aic_annos):
        image_path = os.path.join(image_dir, anno['image_id'])+'.jpg'
        try:
            img = Image.open(image_path)
        except PIL.UnidentifiedImageError as err:
            print(err)
            continue
        except FileNotFoundError as err:
            print(err)
            continue

        width, height = img.size
        coco_img = {}
        coco_img['license'] = 0
        coco_img['file_name'] = os.path.join(anno['image_id'])+'.jpg'
        coco_img['width'] = width
        coco_img['height'] = height
        coco_img['data_captured'] = 0
        coco_img['coco_url'] = ''
        coco_img['flickr_url'] = ''
        coco_img['id'] = index
        humans = anno['keypoint_annotations'].keys()
        if len(anno['keypoint_annotations'].keys()) != 1:
            continue
        for human in humans:
            num_keypoint = 0
            cocokey = list()
            keypoints = anno['keypoint_annotations'][human]
            for i in range(14):
                if keypoints[i*3+2] != 3:
                    num_keypoint += 1
                keypoints[i*3+2] = 3 - keypoints[i*3+2]
            # nose<-----neck(aic)
            cocokey[0:3] = keypoints[transformer['head']*3 : transformer['head']*3+3]
            # left_eye
            cocokey[3:6] = [0, 0, 0]
            # right_eye
            cocokey[6:9] = [0, 0, 0]
            # left_ear
            cocokey[9:12] = [0, 0, 0]
            # right_ear
            cocokey[12:15] = [0, 0, 0]
            # other joints
            cocokey[15:51] = keypoints[transformer['ls']*3 : transformer['ra']*3+3]

            coco_anno = {}
            coco_anno['num_keypoints'] = num_keypoint
            coco_anno['image_id'] = index
            coco_anno['id'] = id
            coco_anno['keypoints'] = cocokey
            bbox = anno['human_annotations'][human]
            coco_anno['bbox'] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            coco_anno['area'] = coco_anno['bbox'][2] * coco_anno['bbox'][3]
            coco_anno['iscrowd'] = 0
            coco_anno['segmentation'] = [[]]
            coco['annotations'].append(coco_anno)
            id += 1
        coco['images'].append(coco_img)
        if index % 1000 == 0:
            print('{}/{}, id:{}'.format(index, len(aic_annos), id))
    output_file = os.path.join(os.path.dirname(args.aic_keypoint_anno),
                               f'coco_{total}_single_' + os.path.basename(args.aic_keypoint_anno))
    with open(output_file, 'w') as fid:
        json.dump(coco, fid)
    print('Saved to {}'.format(output_file))

def satisify_aic_keypoint(args : argparse.Namespace):
    aic_annos = json.load(open(args.aic_keypoint_anno, 'r'))
    if False:
        total = 10000
        aic_annos = aic_annos[:total]
    else:
        total = 'all'

    image_dir = args.aic_imagedir
    coco = {}
    coco['images'] = list()
    coco['annotations'] = list()
    transformer = {'head': 0, 'neck': 1,
                   'ls': 2, 'le': 4, 'lw': 6,
                   'rs': 3, 're': 5, 'rw': 7,
                   'lh': 8, 'lk': 10, 'la': 12,
                   'rh': 9, 'rk': 11, 'ra': 13}
    id = 0
    for index, anno in enumerate(aic_annos):
        image_path = os.path.join(image_dir, anno['image_id'])+'.jpg'
        try:
            img = Image.open(image_path)
        except PIL.UnidentifiedImageError as err:
            print(err)
            continue
        except FileNotFoundError as err:
            print(err)
            continue

        width, height = img.size
        coco_img = {}
        coco_img['license'] = 0
        coco_img['file_name'] = os.path.join(anno['image_id'])+'.jpg'
        coco_img['width'] = width
        coco_img['height'] = height
        coco_img['data_captured'] = 0
        coco_img['coco_url'] = ''
        coco_img['flickr_url'] = ''
        coco_img['id'] = index
        if len(anno['keypoint_annotations'].keys()) == 1:
            id += 1

        if index % 1000 == 0:
            print('{}/{}, id:{}'.format(index, len(aic_annos), id))
    print('single person count: ', id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert AI challenger keypoint to coco format')
    parser.add_argument('--aic_keypoint_anno',
                        default='/media/weixing/diskD/dataset/ai_challenger/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json',
                        type=str)
    parser.add_argument('--aic_imagedir',
                        default='/media/weixing/diskD/dataset/ai_challenger/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902',
                        type=str)
    parser.add_argument('--coco_keypoint_anno',
                        default='',
                        type=str)
    args = parser.parse_args()
    print(args)
    print(type(args))

    convert_aic2coco_keypoint(args)
    #satisify_aic_keypoint(args)