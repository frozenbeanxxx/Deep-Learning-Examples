'''
将coco keypoints数据集标注格式，转为14个点，
14个点按顺序表示[head, neck, ls, le, lw, rs, re, rw, lh, lk, la, rh, rk, ra]
l:left, r:right, s:shoulder, e:elbow, w:wrist, h:hip, k:knee, a:ankle
'''

import os
import json
import argparse


def convert_coco_to_kps14(args, data_root, out_sub_dir, trans_map, start_idx=2):
    print(args)
    in_ann_file = os.path.join(data_root, args['ann_file'])
    keypoints_name = ['head', 'neck', 'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist',
                 'left_hip', 'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle']
    skeleton = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[2,8],[8,9],[9,10],[5,11],[11,12],[12,13],[8,11]]
    skeleton = [[i[0]+1,i[1]+1] for i in skeleton]

    joints_num = 14
    coco_annos = json.load(open(in_ann_file, 'r'))
    for image in coco_annos['images']:
        image['file_name'] = os.path.join(args['sub_image_path'], image['file_name'])
        image['id'] += args['extid']
    for annotation in coco_annos['annotations']:
        if 'segmentation' in annotation.keys():
            del annotation['segmentation']
        annotation['image_id'] += args['extid']
        annotation['id'] += args['extid']
        new_kps = [0] * joints_num * 3
        for i in range(start_idx, joints_num):
            new_kps[i*3:i*3+3] = annotation['keypoints'][trans_map[i]*3:trans_map[i]*3+3]
        annotation['keypoints'] = new_kps
    for category in coco_annos['categories']:
        category['keypoints'] = keypoints_name
        category['skeleton'] = skeleton

    output_filedir = os.path.join(data_root, out_sub_dir)#, args['out_filename'])
    os.makedirs(output_filedir, exist_ok=True)
    output_filepath = os.path.join(output_filedir, args['out_filename'])
    with open(output_filepath, 'w') as file_obj:
        json.dump(coco_annos, file_obj)
    print('over')


if __name__ == '__main__':
    data_root = '/media/weixing/diskD/dataset'
    out_sub_dir = 'keypoints_merge_annotations'

    args_set = [
        {'sub_image_path': 'coco/train2014', 'extid': 110000000,
         'ann_file': 'coco/annotations/person_keypoints_train2014.json', 'out_filename': 'kps14_coco_train2014.json'},
        {'sub_image_path': 'coco/val2014', 'extid': 120000000,
         'ann_file': 'coco/annotations/person_keypoints_val2014.json', 'out_filename': 'kps14_coco_val2014.json'},
        {'sub_image_path': 'coco/train2017', 'extid': 130000000,
         'ann_file': 'coco/annotations/person_keypoints_train2017.json', 'out_filename': 'kps14_coco_train2017.json'},
        {'sub_image_path': 'coco/val2017', 'extid': 140000000,
         'ann_file': 'coco/annotations/person_keypoints_val2017.json', 'out_filename': 'kps14_coco_val2017.json'}
    ]
    trans_map = [0, 0, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16]
    #for args in args_set:
    #    convert_coco_to_kps14(args, data_root, out_sub_dir, trans_map)

    args_set_aic = [
        {'sub_image_path': 'ai_challenger/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170909', 'extid': 210000000,
         'ann_file': 'ai_challenger/annotations/aic_train.json', 'out_filename': 'kps14_aic_train.json'},
        {'sub_image_path': 'ai_challenger/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911', 'extid': 220000000,
         'ann_file': 'ai_challenger/annotations/aic_val.json', 'out_filename': 'kps14_aic_val.json'}
    ]
    aic_coco_trans_map = [12, 13, 3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    #for args in args_set_aic:
    #    convert_coco_to_kps14(args, data_root, out_sub_dir, aic_coco_trans_map, start_idx=0)

    args_set_mpii = [
        {'sub_image_path': 'mpii', 'extid': 310000000,
         'ann_file': 'mpii/annotations/train.json', 'out_filename': 'kps14_mpii_train.json'},
    ]
    mpii_coco_trans_map = [9, 8, 13, 14, 15, 12, 11, 10, 3, 4, 5, 2, 1, 0]
    for args in args_set_mpii:
        convert_coco_to_kps14(args, data_root, out_sub_dir, mpii_coco_trans_map, start_idx=0)