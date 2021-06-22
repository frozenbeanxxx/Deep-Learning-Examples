import json
import cv2
import os
import numpy as np
from tqdm import tqdm

# =======================================读取mpii标注=======================================

root_dir = '/media/weixing/diskD/dataset/mpii'
mpii_imgpath = "images"  # mpii图片的地址


#mpiipath = "annot/train.json"  # mpii train的地址
#writepath = "annotations/person_keypoints_mpii2coco_train.json"  # mpii转换成coco的地址(train)
#p = 100000  # train

mpiipath = "annot/valid.json"  # mpii valid的地址
writepath = "annotations/person_keypoints_mpii2coco_val.json"  # mpii转换成coco的地址(val)
p = 200000  # val

mpiipath = os.path.join(root_dir, mpiipath)
writepath = os.path.join(root_dir, writepath)
mpii_imgpath = os.path.join(root_dir, mpii_imgpath)






with open(mpiipath, 'r', encoding='utf-8')as f:
    mpii = json.load(f)

# cocopath = "./coco/annotations/new_person_keypoints_train2017.json"  # 原coco train2017的地址
# with open(cocopath, 'r', encoding='utf-8')as f:
#     origin_coco = json.load(f)

# =======================================数据集转换=======================================

coco = { # 定义coco标注的总字典，包含三个子字典
    'annotations': [],
    "categories": [{"name": "person", "id": 1}],
    "images": []
}
keypoints = []
skeleton = [[0,1],[1,2],[2,6],[6,3],[3,4],[4,5],[6,7],[7,8],[8,9],[8,12],[12,11],[11,10],[8,13],[13,14], [14, 15]]
coco["categories"][0]['keypoints'] = keypoints
coco["categories"][0]['skeleton'] = skeleton

# start = 1000
# end = 2000
# for i in tqdm(range(start,end)):
for i in tqdm(range(len(mpii))):
    mpii_inf = mpii[i]  # mpii_inf是mpii.json中第i个标注
    # 必须要在for循环中定义coco.json文件中的两个字典，
    # 因为append()是浅复制，在for外面定义会导致之前append的block内容被覆盖
    numpoints = 0
    for j in range(16):
        if mpii_inf['joints_vis'][j] == 1:
            numpoints += 1
    if numpoints <= 6:
        continue
    annot_block = {
        "category_id": 1,
        "bbox": [],
        "id": 0,
        "image_id": 0,
        "keypoints": [],
        "iscrowd": 0,
        "num_keypoints": 0,
        "area": 0,
    }
    images_block = {
        "file_name": "000000000.jpg",
        "height": 0,
        "width": 0,
        "id": 0
    }

#           ===========================id&image_id的转换============================

    ids = ''.join(a for a in mpii_inf['image'] if a in "0123456789")
    ids = int(ids)
    annot_block['id'] = i + p  # p:(see above)
    annot_block['image_id'] = ids
    images_block['file_name'] = mpii_inf['image']
    images_block['id'] = ids

    annot_block['num_keypoints'] = sum(mpii_inf['joints_vis'])

#           ===========================keypoints&bbox的转换============================

    landmarks = []
    for j in range(16):
        if mpii_inf['joints_vis'][j] == 1:
            annot_block['keypoints'].extend([int(mpii_inf['joints'][j][0]), int(mpii_inf['joints'][j][1]), 2])
            points = []
            points.extend([int(mpii_inf['joints'][j][0]), int(mpii_inf['joints'][j][1])])
            landmarks.append(points)
        else:
            annot_block['keypoints'].extend([0, 0, 0])
    x, y, w, h = cv2.boundingRect(np.array(landmarks))
    x = x - 30
    y = y - 30
    w = w + 60
    h = h + 60

    img = cv2.imread(os.path.join(mpii_imgpath, images_block['file_name']))
    imgh, imgw, ii = img.shape
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, imgw - x)
    h = min(h, imgh - y)
    annot_block['bbox'].extend([x, y, w, h])
    images_block['height'] = imgh
    images_block['width'] = imgw
    annot_block['area'] = imgh * imgw

#           ========================显示图片，可注释掉==============================

    # print(os.path.join(mpii_imgpath, images_block['file_name']))
    # for k in range(16):
    #     cv2.circle(img,((annot_block['keypoints'][k * 3]),int(annot_block['keypoints'][k * 3 + 1])),2,(0,255,0),3)


    # print(img.shape)
    # print('x1y1,x2y2: {}, {}, {}, {}'.format(x, y, x + w, y + h))
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

#           ========================append两个字典=================================

    coco['annotations'].append(annot_block)
    coco['images'].append(images_block)

# =======================================写入文件========================================
# with open(writepath, 'w', encoding='utf-8') as f:
with open(writepath, 'w') as f:
    json.dump(coco, f)
print('ret')