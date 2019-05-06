# -*- coding: utf-8 -*-
import sys
import random
import time
from PIL import Image
import argparse
import apiutil

#print(sys.version_info)
if sys.version_info.major != 3:
    print('Please run under Python3')
    exit(1)

VERSION = "0.0.1"

# 我申请的 Key，随便用，嘻嘻嘻
# 申请地址 http://ai.qq.com
AppID = '1106858595'
AppKey = 'bNUNgOpY6AeeJjFu'

DEBUG_SWITCH = True
FACE_PATH = './'

# 审美标准
BEAUTY_THRESHOLD = 80

# 最小年龄
GIRL_MIN_AGE = 14

def main():
    image_path = 'D:/dataset/test_image/lena.png'
    with open(image_path, 'rb') as bin_data:
            image_data = bin_data.read()

    ai_obj = apiutil.AiPlat(AppID, AppKey)
    rsp = ai_obj.face_detectface(image_data, 0)
    print(rsp)
    major_total = 0
    minor_total = 0

    if rsp['ret'] == 0:
        beauty = 0
        for face in rsp['data']['face_list']:

            msg_log = '[INFO] gender: {gender} age: {age} expression: {expression} beauty: {beauty}'.format(
                gender=face['gender'],
                age=face['age'],
                expression=face['expression'],
                beauty=face['beauty'],
            )
            print(msg_log)
            face_area = (face['x'], face['y'], face['x']+face['width'], face['y']+face['height'])
            img = Image.open(image_path)
            cropped_img = img.crop(face_area).convert('RGB')
            cropped_img.save(FACE_PATH + face['face_id'] + '.png')
            # 性别判断
            if face['beauty'] > beauty and face['gender'] < 50:
                beauty = face['beauty']

            if face['age'] > GIRL_MIN_AGE:
                major_total += 1
            else:
                minor_total += 1

        # 是个美人儿~关注点赞走一波
        if beauty > BEAUTY_THRESHOLD and major_total > minor_total:
            print('beauty')


if __name__ == "__main__":
    main()