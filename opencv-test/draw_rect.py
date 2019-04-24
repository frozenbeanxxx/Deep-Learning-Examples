import os
import cv2
import random

cv2.namedWindow("win", 0)

def draw1():
    root_path = "D:\\dataset\\test2"
    
    while True:
        num = random.randint(1,761)
        image_path = root_path + "/image" + "/gt_img_" + str(num) + ".jpg"
        if(os.path.exists(image_path) == False):
            continue
        img = cv2.imread(image_path)
        anno_path = root_path + "/annotations" + "/gt_img_" + str(num) + ".txt"
        f = open(anno_path, 'r')
        lines = f.readlines()
        for line in lines:
            strings = line.split(" ")
            cv2.rectangle(img, (int(strings[0]), int(strings[1])), \
                (int(strings[2]), int(strings[3])), (0, 0, 255), 3)

        cv2.imshow("win", img)
        c = cv2.waitKey(0)
        if(c & 0xFF == ord('q')):
            break

def draw2():
    image_path = "D:/dataset/ocr/auto_gen/images/2019-03-07-16-31-31_0.jpg"
    anno_path = "D:/dataset/ocr/auto_gen/annotations/2019-03-07-16-31-31_0.txt"
    img = cv2.imread(image_path)
    f = open(anno_path, 'r')
    lines = f.readlines()
    
    for line in lines:
        strings = line.split(" ")
        cv2.rectangle(img, (int(strings[0]), int(strings[1])), \
            (int(strings[2]), int(strings[3])), (0, 0, 255), 1)

    cv2.imshow("win", img)
    cv2.waitKey(0)

draw2()