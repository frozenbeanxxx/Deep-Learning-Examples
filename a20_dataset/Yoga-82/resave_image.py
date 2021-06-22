import os
import shutil
import argparse
import cv2

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('download yoga82')
    parser.add_argument('--srcdir',
                        default='/media/weixing/diskD/dataset_res/yoga/images',
                        type=str)
    parser.add_argument('--dstdir',
                        default='/media/weixing/diskD/dataset_res/yoga/images_20210514',
                        type=str)
    args = parser.parse_args()
    return args

def resave_image():
    args = parse_args()
    print(args)
    category = os.listdir(args.srcdir)
    #category = ['Akarna_Dhanurasana']
    #category = ['Rajakapotasana', 'Upward_Bow_(Wheel)_Pose_or_Urdhva_Dhanurasana_', 'Sitting pose 1 (normal)']
    count = 0
    for cat in category:
        image_dir = os.path.join(args.srcdir, cat)
        files = os.listdir(image_dir)
        cnt = 0
        for file in files:
            image_path = os.path.join(image_dir, file)
            #print(image_path)
            img = cv2.imread(image_path)
            if img is not None:
                dstdir = os.path.join(args.dstdir, cat, file)
                cv2.imwrite(dstdir, img)
                cnt += 1
        print(image_dir, '   cnt:', cnt)
        count += cnt
    print('count:', count)

def test_one_image():
    image_path = '/media/weixing/diskD/dataset_res/yoga/images/Bow_Pose_or_Dhanurasana_/684.jpg'
    img = cv2.imread(image_path)
    print(img, img == None)
    if img:
        print(img.shape)
        cv2.imshow('win', img)
        cv2.waitKey(0)

def gen_image_path_txt():
    parser = argparse.ArgumentParser('yoga82')
    parser.add_argument('--srcdir',
                        default='/media/weixing/diskD/dataset_res/yoga/images',
                        type=str)
    parser.add_argument('--outdir',
                        default='/media/weixing/diskD/dataset_res/yoga/Yoga-82',
                        type=str)
    args = parser.parse_args()

    category = os.listdir(args.srcdir)
    category = ['Akarna_Dhanurasana']
    count = 0
    for cat in category:
        image_dir = os.path.join(args.srcdir, cat)
        files = os.listdir(image_dir)
        cnt = 0
        for file in files:
            image_path = os.path.join(image_dir, file)
            img = cv2.imread(image_path)
            if img is None:
                dstdir = os.path.join(args.dstdir, cat)
                shutil.move(image_path, dstdir)
                cnt += 1
        print(image_dir, '   cnt:', cnt)
        count += cnt
    print('count:', count)

if __name__ == '__main__':
    #test_one_image()
    resave_image()
    #gen_image_path_txt()