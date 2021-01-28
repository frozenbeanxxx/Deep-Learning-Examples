import io
import os
import argparse
from tqdm import tqdm
import shutil
from PIL import Image as pil_image
import cv2

def parse_args():
    parser = argparse.ArgumentParser("filter error images")
    parser.add_argument('image_dir')
    args = parser.parse_args()
    return args

def main(args) -> None:
    print(args)
    image_dir = args.image_dir
    filelist = os.listdir(image_dir)
    for file in filelist:
        path = image_dir + os.sep + file
        with open(path, 'rb') as f:
            try:
                img = pil_image.open(io.BytesIO(f.read()))
            except:
                print(path)

def main_opencv(args) -> None:
    print(args)
    image_dir = args.image_dir
    filelist = os.listdir(image_dir)
    for file in tqdm(filelist):
        if not (file[-3:] == 'jpg' or file[-3:] == 'png'):
            print(file)
            continue
        path = image_dir + os.sep + file
        img = cv2.imread(path)
        if img is None:
            print(path)
            #os.remove(path)



if __name__ == '__main__':
    args = parse_args()
    #main(args)
    main_opencv(args)