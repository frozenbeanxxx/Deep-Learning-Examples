import argparse
import cv2

def parse_args():
    parser = argparse.ArgumentParser("load image")
    parser.add_argument('image_path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    image_path = args.image_path
    img = cv2.imread(image_path)
    if img is None:
        print(image_path)
    cv2.imshow('win', img)
    cv2.waitKey(0)