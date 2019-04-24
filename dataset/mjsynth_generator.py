from tqdm import *
import os 
import os.path as osp 
import time 
import progressbar 

def create_dataset_from_file(root, file_path):
    #p = progressbar.ProgressBar()
    with open(osp.join(root, file_path), "r") as f:
        readlines = f.readlines()

    #img_paths = []
    #for img_name in tqdm(readlines[:10], desc="read dir:"): #, 
    #    img_name = img_name.rstrip().strip()
    #    img_name = img_name.split(" ")[0]
    #    img_path = root + "/" + img_name
    #    # if osp.exists(img_path):
    #    img_paths.append(img_path)
    #img_paths = img_paths[:100]
    img_paths = [root + "/" + img_name.rstrip().strip().split(" ")[0] for img_name in tqdm(readlines[:10], desc="read dir:")]
    labels = [img_path.split("/")[-1].split("_")[-2] for img_path in tqdm(img_paths, desc="generator label:")]
    return img_paths, labels

root = "D:/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px"
start_time = time.time()
img_paths, labels = create_dataset_from_file(root, "annotation_train.txt")
print('time taken : {}'.format(time.time() - start_time))
print(img_paths)
print(labels)