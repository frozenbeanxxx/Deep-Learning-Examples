import os
import argparse
import requests
import time
from natsort import natsorted
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('download yoga82')
    parser.add_argument('--root',
                        default='/media/weixing/diskD/dataset_res/yoga/Yoga-82',
                        type=str)
    parser.add_argument('--outdir',
                        default='/media/weixing/diskD/dataset_res/yoga/images_20210514',
                        type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    os.makedirs(args.outdir, exist_ok=True)
    image_link_file_dir = os.path.join(args.root, 'yoga_dataset_links')
    image_link_file = os.listdir(image_link_file_dir)
    image_link_file = natsorted(image_link_file)
    #image_link_file = image_link_file[0:1]

    total_num = 0
    for i, _file in enumerate(image_link_file):
        file = os.path.join(image_link_file_dir, _file)
        if file[-3:] != 'txt':
            continue
        with open(file, 'r') as f:
            #print(file)
            lines = f.readlines()
            print(i, _file, len(lines))
            total_num += len(lines)
            #continue
            #for line in tqdm(lines):
            for line in lines:
                name, url = line.strip().split('\t')
                origin_ext = url.split('?')[0].split('/')[-1].split('.')[-1]
                ext_list = ['jpg', 'JPG', 'png', 'jpeg', 'GIF', 'gif', 'PNG']
                if origin_ext not in ext_list:
                    print(origin_ext, url, url.split('/'))
                category = name.split('/')[0]
                os.makedirs(os.path.join(args.outdir, category), exist_ok=True)
                #continue

                try:
                    r = requests.get(url, allow_redirects=True, timeout=10)
                except :
                    continue
                with open(os.path.join(args.outdir, name), "wb") as f:
                    f.write(r.content)

                time.sleep(0.01)
    print('total_num', total_num)