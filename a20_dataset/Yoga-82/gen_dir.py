import os
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('download yoga82')
    parser.add_argument('--root',
                        default='/media/weixing/diskD/dataset_res/yoga/Yoga-82',
                        type=str)
    parser.add_argument('--outdir',
                        default='/media/weixing/diskD/dataset_res/yoga/images_error',
                        type=str)
    args = parser.parse_args()
    return args

def generate_directory():
    args = parse_args()
    print(args)
    os.makedirs(args.outdir, exist_ok=True)
    image_link_file_dir = os.path.join(args.root, 'yoga_dataset_links')
    image_link_file = os.listdir(image_link_file_dir)
    image_link_file = image_link_file
    for _file in image_link_file:
        file = os.path.join(image_link_file_dir, _file)
        if file[-3:] != 'txt':
            continue
        with open(file, 'r') as f:
            print(file)
            lines = f.readlines()
            # for line in tqdm(lines):
            for line in lines:
                name, url = line.strip().split('\t')
                print(url)
                category = name.split('/')[0]
                os.makedirs(os.path.join(args.outdir, category), exist_ok=True)
                break

if __name__ == '__main__':
    generate_directory()