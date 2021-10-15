import os
from natsort import natsorted


def dir_rename(root_dir):
    name_dict = {'0': 'airplane', '1': 'automobile', '2': 'bird', '3': 'cat', '4': 'deer',
                 '5': 'dog', '6': 'frog', '7': 'horse', '8': 'ship', '9': 'truck'}
    for dirname in name_dict.keys():
        src_dir = os.path.join(root_dir, dirname)
        if os.path.exists(src_dir):
            dst_dir = os.path.join(root_dir, name_dict[dirname])
            os.rename(src_dir, dst_dir)
            print(src_dir)
            print(dst_dir)
    print('over {}'.format(root_dir))


def dir_rename_run():
    #root_dir = '/media/weixing/diskD/dataset/pytorch/cifar10/images/train'
    root_dir = '/home/weixing/temp/99_misc/cifar10/images/train'
    dir_rename(root_dir)
    #root_dir = '/media/weixing/diskD/dataset/pytorch/cifar10/images/val'
    root_dir = '/home/weixing/temp/99_misc/cifar10/images/val'
    dir_rename(root_dir)
    print('over')


if __name__ == '__main__':
    dir_rename_run()