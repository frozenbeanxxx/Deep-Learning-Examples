import os 


def replace_slash():
    '''
    \ -> /, windows路径变linux路径，
    下载的webface路径列表清单斜杠方向需要整理，
    http://pan.baidu.com/s/1hrKpbm8，这个是其他人清洗过的webface，比原始的图片数目少了一些。
    '''
    src_file_path = '/media/weixing/diskD/dataset/face/webface/train_list_org.txt'
    dst_file_path = '/media/weixing/diskD/dataset/face/webface/train_list.txt'
    with open(src_file_path) as fsrc:
        lines = fsrc.readlines()
        a = lines#[:10]
        #print(a)
        print('read over')
        new_lines = [s.replace('\\', '/') for s in a]
        #print(new_lines)
        print('process over')

        with open(dst_file_path, 'w') as fdst:
            fdst.writelines(new_lines)


def validate_path():
    image_root_dir = '/media/weixing/diskD/dataset/face/webface/CASIA-WebFace'
    label_file_path = '/media/weixing/diskD/dataset/face/webface/train_list.txt'
    with open(label_file_path) as f:
        lines = f.readlines()
        for l in lines:
            path, label = l.strip().split(' ')
            image_path = os.path.join(image_root_dir, path)
            if not os.path.exists(image_path):
                print(path)


def main():
    print('over')

if __name__ == '__main__':
    #main()
    #replace_slash()
    #validate_path()