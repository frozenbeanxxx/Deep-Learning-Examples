import os 
import cv2
import numpy as np 

def t1():
    # generate n dir 
    root_dir = 'E:\\dataset\\hv_pubg\\killed_num\\20190810_2'
    for i in range(102):
        dir_name = os.path.join(root_dir, str(i))
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

def t2():
    # put all file_path in dir to a.txt
    #root_dir = 'E:/dataset/test_image'
    root_dir = 'E:\\dataset\\hv_pubg\\killed_num\\20190810_2'
    #files_path = os.walk(root_dir)
    #print(list(files_path))
    file_list = []
    for parent, dirnames, filenames in os.walk(root_dir):#,  followlinks=True):
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            #print('文件名：%s' % filename)
            #print('文件完整路径：%s' % file_path)
            file_list.append(file_path + '\n')
    print('length: ', len(file_list))
    #print('file_list: \n', file_list)
    txt_file_path = "E:\\dataset\\hv_pubg\\killed_num/20190810_3.txt"
    with open(txt_file_path,"w") as f:
        f.writelines(file_list)

def t3():
    txt_file_path = "E:\\dataset\\hv_pubg\\killed_num/20190810_3.txt"
    file_list = []
    with open(txt_file_path,"r") as f:    #设置文件对象
        for line in f:
            line = line.strip('\n')
            file_list.append(line)
            img = cv2.imread(line)
            try:
                shape = img.shape
                img.astype(np.float32)
            except:
                print("file: ", line)
            #cv2.imshow('win', img)
            #cv2.waitKey(0)
            #print(line)
    #print(file_list)
    file_list2 = file_list[:100]
    for i in file_list2:
        a = int(i.split('\\')[-2])
        #print(a)

if __name__ == "__main__":
    #t1()
    t2()