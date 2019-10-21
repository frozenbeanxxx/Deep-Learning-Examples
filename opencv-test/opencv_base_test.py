import cv2 
import random 
import numpy as np  

def read_image():
    image_path = "E:/dataset/hl_pokemon/PokemonElf/number/#001.png"
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    shape = img.shape 
    print(shape)
    print(img)

def GaussionTest():
    image_path = "/media/wx/0B8705400B870540/dataset/test_image/timg.jpg"
    img = cv2.imread(image_path)
    rand_ksize = random.randint(1, 3)
    rand_ksize = rand_ksize if rand_ksize % 2 == 1 else rand_ksize - 1
    rand_ksize = 3
    kernel_size = (rand_ksize, rand_ksize)
    print(kernel_size)
    sigma = 0.5
    img = cv2.GaussianBlur(img, kernel_size, sigma, sigmaY=sigma)
    cv2.imshow("win", img)
    cv2.waitKey(0)

def ImageSliceTest():
    image_path = "/media/wx/0B8705400B870540/temp/20190301_samsung_SAMSUNG-SM-G890A_1440_2560_218 - 副本.JPEG"
    img = cv2.imread(image_path)[:,:,::-1]
    #img = cv2.imread(image_path)
    cv2.imshow("win", img)
    cv2.waitKey(0)

def convert_color_channels():
    image_path = "D:/dataset/test_image/timg.jpg"
    img = cv2.imread(image_path)
    img_width, img_height = 200, 100
    #img = cv2.resize(img, dsize=(img_width, img_height))
    img = img[:,:,::-1]
    #img = img.T
    #img = img.transpose(1,0,2)
    print(img.shape)
    #img = cv2.imread(image_path)
    cv2.imshow("win", img)
    cv2.waitKey(0)

def save_image():
    image_path = "D:\\temp\\aaa/a1.bmp"
    img = cv2.imread(image_path)
    cv2.imwrite("D:\\temp\\aaa/a1.jpg", img)

def try_open():
    image_path = "D:\\temp\\aaa/a1.bmp"
    img = cv2.imread(image_path, 0)
    #a = img.shape
    img = np.expand_dims(img, axis=-1)
    try:
        #a = img.shape
        print('img', img)
        print('shape', img.shape)
    except:
        # AttributeError:
        print('error')

read_image()
#GaussionTest()
#ImageSliceTest()
#convert_color_channels()
#save_image()
#try_open()