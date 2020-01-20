import os 
import random as rnd
import numpy as np 
import cv2 
from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter

def generate_horizontal_text(text, font, text_color, font_size, space_width):
    image_font = ImageFont.truetype(font=font, size=font_size)
    words = text.split(' ')
    space_width = image_font.getsize(' ')[0] * space_width

    words_width = [image_font.getsize(w)[0] for w in words]
    text_width =  sum(words_width) + int(space_width) * (len(words) - 1)
    text_height = max([image_font.getsize(w)[1] for w in words])
    
    txt_img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))

    txt_draw = ImageDraw.Draw(txt_img)

    colors = [ImageColor.getrgb(c) for c in text_color.split(',')]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2]))
    )
    #print(f'char w: {text_width}, h: {text_height}', 'font_size:', font_size, 'fit:', fit, 'shape:', txt_img.size)
    for i, w in enumerate(words):
        txt_draw.text((sum(words_width[0:i]) + i * int(space_width), 0), w, fill=fill, font=image_font)

    return txt_img

def t1():
    out_dir = 'E:/temp/text'
    #font_dir = r'E:\prj_data\hv_ocr\cod\font'
    #font_dir = r'E:\prj_data\hv_ocr_number\font2'
    #font_dir = r'E:\dataset\hv_ocr\font_201912171515_2'
    #font_dir = r'E:\temp\font\android-fonts\fonts'
    #font_dir = r'E:\temp\font\win'
    font_dir = r'E:\prj_data\hv_ocr\cod\font'
    font_list = os.listdir(font_dir)
    texts = ['0123456789', 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "~!@#$%^&*(){}[]_+`-=:;'<>?,./|"]
    for font in font_list:
        font_path = os.path.join(font_dir, font)
        font_name = font.split('.')[0]
        print(font, font_name, font_path)
        for i, text in enumerate(texts):
            image = generate_horizontal_text(text, font_path, '#0000ff,#0000ff', 128, 0)
            image.save(os.path.join(out_dir, font_name) + '_' + str(i) + '.png')

if __name__ == "__main__":
    t1()