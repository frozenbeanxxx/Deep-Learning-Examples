import numpy as np
from PIL import Image, ImageFont, ImageDraw

def t1():
    image_path = 'E:/dataset/coco/images/train2014/COCO_train2014_000000176470.jpg'
    image = Image.open(image_path)
    print(image)
    font = ImageFont.truetype(font='data/FiraMono-Medium.otf',
                    size=np.floor(1e-1 * image.size[1] + 0.5).astype('int32'))
    print(font)
    draw = ImageDraw.Draw(image)
    label = 'abcd'
    words_width = [font.getsize(w)[0] for w in label]
    text_width =  sum(words_width)
    text_height = max([font.getsize(w)[1] for w in label])
    print('1: ', text_width, text_height)
    label_size = draw.textsize(label, font)
    print(label_size)
    x = 10
    y = 10
    # draw.rectangle([(x, y), (x + label_size[0], y + label_size[1])], fill=(255,0,0))
    draw.rectangle([(x, y), (x + label_size[0], y + label_size[1])], outline=(255,0,0))
    draw.text((x, y), label, fill=(0, 255, 0), font=font)

    label = 'abcdqgjy'
    words_width = [font.getsize(w)[0] for w in label]
    text_width =  sum(words_width)
    text_height = max([font.getsize(w)[1] for w in label])
    print('1: ', text_width, text_height)
    label_size = draw.textsize(label, font)
    print(label_size)
    x = 10
    y = 10 + image.size[1] // 2
    # draw.rectangle([(x, y), (x + label_size[0], y + label_size[1])], fill=(255,0,0))
    draw.rectangle([(x, y), (x + label_size[0], y + label_size[1])], outline=(255,0,0))
    draw.text((x, y), label, fill=(0, 255, 0), font=font)

    image.save('E:/temp/a1/a.jpg')

if __name__ == "__main__":
    import fire
    fire.Fire()