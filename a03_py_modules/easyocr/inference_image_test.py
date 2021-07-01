import easyocr

def simple_example():
    reader = easyocr.Reader(['ch_sim','en']) # need to run only once to load model into memory
    result = reader.readtext('data/chinese.jpg')
    print(result)


if __name__ == '__main__':
    simple_example()