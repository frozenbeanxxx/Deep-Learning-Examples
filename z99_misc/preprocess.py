import fire
import math

def t1_2(objH, objW, bgH, bgW, divisor=16, min_size=32, objmin=16):
    ratioH = bgH / objH 
    ratioW = bgW / objW
    a = min(objH, objW)
    ratio = objmin / a
    new_objH = objH * ratio
    new_objW = objW * ratio
    new_bgH = bgH * ratio
    new_bgW = bgW * ratio
    b = min(new_bgH, new_bgW)
    new_b = max(min_size, math.ceil(b / divisor) * divisor)
    ratio2 = new_b / b
    modelH = max(min_size, round(new_bgH * ratio2 / divisor) * divisor)
    modelW = max(min_size, round(new_bgW * ratio2 / divisor) * divisor)

    return (modelH, modelW)

def t1():
    objH = 74
    objW = 224
    bgH = 280
    bgW = 540
    input_size = t1_2(objH, objW, bgH, bgW)
    print(input_size)

if __name__ == "__main__":
    t1()