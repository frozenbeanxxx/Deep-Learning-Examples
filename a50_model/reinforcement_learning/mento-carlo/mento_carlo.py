import random
import math

def cal_pi():
    for _ in range(10):
        cnt = 0
        MAX_ITERS = 1000000
        for _ in range(MAX_ITERS):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x**2 + y**2 <= 1:
                cnt += 1
        print(cnt*4.0 / MAX_ITERS)
            
def cal_integral():
    print(math.log(2, math.e))
    for _ in range(10):
        cnt = 0
        MAX_ITERS = 1000000
        for _ in range(MAX_ITERS):
            x = random.uniform(1.0, 2.0)
            y = random.uniform(0.0, 1.0)
            if x*y <= 1.0:
                cnt += 1
        print(1.0*(cnt/MAX_ITERS))

def cal_fun_maximum():
    ymax = 0
    y = lambda x:200*math.sin(x)*math.exp(-0.05*x)
    num = 1000
    for i in range(num):
        x0 = random.uniform(-2, 2)
        if y(x0) > ymax:
            ymax = y(x0)
            print(ymax)

if __name__ == "__main__":
    #cal_pi()
    #cal_integral()
    cal_fun_maximum()