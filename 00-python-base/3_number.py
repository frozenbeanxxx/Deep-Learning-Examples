import math
import random
import string

if __name__ == "__main__":
    print("math function")
    print("in python: abs, max, min, round")
    print("in math: fabs, ceil, floor, exp, log, ")
    a = -43.56
    print(f"abs({a}) = ", abs(a))
    c = 3 + 4j
    print(f"abs({c}) = ", abs(c))
    print(f"ceil({a}) = ", math.ceil(a))
    print(f"math.ceil(math.pi) = ", math.ceil(math.pi))
    print(f"math.exp({a}) = ", math.exp(a))
    print(f"math.fabs({a}) = ", math.fabs(a))
    print(f"math.floor(math.pi) = ", math.floor(math.pi))
    print("math.log(math.e) = ", math.log(math.e))
    print("math.log(100, 10) = ", math.log(100, 10))
    l = [100, 0, -100]
    print(f"max({l}) = ", max(l))
    print(f"min({l}) = ", min(l))
    print(f"math.modf({a}) = ", math.modf(a))
    print("math.pow(3,2) = ", math.pow(3, 2))
    print(f"round({math.pi}, 6) = ", round(math.pi, 6))
    print(f"math.sqrt({math.pi}) = ", math.sqrt(math.pi))

    print("\nrandom")
    print("random.choice(range(10)) = ", random.choice(range(10)))
    print("random.choice([1,2,3]) = ", random.choice([1,2,3]))
    print("random.choice('world') = ", random.choice("world"))
    b = string.digits + string.ascii_letters + string.punctuation
    print("b = ", b)
    print("password = ", random.choices(b, k=8))
    print("random.randrange(10,20) = ", random.randrange(10, 20), ", randrange is [)")
    print("random.randint(0,10) = ", random.randint(0, 10), ", randint is []")
    print("random.uniform(10.0, 13.0) = ", random.uniform(10.0, 13.0))
    l2 = list(range(10))
    random.shuffle(l2)
    print("l2 = list(range(10)), random.shuffle(l2), l2 = ", l2)

    print("\nTriangle Function")
    print("math.sin(math.pi) = ", math.sin(math.pi))
    print("math.cos(math.pi) = ", math.cos(math.pi))
    print("math.tan(math.pi) = ", math.tan(math.pi))
    print("math.asin(0) = ", math.asin(0))
    print("math.acos(0) = ", math.acos(0))
    print("math.atan(0) = ", math.atan(0))
    print("math.atan2(1, 1) = ", math.atan2(1, 1))
    print("math.hypot(3, 4) = ", math.hypot(3, 4))
    print("math.radians(90) = ", math.radians(90))
    print("math.degrees(math.pi / 2) = ", math.degrees(math.pi / 2))
