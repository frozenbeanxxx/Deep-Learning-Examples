import sys

def iterator_1():
    print("\niterator_1")
    l = [1, 2, 3, 4, 5]
    it = iter(l)
    print("next(it) = ", next(it))
    print("next(it) = ", next(it))

    print("iterator used in for")
    for x in it:
        print(x, end=' ')
    print()

def iterator_2():
    print("\niterator_2")
    list=[1,2,3,4]
    it = iter(list)    # 创建迭代器对象
    
    while True:
        try:
            print (next(it))
        except StopIteration:
            break
            #sys.exit()

def iterator_3():
    print("\niterator_3")
    class MyNumbers:
        def __iter__(self):
            self.a = 1
            return self
        def __next__(self):
            x = self.a 
            self.a += 1
            return x 
    
    myclass = MyNumbers()
    myiter = iter(myclass)
    
    print(next(myiter))
    print(next(myiter))
    print(next(myiter))
    print(next(myiter))
    print(next(myiter))

def iterator_4():
    print("\niterator_4")
    class MyNumbers:
        def __iter__(self):
            self.a = 1
            return self
        
        def __next__(self):
            if self.a <= 5:
                x = self.a
                self.a += 1
                return x
            else:
                raise StopIteration
        
    myclass = MyNumbers()
    myiter = iter(myclass)
 
    for x in myiter:
        print(x)

def generator_1():
    print("\ngenerator_1")
    def fibonacci(n):
        a, b, counter = 0, 1, 1
        while True:
            if(counter > n):
                return 
            yield a 
            a, b = b, a + b
            counter += 1
    
    f = fibonacci(10)
    while True:
        try:
            print(next(f), end=' ')
        except StopIteration:
            print()
            break


if __name__ == "__main__":
    iterator_1()
    iterator_2()
    iterator_3()
    iterator_4()

    generator_1()