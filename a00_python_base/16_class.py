def simple_class():
    print("\nsimple_class")
    class MyClass:
        i = 12345
        def f(self):
            return "hello world"
    x = MyClass()
    print("MyClass 类的属性 i 为：", x.i)
    print("MyClass 类的方法 f 输出为：", x.f())
    
def class_with_init():
    print("\nclass_with_init")
    class Complex:
        def __init__(self, realpart, imagpart):
            self.r = realpart
            self.i = imagpart
    x = Complex(3.0, -4.5)
    print("Complex x: ", x.r, x.i)   # 输出结果：3.0 -4.5

def replace_self():
    print("\nreplace_self")
    class Test:
        def prt(another_self):
            print(another_self)
            print(another_self.__class__)
    t = Test()
    t.prt()

def define_method():
    print("\ndefine_method")
    class People:
        name = ''
        age = 0
        __weight = 0
        def __init__(self, n, a, w):
            self.name = n
            self.age = a 
            self.__weight = w
        def speak(self):
            print("%s said i am %d" % (self.name, self.age))
    p = People('angle', 10, 30)
    p.speak()

def inherit_test():
    print("\ninherit_test")
    class People:
        name = ''
        age = 0
        __weight = 0
        def __init__(self, n, a, w):
            self.name = n
            self.age = a 
            self.__weight = w
        def speak(self):
            print("%s said i am %d" % (self.name, self.age))
    class Student(People):
        grade = ''
        def __init__(self, n, a, w, g):
            People.__init__(self, n, a, w)
            self.grade = g
        def speak(self):
            print("%s said i am %d, and read %d grade" % (self.name, self.age, self.grade))
    p = Student('angle', 10, 30, 1)
    p.speak()

def multi_inherit():
    print("\nmulti_inherit")
    class people:
        name = ''
        age = 0
        __weight = 0
        def __init__(self,n,a,w):
            self.name = n
            self.age = a
            self.__weight = w
        def speak(self):
            print("%s 说: 我 %d 岁。" %(self.name,self.age))
    class student(people):
        grade = ''
        def __init__(self,n,a,w,g):
            people.__init__(self,n,a,w)
            self.grade = g
        def speak(self):
            print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))
    class speaker():
        topic = ''
        name = ''
        def __init__(self,n,t):
            self.name = n
            self.topic = t
        def speak(self):
            print("我叫 %s，我是一个演说家，我演讲的主题是 %s"%(self.name,self.topic))
    class sample(speaker, student):
        def __init__(self, n, a, w, g, t):
            student.__init__(self, n, a, w, g)
            speaker.__init__(self, n, t)
    test = sample("Tim", 25, 80, 4, "python")
    test.speak()

def overwrite_method():
    print("\noverwrite_method")
    class Parent:        # 定义父类
        def myMethod(self):
            print ('调用父类方法')
    class Child(Parent): # 定义子类
        def myMethod(self):
            print ('调用子类方法')
    c = Child()          # 子类实例
    c.myMethod()         # 子类调用重写方法
    super(Child, c).myMethod() #用子类对象调用父类已被覆盖的方法

def operator_overload():
    print("\noperator_overload")
    class Vector:
        def __init__(self, a, b):
            self.a = a 
            self.b = b
        def __str__(self):
            return "Vector(%d, %d)" % (self.a, self.b)
        def __add__(self, othor):
            return Vector(self.a + othor.a, self.b + othor.b)
    v1 = Vector(22, 10)
    v2 = Vector(5, 2)
    print(v1 + v2)

def test_class_member():
    print("\ntest_class_member")
    class MemberCounter:
        members = 0
        def init(self):
            MemberCounter.members += 1
    m1 = MemberCounter()
    m1.init()
    print("MemberCounter.members = ", m1.members)
    m2 = MemberCounter()
    m2.init()
    print("MemberCounter.members = ", m2.members)

if __name__ == "__main__":
    simple_class()
    class_with_init()
    replace_self()
    define_method()
    inherit_test()
    multi_inherit()
    overwrite_method()
    operator_overload()
    test_class_member()