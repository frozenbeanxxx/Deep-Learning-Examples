import keyword
import sys

if __name__ == "__main__":
    print("hello python\n")

    print("define encode method: # -*- coding: cp-1252 -*-\n")

    print("python keywords: ", keyword.kwlist)
    print("python keywords number: ", len(keyword.kwlist))

    print("number type: int, bool, float, complex\n")

    s = "this is a string"
    print("s:", s, ", s[0] =", s[0], ", s[0-4] =", s[:4], "\n")

    print("no new line: print(x, end=' ')\n")

    print("sys argv:")
    for i in sys.argv:
        print(i)
    print("sys.path =", sys.path, "\n")
    
    print("help: python -h")

