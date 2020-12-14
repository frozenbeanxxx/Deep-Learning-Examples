import string

if __name__ == "__main__":
    print("\nString Function")
    s1 = "there is a pig"
    print(f"s1 = {s1}, s1.capitalize()", s1.capitalize())
    print("s1.center(100, '*') = ", s1.center(100, '*'))
    print("l1 = s1.split() = ", s1.split())
    l1 = s1.split()
    print("' '.join(l1) = ", ' '.join(l1))