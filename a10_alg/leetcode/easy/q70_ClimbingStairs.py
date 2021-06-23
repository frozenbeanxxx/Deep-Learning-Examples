class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 2:
            return 2
        a, b = 1, 2
        for i in range(2, n):
            a, b = b, a+b 
        return b

if __name__ == "__main__":
    s = Solution()
    for i in range(10):
        print(i, s.climbStairs(i+1))