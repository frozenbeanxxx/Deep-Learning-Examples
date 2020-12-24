class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        if len(s) <= 0:
            return 0
        sum = 0
        i = len(s) - 1
        print(i)
        while s[i] == ' ':
            i -= 1
        while i >= 0:
            if s[i] == ' ':
                return sum
            else:
                sum += 1
            i -= 1
        return sum

if __name__ == "__main__":
    s = Solution()
    l = ['hello world', '']
    for i in l:
        print(i, s.lengthOfLastWord(i))