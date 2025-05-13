class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        m=10**9 + 7
        a = [0] * 26
        ass = "abcdefghijklmnopqrstuvwxyz"
        for c in s:
            a[ass.index(c)] += 1
        for i in range(t):
            b = [a[-1]] + a[:-1]
            b[1] = (b[1]+a[-1])%m
            a=b
        result = 0
        for k in a:
            result += k
        print(result%m)
        return result%m


s = "j"
t = 7517

Solution().lengthAfterTransformations(s, t)
