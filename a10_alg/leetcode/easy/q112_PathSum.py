# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        if not root.left and not root.right:
            return root.val == sum
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)
                

if __name__ == "__main__":
    root = TreeNode(5)
    c1 = TreeNode(4)
    c2 = TreeNode(11)
    c3 = TreeNode(2)
    root.left = c1
    c1.left = c2
    c2.left = c3

    s = Solution()
    res = s.hasPathSum(root, 22)
    print("res = ", res)
    