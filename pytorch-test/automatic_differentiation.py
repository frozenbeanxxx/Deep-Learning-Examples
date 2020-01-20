import torch

x = torch.ones(2, 2, requires_grad=True)
print('x:\n', x)
y = x + 2
#y = y * y
zz = y.mean()
#zz = y.sum()
#zz.backward()
print(x.grad)
print('y:\n', y)
print('y.grad_fn:\n', y.grad_fn)

z = y * y * 3
out = z.mean()
print('z, out:\n', z, out)
out.backward()
print(x.grad, y.grad, z.grad)

a = torch.randn(2, 2)
print('a:\n', a)
a = ((a * 3) / (a - 1))
print(a.requires_grad, a)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


x = torch.randn(3, requires_grad=True)

y = x * 2
print('y:\n', y)
while y.data.norm() < 1000:
    print(y.data.norm())
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)


print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
    
