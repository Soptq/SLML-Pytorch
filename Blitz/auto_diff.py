import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
# print(y)
z = y * y * 3
out = z.mean()
# print(z, out)
out.backward()  # equals to out.backward(torch.tensor(1.))
# print(x.grad)   # out = 0.25 * sum( 3 * (x + 2) ^ 2 )  =>  d(out)/d(x) = 0.25 * (  6 * (x + 2) ) = 1.5 * (x + 2) = 4.5


x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)