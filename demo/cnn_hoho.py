import torch
from torch.autograd import Variable


# input = torch.ones(1, 1, 5, 5)
# input = Variable(input)
# x = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, groups=1)
# out = x(input)

input = torch.ones(1, 1, 5, 5)
input = Variable(input)
x = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, groups=1)
out = x(input)

print('input: ', input)
print('output: ', out)
print('parameter: ', list(x.parameters()))