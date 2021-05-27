import torch

a = torch.rand(2, 3, 4, 5)

print(a)

sizes = a.size()

a = a.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)

print(a)
