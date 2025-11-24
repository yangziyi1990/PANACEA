import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
c = a @ b
print(c.shape)
