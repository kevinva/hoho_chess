from torchvision.models import resnet50
from thop import profile
from thop import clever_format
import torch

model = resnet50()
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input,))
flops, params = clever_format([macs, params], "%.3f")

print(f'flops={flops}, params={params}')